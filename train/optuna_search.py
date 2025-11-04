import gc
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
import numpy as np
import optuna
import torch
from datasets import Dataset, load_dataset
from optuna.exceptions import TrialPruned
from optuna.integration import WeightsAndBiasesCallback
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from transformers import EarlyStoppingCallback

from trl import SFTConfig, SFTTrainer

from finetune import format_prompts


def load_raw_datasets(train_file: str, val_file: str, sample_ratio: float = 1.0):
    data_files = {"train": train_file, "validation": val_file}
    dataset_dict = load_dataset("json", data_files=data_files)
    
    train_dataset = dataset_dict["train"]
    val_dataset = dataset_dict["validation"]
    
    train_size = int(len(train_dataset) * sample_ratio)
    val_size = int(len(val_dataset) * sample_ratio)
    train_dataset = train_dataset.shuffle(seed=42).select(range(train_size))
    val_dataset = val_dataset.shuffle(seed=42).select(range(val_size))
    
    return train_dataset, val_dataset


def initialize_model(
    model_name: str,
    max_seq_length: int,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    seed: int,
):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        max_lora_rank=lora_rank,
        attn_implementation="flash_attention_2",
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="qwen25",
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def save_best_params(output_dir: Path, study: optuna.Study):
    best_params = dict(study.best_trial.params)
    best_params["effective_batch_size"] = study.best_trial.user_attrs["effective_batch_size"]
    best_params["best_eval_loss"] = study.best_value
    best_params["trial_number"] = study.best_trial.number

    best_metrics = study.best_trial.user_attrs.get("eval_metrics")
    if best_metrics:
        best_params["eval_metrics"] = best_metrics
        best_params["eval_perplexity"] = math.exp(best_metrics["eval_loss"])

    best_file = output_dir / "best_hparams.json"
    with best_file.open("w", encoding="utf-8") as fp:
        json.dump(best_params, fp, indent=2)

    return best_file


def objective(
    trial: optuna.Trial,
    model_name: str,
    seed: int,
    raw_train_dataset: Dataset,
    raw_val_dataset: Dataset,
    output_path: Path,
    num_train_epochs: float,
    eval_steps: int,
    logging_steps: int,
    save_strategy: str,
    save_steps: int,
    save_total_limit: int,
    early_stopping_patience: int,
    use_wandb: bool,
    log_individual_trial_wandb: bool,
) -> float:
    # Define the Optuna search space for QLoRA
    learning_rate = trial.suggest_float("learning_rate", 7e-5, 1.5e-4, log=True)
    per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32])
    gradient_accumulation_steps = trial.suggest_categorical("gradient_accumulation_steps", [1, 2])
    lora_dropout = trial.suggest_categorical("lora_dropout", [0.0, 0.05])
    max_seq_length = trial.suggest_categorical("max_seq_length", [512, 1024])

    effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps
    trial.set_user_attr("effective_batch_size", effective_batch_size)

    params = {
        "learning_rate": learning_rate,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_ratio": 0.05,
        "weight_decay": 0.01,
        "lr_scheduler_type": "constant_with_warmup",
        "lora_rank": 64,
        "lora_alpha": 16,
        "lora_dropout": lora_dropout,
        "max_seq_length": max_seq_length,
        "optimizer": "adamw_8bit",
        "max_grad_norm": 0.3,
    }

    trial_dir = output_path / f"trial-{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    model = None
    tokenizer = None
    trainer = None

    try:
        model, tokenizer = initialize_model(
            model_name=model_name,
            max_seq_length=params["max_seq_length"],
            lora_rank=params["lora_rank"],
            lora_alpha=params["lora_alpha"],
            lora_dropout=params["lora_dropout"],
            seed=seed + trial.number,
        )

        # Apply chat formatting to datasets for SFTTrainer
        remove_cols_train = [c for c in raw_train_dataset.column_names if c != "text"]
        remove_cols_val = [c for c in raw_val_dataset.column_names if c != "text"]

        train_dataset = raw_train_dataset.map(
            lambda batch: format_prompts(batch, tokenizer),
            batched=True,
            remove_columns=remove_cols_train,
        )
        val_dataset = raw_val_dataset.map(
            lambda batch: format_prompts(batch, tokenizer),
            batched=True,
            remove_columns=remove_cols_val,
        )

        # Build SFT training arguments for this trial
        per_device_eval_batch_size = min(params["per_device_train_batch_size"] * 2, 8)
        report_to = "wandb" if (use_wandb and log_individual_trial_wandb) else "none"

        training_args = SFTConfig(
            output_dir=str(trial_dir),
            per_device_train_batch_size=params["per_device_train_batch_size"],
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=params["gradient_accumulation_steps"],
            num_train_epochs=num_train_epochs,
            learning_rate=params["learning_rate"],
            warmup_ratio=params["warmup_ratio"],
            weight_decay=params["weight_decay"],
            lr_scheduler_type=params["lr_scheduler_type"],
            max_seq_length=params["max_seq_length"],
            optim=params["optimizer"],
            max_grad_norm=params["max_grad_norm"],
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=logging_steps,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy=save_strategy,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            dataset_text_field="text",
            packing=False,
            remove_unused_columns=False,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=report_to,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            seed=3407,
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
        )

        trainer.train()
        metrics = trainer.evaluate()

        eval_loss = metrics["eval_loss"]
        trial.set_user_attr("eval_metrics", metrics)
        return eval_loss

    except (RuntimeError, ValueError) as exc:
        message = str(exc).lower()
        if "out of memory" in message:
            trial.set_user_attr("oom", True)
            raise TrialPruned("CUDA OOM") from exc
        raise

    finally:
        if trainer:
            trainer.model = None
            del trainer
        if model:
            del model
        if tokenizer:
            del tokenizer

        torch.cuda.empty_cache()
        gc.collect()


def main(
    train_file: str = "dataset/train.json",
    val_file: str = "dataset/val.json",
    model_name: str = "unsloth/Qwen2.5-72B-Instruct-bnb-4bit",
    output_dir: str = "train_results",
    study_name: str = "sft-optuna-search",
    n_trials: int = 30,
    timeout: Optional[int] = None,
    seed: int = 3407,
    num_train_epochs: float = 1.0,
    eval_steps: int = 100,
    logging_steps: int = 10,
    save_strategy: str = "no",
    save_steps: int = 200,
    save_total_limit: int = 2,
    early_stopping_patience: int = 3,
    use_wandb: bool = True,
    log_individual_trial_wandb: bool = False,
    sample_ratio: float = 0.1,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=TPESampler(multivariate=True, seed=seed),
        pruner=SuccessiveHalvingPruner(
            min_resource=1,
            reduction_factor=3,
            min_early_stopping_rate=0,
        ),
        storage=f"sqlite:///{output_path / 'study.db'}",
        load_if_exists=True,
    )

    # Setup W&B if enabled
    if use_wandb:
        os.environ["WANDB_PROJECT"] = "cyber-llm-optuna"

    optuna_callbacks = []
    wandb_callback = None
    wandb_module = None
    if use_wandb and not log_individual_trial_wandb:
        try:
            import wandb as wandb_module  # type: ignore
        except ImportError as exc:
            raise RuntimeError("use_wandb=True requires the 'wandb' package") from exc

        wandb_callback = WeightsAndBiasesCallback(
            metric_name="eval_loss",
            wandb_kwargs={"project": "cyber-llm-optuna", "reinit": True},
        )
        optuna_callbacks.append(wandb_callback)

    raw_train_dataset, raw_val_dataset = load_raw_datasets(train_file, val_file, sample_ratio)

    study.optimize(
        lambda trial: objective(
            trial=trial,
            model_name=model_name,
            seed=seed,
            raw_train_dataset=raw_train_dataset,
            raw_val_dataset=raw_val_dataset,
            output_path=output_path,
            num_train_epochs=num_train_epochs,
            eval_steps=eval_steps,
            logging_steps=logging_steps,
            save_strategy=save_strategy,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            early_stopping_patience=early_stopping_patience,
            use_wandb=use_wandb,
            log_individual_trial_wandb=log_individual_trial_wandb,
        ),
        n_trials=n_trials,
        timeout=timeout,
        callbacks=optuna_callbacks,
        n_jobs=1,
    )

    if wandb_callback is not None and wandb_module is not None:
        active_run = wandb_module.run
        if active_run:
            active_run.summary["best_eval_loss"] = study.best_value
            active_run.summary["best_trial"] = study.best_trial.number
            best_metrics = study.best_trial.user_attrs.get("eval_metrics")
            if best_metrics:
                for key, value in best_metrics.items():
                    active_run.summary[f"best/{key}"] = value
            wandb_module.finish()

    save_best_params(output_path, study)


if __name__ == "__main__":
    """pixi run python train/optuna_search.py --n_trials=20
       pixi run python train/optuna_search.py --sample_ratio=0.1 --n_trials=10
    """
        import fire
    fire.Fire(main)
