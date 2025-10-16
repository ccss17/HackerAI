#!/usr/bin/env python3
"""Optuna hyperparameter search for GPT-OSS-120B LoRA-based SFT training.

Usage (Fire CLI):

    pixi run python train/optuna_search_gpt_oss.py \
        --train_file=train/data/train.json \
        --val_file=train/data/val.json \
        --training_mode=qlora

    pixi run python train/optuna_search_gpt_oss.py \
        --model_name=unsloth/gpt-oss-120b-unsloth-bnb-4bit \
        --training_mode=lora \
        --n_trials=10

    pixi run python train/optuna_search_gpt_oss.py \
        --sample_ratio=0.1 \
        --n_trials=20

Set ``--training_mode=qlora`` (default) to load the base model in 4-bit during evaluation,
or ``--training_mode=lora`` to evaluate LoRA adapters on a full-precision base model.
Use ``--sample_ratio`` to sample a fraction of the dataset for faster hyperparameter search
(default 0.1 for 10% sampling). The best hyperparameters (plus the training mode) are written to
``train/optuna_results/best_hparams_gpt_oss.json`` for reuse by ``train/finetune_gpt_oss.py``.
"""

from __future__ import annotations

import gc
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Environment configuration (must run before importing unsloth)
# ---------------------------------------------------------------------------
# os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
# os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
# os.environ.setdefault("TORCH_LOGS", "-dynamo")
# os.environ.setdefault("TORCHDYNAMO_VERBOSE", "0")
# os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
# os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
# os.environ.setdefault("CUDA_VERSION", "12.0")

# Disable Unsloth fused kernels that rely on Triton (ensure string "1")
# os.environ.setdefault("UNSLOTH_DISABLE_OPTIMIZED_LOSS", "1")
# os.environ.setdefault("UNSLOTH_DISABLE_FAST_GENERATION", "1")

# Skip Unsloth fused cut cross-entropy kernels (force pure PyTorch path)
# os.environ.setdefault("UNSLOTH_ENABLE_CCE", "0")

from unsloth import FastLanguageModel, is_bfloat16_supported
import fire
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


def print_section(title: str, *, leading_newline: bool = True) -> None:
    """Pretty-print a section header."""

    if leading_newline:
        print()
    print(title)
    print("=" * len(title))


def get_gpu_info() -> Tuple[str, str]:
    """Return (gpu_name, gpu_memory_gib_str) for primary device.

    If CUDA is unavailable, returns ("cpu", "NA").
    """

    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "cuda_device"
        try:
            props = torch.cuda.get_device_properties(0)
            mem_gib = int(round(props.total_memory / (1024**3)))
            mem_str = f"{mem_gib}GiB"
        except Exception:
            mem_str = "NA"
        # Sanitize spaces for readability
        return (name.replace(" ", "_"), mem_str)
    return ("cpu", "NA")


def format_prompts(
    examples: Dict[str, Any], tokenizer, model_name: str = None
) -> Dict[str, Any]:
    """Format instruction/response pairs using the tokenizer chat template for GPT-OSS."""

    instructions = examples.get("instruction", [])
    responses = examples.get("response", [])
    texts = []

    for instruction, response in zip(instructions, responses):
        # GPT-OSS uses standard role/content format
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]

        text = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)

    return {"text": texts}


def load_raw_datasets(
    train_file: str, val_file: str, sample_ratio: float = 1.0
) -> Tuple[Dataset, Dataset]:
    """Load raw JSON datasets for training and validation.
    
    Args:
        train_file: Path to training data file
        val_file: Path to validation data file  
        sample_ratio: Fraction of data to sample (0.0 to 1.0). Default 1.0 (use all data).
    """

    data_files = {"train": train_file, "validation": val_file}
    dataset_dict = load_dataset("json", data_files=data_files)
    
    train_dataset = dataset_dict["train"]
    val_dataset = dataset_dict["validation"]
    
    # Sample data if sample_ratio < 1.0
    if sample_ratio < 1.0:
        train_size = int(len(train_dataset) * sample_ratio)
        val_size = int(len(val_dataset) * sample_ratio)
        
        # Use random sampling with fixed seed for reproducibility
        train_dataset = train_dataset.shuffle(seed=42).select(range(train_size))
        val_dataset = val_dataset.shuffle(seed=42).select(range(val_size))
    
    return train_dataset, val_dataset


def prepare_sft_datasets(
    train_dataset: Dataset,
    val_dataset: Dataset,
    tokenizer,
    model_name: str = None,
) -> Tuple[Dataset, Dataset]:
    """Apply chat formatting to datasets for SFTTrainer."""

    remove_cols_train = [c for c in train_dataset.column_names if c != "text"]
    remove_cols_val = [c for c in val_dataset.column_names if c != "text"]

    formatted_train = train_dataset.map(
        lambda batch: format_prompts(batch, tokenizer, model_name),
        batched=True,
        remove_columns=remove_cols_train,
    )
    formatted_val = val_dataset.map(
        lambda batch: format_prompts(batch, tokenizer, model_name),
        batched=True,
        remove_columns=remove_cols_val,
    )

    return formatted_train, formatted_val


def sample_hyperparameters(
    trial: optuna.Trial, training_mode: str
) -> Dict[str, Any]:
    """Define the Optuna search space for GPT-OSS."""

    learning_rate = trial.suggest_float(
        "learning_rate", 1e-4, 3e-4, log=True
    )
    per_device_train_batch_size = trial.suggest_categorical(
        "per_device_train_batch_size", [1, 2, 4]  # Reduced for 120B model
    )
    gradient_accumulation_steps = trial.suggest_categorical(
        "gradient_accumulation_steps", [1, 2, 4]
    )
    warmup_ratio = 0.05
    weight_decay = 0.01
    lr_scheduler_type = "linear"  # GPT-OSS default
    lora_rank = trial.suggest_categorical("lora_rank", [8, 16, 32, 64])  # GPT-OSS base is 8
    lora_alpha = trial.suggest_categorical("lora_alpha", [8, 16, 32])  # GPT-OSS uses 16
    lora_dropout = trial.suggest_categorical("lora_dropout", [0.0, 0.05])
    max_seq_length = trial.suggest_categorical("max_seq_length", [512, 1024, 1536])  # User specified
    optimizer = "adamw_8bit"
    max_grad_norm = 0.3

    effective_batch_size = (
        per_device_train_batch_size * gradient_accumulation_steps
    )
    trial.set_user_attr("effective_batch_size", effective_batch_size)

    params = {
        "learning_rate": learning_rate,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "lr_scheduler_type": lr_scheduler_type,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "max_seq_length": max_seq_length,
        "optimizer": optimizer,
        "max_grad_norm": max_grad_norm,
    }

    if training_mode == "lora":
        params["load_in_4bit"] = False

    return params


def configure_training_args(
    trial_number: int,
    params: Dict[str, Any],
    output_dir: Path,
    num_train_epochs: float,
    eval_steps: int,
    logging_steps: int,
    save_strategy: str,
    save_steps: int,
    save_total_limit: int,
    use_wandb: bool,
    log_individual_trial_wandb: bool,
    study_name: str,
    training_mode: str,
) -> SFTConfig:
    """Build SFT training arguments for a given trial."""

    per_device_eval_batch_size = min(
        params["per_device_train_batch_size"] * 2, 4  # Reduced for 120B model
    )

    report_to = (
        "wandb" if (use_wandb and log_individual_trial_wandb) else "none"
    )
    run_name = None
    if report_to == "wandb":
        gpu_name, gpu_mem = get_gpu_info()
        # params does not include model_name here; use study name for uniqueness
        run_name = f"{study_name}/{gpu_name}/{gpu_mem}"

    training_args = SFTConfig(
        output_dir=str(output_dir),
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
        run_name=run_name,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        seed=3407,
    )

    return training_args


def initialize_model(
    model_name: str,
    max_seq_length: int,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    seed: int,
    training_mode: str,
) -> Tuple[Any, Any]:
    """Load the base model with LoRA adapters and return (model, tokenizer)."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=training_mode == "qlora",
        max_lora_rank=lora_rank,
    )

    # GPT-OSS uses native chat template - no custom setup needed
    # The tokenizer will automatically use the correct GPT-OSS chat template

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    model.config.use_cache = False

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def save_best_params(output_dir: Path, study: optuna.Study) -> Path:
    """Persist best hyperparameters for later reuse."""

    best_params = dict(study.best_trial.params)
    best_params["effective_batch_size"] = study.best_trial.user_attrs.get(
        "effective_batch_size"
    )
    best_params["best_eval_loss"] = study.best_value
    best_params["trial_number"] = study.best_trial.number

    best_metrics = study.best_trial.user_attrs.get("eval_metrics")
    if best_metrics is not None:
        best_params["eval_metrics"] = best_metrics
        if isinstance(best_metrics, dict) and "eval_loss" in best_metrics:
            loss = best_metrics["eval_loss"]
            if loss is not None and not math.isnan(loss):
                best_params["eval_perplexity"] = math.exp(loss)

    best_file = output_dir / "best_hparams_gpt_oss.json"
    with best_file.open("w", encoding="utf-8") as fp:
        json.dump(best_params, fp, indent=2)

    return best_file


def main(
    train_file: str = "train/data/train.json",
    val_file: str = "train/data/val.json",
    model_name: str = "unsloth/gpt-oss-120b-unsloth-bnb-4bit",
    output_dir: str = "train/optuna_results",
    study_name: str = "gpt-oss-optuna-search",
    storage: Optional[str] = None,
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
    wandb_project: str = "cyber-llm-gpt-oss-optuna",
    wandb_entity: Optional[str] = None,
    wandb_group: Optional[str] = None,
    log_individual_trial_wandb: bool = False,
    training_mode: str = "qlora",
    sample_ratio: float = 0.1,
) -> None:
    """Run Optuna hyperparameter search for GPT-OSS SFT fine-tuning."""

    np.random.seed(seed)
    torch.manual_seed(seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Default storage to SQLite inside output directory
    if storage is None:
        storage = f"sqlite:///{output_path / 'study_gpt_oss.db'}"

    print_section("GPT-OSS Optuna Configuration", leading_newline=False)
    print(f"Study name: {study_name}")
    print(f"Storage: {storage}")
    print(f"Trials: {n_trials}")
    print(f"Timeout: {timeout}")
    print(f"Model: {model_name}")
    print(f"Train file: {train_file}")
    print(f"Validation file: {val_file}")
    print(f"Output directory: {output_path}")
    print(f"W&B enabled: {use_wandb}")
    print(f"Log per-trial W&B: {log_individual_trial_wandb}")
    print(f"Training mode: {training_mode}")

    print_section("Loading datasets")
    raw_train_dataset, raw_val_dataset = load_raw_datasets(
        train_file, val_file, sample_ratio
    )
    print(f"Train samples: {len(raw_train_dataset)}")
    print(f"Validation samples: {len(raw_val_dataset)}")
    if sample_ratio < 1.0:
        print(f"Sampling ratio: {sample_ratio:.1%}")

    sampler = TPESampler(multivariate=True, seed=seed)
    pruner = SuccessiveHalvingPruner(
        min_resource=1,
        reduction_factor=3,
        min_early_stopping_rate=0,
    )

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    optuna_callbacks = []
    wandb_callback: Optional[WeightsAndBiasesCallback] = None
    wandb_module = None
    if use_wandb and not log_individual_trial_wandb:
        try:
            import wandb as wandb_module  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "use_wandb=True requires the 'wandb' package to be installed."
            ) from exc

        wandb_kwargs: Dict[str, Any] = {
            "project": wandb_project,
            "reinit": True,
        }
        # Set run name as "{MODEL_NAME}/{GPU_NAME}/{GPU_MEMORY}"
        gpu_name, gpu_mem = get_gpu_info()
        wandb_kwargs["name"] = f"{model_name}/{gpu_name}/{gpu_mem}"
        if wandb_entity:
            wandb_kwargs["entity"] = wandb_entity
        if wandb_group:
            wandb_kwargs["group"] = wandb_group

        wandb_callback = WeightsAndBiasesCallback(
            metric_name="eval_loss",
            wandb_kwargs=wandb_kwargs,
        )
        optuna_callbacks.append(wandb_callback)

    if use_wandb and log_individual_trial_wandb:
        os.environ.setdefault("WANDB_PROJECT", wandb_project)
        if wandb_entity:
            os.environ.setdefault("WANDB_ENTITY", wandb_entity)

    def objective(trial: optuna.Trial) -> float:
        params = sample_hyperparameters(trial, training_mode)

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
                training_mode=training_mode,
            )

            train_dataset, val_dataset = prepare_sft_datasets(
                raw_train_dataset, raw_val_dataset, tokenizer, model_name
            )

            training_args = configure_training_args(
                trial_number=trial.number,
                params=params,
                output_dir=trial_dir,
                num_train_epochs=num_train_epochs,
                eval_steps=eval_steps,
                logging_steps=logging_steps,
                save_strategy=save_strategy,
                save_steps=save_steps,
                save_total_limit=save_total_limit,
                use_wandb=use_wandb,
                log_individual_trial_wandb=log_individual_trial_wandb,
                study_name=study_name,
                training_mode=training_mode,
            )

            callbacks = []
            if early_stopping_patience > 0:
                callbacks.append(
                    EarlyStoppingCallback(
                        early_stopping_patience=early_stopping_patience
                    )
                )

            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                processing_class=tokenizer,
                callbacks=callbacks,
            )

            trainer.train()
            metrics = trainer.evaluate()

            eval_loss = metrics.get("eval_loss")
            if eval_loss is None or math.isnan(eval_loss):
                raise TrialPruned("Objective returned NaN")

            trial.set_user_attr("eval_metrics", metrics)
            return float(eval_loss)

        except (RuntimeError, ValueError) as exc:
            # Handle common CUDA OOM and GPU memory issues
            message = str(exc).lower()
            if (
                "out of memory" in message
                or "cuda error" in message
                or "dispatched on the cpu" in message
                or "gpu ram" in message
                or "some modules are dispatched" in message
            ):
                trial.set_user_attr("oom", True)
                trial.set_user_attr("error_type", "gpu_memory")
                raise TrialPruned("GPU Memory Error") from exc
            raise

        finally:
            if trainer is not None:
                trainer.model = None
                del trainer
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            if use_wandb and log_individual_trial_wandb:
                try:
                    import wandb  # type: ignore

                    if wandb.run is not None:
                        wandb.finish()
                except ImportError:
                    pass

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    print_section("Starting Optuna search")
    print(f"Training mode: {training_mode}")
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        callbacks=optuna_callbacks,
        n_jobs=1,
    )

    if wandb_callback is not None and wandb_module is not None:
        active_run = wandb_module.run
        if active_run is not None:
            active_run.summary["best_eval_loss"] = study.best_value
            active_run.summary["best_trial"] = study.best_trial.number
            active_run.summary["best_effective_batch_size"] = (
                study.best_trial.user_attrs.get("effective_batch_size")
            )
            best_metrics = study.best_trial.user_attrs.get("eval_metrics")
            if isinstance(best_metrics, dict):
                for key, value in best_metrics.items():
                    if isinstance(value, (int, float)):
                        active_run.summary[f"best/{key}"] = value
            wandb_module.finish()

    print_section("Best trial summary")
    print(f"Trial number: {study.best_trial.number}")
    print(f"Best eval loss: {study.best_value:.4f}")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

    best_file = save_best_params(output_path, study)
    print(f"\nBest hyperparameters written to: {best_file}")

    best_metrics = study.best_trial.user_attrs.get("eval_metrics", {})
    if isinstance(best_metrics, dict):
        print_section("Validation metrics")
        for key, value in best_metrics.items():
            print(f"  {key}: {value}")

    if use_wandb and log_individual_trial_wandb:
        try:
            import wandb  # type: ignore

            if wandb.run is not None:
                wandb.finish()
        except ImportError:
            pass

    print(
        "\nNext: run train/finetune_gpt_oss.py with the saved hyperparameters for a full training run."
    )


if __name__ == "__main__":
    fire.Fire(main)
