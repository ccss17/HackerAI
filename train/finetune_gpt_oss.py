#!/usr/bin/env python3
"""
GPT-OSS-120B SFT fine-tuning (LoRA / QLoRA).

Choose training mode (Fire CLI via --training_mode={qlora,lora})
Default is QLoRA (4-bit quantized base model). Set --training_mode=lora to load the
model in full precision while still applying LoRA adapters.

QLoRA quick start:

    pixi run python train/finetune_gpt_oss.py \
        --model_name=unsloth/gpt-oss-120b-unsloth-bnb-4bit \
        --lora_rank=8 \
        --lora_alpha=16 \
        --lora_dropout=0.0 \
        --batch_size=1 \
        --gradient_accumulation_steps=4

Notes:
- QLoRA path: base model loaded in 4-bit (`load_in_4bit=True`), memory efficient.
- LoRA path: set `--training_mode=lora` to load the model in fp16/bf16 while keeping LoRA.
- Increase `lora_rank`, `lora_alpha`, and `gradient_accumulation_steps` for deeper tuning;
  adjust `learning_rate` accordingly (e.g., 1e-4 to 3e-4 for GPT-OSS).
- The script reads `train/optuna_results/best_hparams_gpt_oss.json` by default; override with
  CLI flags to experiment manually.
- Resume training or stitch full weights via the usual Trainer checkpoints / LoRA exports.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional

from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
import wandb
import fire
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import EarlyStoppingCallback
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Disable PyTorch Dynamo compilation to avoid data-dependent branching errors
# os.environ["TORCH_COMPILE_DISABLE"] = "1"
# os.environ["TORCHDYNAMO_DISABLE"] = "1"
# os.environ["TORCH_LOGS"] = "-dynamo"  # Disable dynamo logging
# os.environ["TORCHDYNAMO_VERBOSE"] = "0"  # Disable verbose dynamo output

# Fix CUDA version detection for Triton
# os.environ["CUDA_HOME"] = "/usr/local/cuda"
# os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"
# os.environ["CUDA_VERSION"] = "12.0"  # Set a compatible CUDA version for Triton

# Disable optimized loss functions that cause Triton issues
# os.environ["UNSLOTH_DISABLE_OPTIMIZED_LOSS"] = "1"


DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
DEFAULT_OPTIMIZER = "adamw_8bit"


def get_target_modules_key(target_modules: List[str]) -> str:
    """Create a compact key describing the selected LoRA target modules."""

    base_names = [module.split("_")[0] for module in target_modules]
    return "-".join(base_names)


DEFAULT_TARGET_MODULES_KEY = get_target_modules_key(DEFAULT_TARGET_MODULES)


def get_device_descriptor() -> str:
    """Summarize the primary training device for logging purposes."""

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        names = []
        for idx in range(device_count):
            try:
                name = torch.cuda.get_device_name(idx)
            except RuntimeError:
                name = "cuda_device"
            names.append(name.replace(" ", "_"))

        unique_names = sorted(set(names))
        descriptor = "+".join(unique_names)
        if device_count > 1:
            descriptor = f"{descriptor}x{device_count}"
        return f"cuda:{descriptor}"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend and mps_backend.is_available():
        return "mps"

    xpu_backend = getattr(torch.backends, "xpu", None)
    if xpu_backend and xpu_backend.is_available():
        return "xpu"

    return "cpu"


def generate_run_name(
    *,
    training_mode: str,
    max_seq_length: int,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    effective_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    num_epochs: float,
    warmup_ratio: float,
    weight_decay: float,
    optimizer: str,
    lr_scheduler_type: str,
    target_modules_key: str,
    device_descriptor: str,
) -> str:
    """Construct a descriptive run name for W&B and Trainer logs."""

    sanitized_device = device_descriptor.replace(" ", "_")
    return (
        f"mode={training_mode};sq={max_seq_length};r={lora_rank};a={lora_alpha};"
        f"d={lora_dropout};bs={effective_batch_size};ga={gradient_accumulation_steps};"
        f"lr={learning_rate:.0e};e={num_epochs};w={warmup_ratio};wd={weight_decay};"
        f"opt={optimizer};sched={lr_scheduler_type};tm={target_modules_key}"
        f" [{sanitized_device}]"
    )


def setup_wandb(
    model_name: str,
    dataset_size: int,
    lora_rank: int,
    max_seq_length: int,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    gradient_accumulation_steps: int,
    weight_decay: float,
    training_mode: str,
    warmup_ratio: float,
    lr_scheduler_type: str,
    disable_wandb: bool = False,
    wandb_project: str = "cyber-llm-gpt-oss",
    wandb_run_name: str = None,
):
    """Initialize Weights & Biases logging."""
    if disable_wandb:
        print("W&B logging disabled")
        return

    try:
        # Generate run name if not provided
        if not wandb_run_name:
            effective_batch_size = batch_size * gradient_accumulation_steps
            wandb_run_name = generate_run_name(
                training_mode=training_mode,
                max_seq_length=max_seq_length,
                lora_rank=lora_rank,
                lora_alpha=lora_rank,
                lora_dropout=0.0,
                effective_batch_size=effective_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                warmup_ratio=warmup_ratio,
                weight_decay=weight_decay,
                optimizer=DEFAULT_OPTIMIZER,
                lr_scheduler_type=lr_scheduler_type,
                target_modules_key=DEFAULT_TARGET_MODULES_KEY,
                device_descriptor=get_device_descriptor(),
            )

        print(f"Initializing W&B project: {wandb_project}")
        print(f"W&B run name: {wandb_run_name}")
        print("W&B will prompt you to login if needed...")

        # Initialize W&B (will prompt for login if not authenticated)
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "model": model_name,
                "lora_rank": lora_rank,
                "max_seq_length": max_seq_length,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "weight_decay": weight_decay,
                "training_mode": training_mode,
                "warmup_ratio": warmup_ratio,
                "lr_scheduler_type": lr_scheduler_type,
                "dataset_size": dataset_size,
                "flash_attention": "detecting",  # Will be updated based on what actually loads
            },
        )
        print("W&B initialized successfully!")

    except Exception as e:
        print(f"WARNING: Failed to initialize W&B: {e}")
        print("Continuing without W&B logging...")


def load_model_and_tokenizer(
    model_name: str,
    max_seq_length: int,
    lora_rank: int,
    lora_alpha: Optional[int] = None,
    lora_dropout: float = 0.0,
    training_mode: str = "qlora",
):
    """Load model and tokenizer with LoRA configuration."""
    print(f"Loading model: {model_name}")
    print(f"Max sequence length: {max_seq_length}")
    print(f"LoRA rank: {lora_rank}")
    if lora_alpha is not None:
        print(f"LoRA alpha: {lora_alpha}")
    print(f"LoRA dropout: {lora_dropout}")

    # Disable torch.compile to avoid Dynamo issues
    torch._dynamo.config.disable = True

    # GPT-OSS uses default attention mechanism (no Flash Attention needed)
    # Following the official Unsloth GPT-OSS guide
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=training_mode == "qlora",
        max_lora_rank=lora_rank,
    )
    
    used_attn_impl = "default"
    print("Using default attention implementation (recommended for GPT-OSS)")

    # GPT-OSS uses native chat template - no custom setup needed
    # The tokenizer will automatically use the correct GPT-OSS chat template

    # Configure LoRA adapters for the quantized model
    print("Configuring LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=DEFAULT_TARGET_MODULES,
        lora_alpha=lora_alpha or lora_rank,
        lora_dropout=lora_dropout,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # Disable optimized loss to avoid Triton issues
    model.config.use_cache = False

    print(
        f"Model loaded successfully with {model.num_parameters()} parameters"
    )
    return model, tokenizer, used_attn_impl


def format_prompts(
    examples: Dict[str, List[str]], tokenizer, model_name: str = None
) -> Dict[str, List[str]]:
    """Format instruction-response pairs for training with GPT-OSS."""
    instructions = examples["instruction"]
    responses = examples["response"]
    texts = []

    for instruction, response in zip(instructions, responses):
        # GPT-OSS uses standard role/content format
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
        
        # Use the tokenizer's chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)

    return {"text": texts}


def load_datasets(train_file: str, val_file: str):
    """Load training and validation datasets."""
    print("Loading datasets...")

    # Load from local JSON files
    train_dataset = load_dataset("json", data_files=train_file, split="train")
    eval_dataset = load_dataset("json", data_files=val_file, split="train")

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")

    return train_dataset, eval_dataset


def setup_training_config(
    model_name: str,
    max_seq_length: int,
    num_epochs: float,
    batch_size: int,
    learning_rate: float,
    gradient_accumulation_steps: int,
    output_dir: str,
    disable_wandb: bool = False,
    early_stopping_patience: int = 3,
    eval_steps: int = 100,
    save_steps: int = 100,
    logging_steps: int = 10,
    save_total_limit: int = 3,
    warmup_ratio: float = 0.05,  # GPT-OSS default
    weight_decay: float = 0.01,  # GPT-OSS default
    max_grad_norm: float = 0.3,
    lr_scheduler_type: str = "linear",  # GPT-OSS default
    optimizer: str = DEFAULT_OPTIMIZER,
    run_name: Optional[str] = None,
    training_mode: str = "qlora",
    lora_rank: int = 8,  # GPT-OSS default
    lora_alpha: Optional[int] = None,
    lora_dropout: float = 0.0,  # GPT-OSS default
):
    """Setup training configuration."""
    # Generate output directory name based on model
    model_short = model_name.split("/")[-1].replace("-", "_")
    if not output_dir:
        output_dir = f"outputs/{model_short}_gpt_oss_sft"

    effective_batch_size = batch_size * gradient_accumulation_steps
    inferred_lora_alpha = lora_alpha or lora_rank
    if run_name is None:
        run_name = generate_run_name(
            training_mode=training_mode,
            max_seq_length=max_seq_length,
            lora_rank=lora_rank,
            lora_alpha=inferred_lora_alpha,
            lora_dropout=lora_dropout,
            effective_batch_size=effective_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            optimizer=optimizer,
            lr_scheduler_type=lr_scheduler_type,
            target_modules_key=DEFAULT_TARGET_MODULES_KEY,
            device_descriptor=get_device_descriptor(),
        )

    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        optim=optimizer,
        lr_scheduler_type=lr_scheduler_type,
        seed=3407,
        report_to="wandb" if not disable_wandb else "none",
        run_name=run_name,
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        packing=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        # Early stopping configuration
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    return training_args


def main(
    model_name: str = "unsloth/gpt-oss-120b-unsloth-bnb-4bit",
    resume_from_checkpoint: str = None,
    lora_rank: int = 8,  # GPT-OSS default
    lora_alpha: int = 16,  # GPT-OSS default
    lora_dropout: float = 0.0,  # GPT-OSS default
    max_seq_length: int = 1024,  # GPT-OSS default
    num_epochs: float = 2,
    batch_size: int = 1,  # Reduced for 120B model
    gradient_accumulation_steps: int = 4,  # Increased to maintain effective batch size
    learning_rate: float = 2e-4,  # GPT-OSS default
    warmup_ratio: float = 0.05,  # GPT-OSS default
    weight_decay: float = 0.01,  # GPT-OSS default
    max_grad_norm: float = 0.3,
    lr_scheduler_type: str = "linear",  # GPT-OSS default
    hparams_path: str = "train/optuna_results/best_hparams_gpt_oss.json",
    train_file: str = "train/data/train.json",
    val_file: str = "train/data/val.json",
    training_mode: str = "qlora",
    output_dir: str = None,
    wandb_project: str = "cyber-llm-gpt-oss",
    wandb_run_name: str = None,
    disable_wandb: bool = False,
    # Early stopping and step configuration
    early_stopping_patience: int = 3,
    eval_steps: int = 100,
    save_steps: int = 100,
    logging_steps: int = 10,
    save_total_limit: int = 2,
):
    """
    Main SFT training function with Fire CLI interface.

    Args:
        model_name: Model to fine-tune (e.g., unsloth/gpt-oss-120b-unsloth-bnb-4bit)
        resume_from_checkpoint: Path to checkpoint to resume from
        lora_rank: LoRA rank (8 for GPT-OSS base, 16-64 for better quality)
        max_seq_length: Maximum sequence length
        num_epochs: Number of training epochs
        batch_size: Training batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate
        max_grad_norm: Maximum gradient norm for gradient clipping
        output_dir: Output directory for checkpoints and final model
        wandb_project: Weights & Biases project name
        wandb_run_name: W&B run name (auto-generated if not provided)
        disable_wandb: Disable Weights & Biases logging
        early_stopping_patience: Number of evaluations to wait before early stopping (0 to disable)
        eval_steps: Number of steps between evaluations
        save_steps: Number of steps between checkpoint saves
        logging_steps: Number of steps between logging
        save_total_limit: Maximum number of checkpoints to keep
    """
    print("=" * 60)
    print("GPT-OSS-120B SFT Fine-Tuning Script (Fire CLI)")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"LoRA Rank: {lora_rank}")
    if lora_alpha is not None:
        print(f"LoRA Alpha: {lora_alpha}")
    print(f"LoRA Dropout: {lora_dropout}")
    print(f"Max Sequence Length: {max_seq_length}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Early Stopping Patience: {early_stopping_patience}")
    print(f"Eval Steps: {eval_steps}")
    print(f"Save Steps: {save_steps}")
    print(f"Logging Steps: {logging_steps}")
    print(f"Save Total Limit: {save_total_limit}")
    print(f"Training Mode: {training_mode}")
    if resume_from_checkpoint:
        print(f"Resuming from: {resume_from_checkpoint}")
    print("=" * 60)

    # Load datasets
    if hparams_path:
        hparams_file = Path(hparams_path)
        if hparams_file.is_file():
            print(f"Loading hyperparameters from: {hparams_file}")
        else:
            print(
                f"Hyperparameter file not found at {hparams_file}. Using CLI/default values."
            )
            hparams_file = None

        if hparams_file:
            with hparams_file.open("r", encoding="utf-8") as fp:
                best_hparams = json.load(fp)

            # Handle nested format where params are under "params" key
            if "params" in best_hparams:
                best_hparams = best_hparams["params"]

            learning_rate = best_hparams.get("learning_rate", learning_rate)
            batch_size = best_hparams.get(
                "per_device_train_batch_size", batch_size
            )
            gradient_accumulation_steps = best_hparams.get(
                "gradient_accumulation_steps", gradient_accumulation_steps
            )
            warmup_ratio = best_hparams.get("warmup_ratio", warmup_ratio)
            weight_decay = best_hparams.get("weight_decay", weight_decay)
            max_grad_norm = best_hparams.get("max_grad_norm", max_grad_norm)
            lr_scheduler_type = best_hparams.get(
                "lr_scheduler_type", lr_scheduler_type
            )
            lora_rank = best_hparams.get("lora_rank", lora_rank)
            lora_alpha_multiplier = best_hparams.get("lora_alpha_multiplier")
            if lora_alpha_multiplier is not None:
                lora_alpha = best_hparams.get(
                    "lora_alpha", lora_rank * lora_alpha_multiplier
                )
            else:
                lora_alpha = best_hparams.get(
                    "lora_alpha", lora_alpha or lora_rank
                )
            lora_dropout = best_hparams.get("lora_dropout", lora_dropout)
            max_seq_length = best_hparams.get("max_seq_length", max_seq_length)
            training_mode = best_hparams.get("training_mode", training_mode)

            print("Loaded hyperparameters:")
            for key in [
                "learning_rate",
                "per_device_train_batch_size",
                "gradient_accumulation_steps",
                "warmup_ratio",
                "weight_decay",
                "max_grad_norm",
                "lr_scheduler_type",
                "lora_rank",
                "lora_alpha",
                "lora_dropout",
                "max_seq_length",
            ]:
                if key in best_hparams:
                    print(f"  {key}: {best_hparams[key]}")

    train_dataset, eval_dataset = load_datasets(train_file, val_file)

    # Setup W&B logging
    setup_wandb(
        model_name=model_name,
        dataset_size=len(train_dataset),
        lora_rank=lora_rank,
        max_seq_length=max_seq_length,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_decay=weight_decay,
        training_mode=training_mode,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        disable_wandb=disable_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )

    # Load model and tokenizer
    model, tokenizer, used_attn_impl = load_model_and_tokenizer(
        model_name,
        max_seq_length,
        lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        training_mode=training_mode,
    )

    # Update W&B config with actual attention implementation used
    if not disable_wandb and wandb.run is not None:
        wandb.config.update(
            {"flash_attention": used_attn_impl}, allow_val_change=True
        )

    # Format datasets
    print("Formatting datasets...")
    train_dataset = train_dataset.map(
        lambda examples: format_prompts(examples, tokenizer, model_name), batched=True
    )
    eval_dataset = eval_dataset.map(
        lambda examples: format_prompts(examples, tokenizer, model_name), batched=True
    )

    # Setup training configuration
    training_args = setup_training_config(
        model_name=model_name,
        max_seq_length=max_seq_length,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        output_dir=output_dir,
        disable_wandb=disable_wandb,
        early_stopping_patience=early_stopping_patience,
        eval_steps=eval_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        save_total_limit=save_total_limit,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        lr_scheduler_type=lr_scheduler_type,
        training_mode=training_mode,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    # Disable optimized loss to avoid Triton CUDA version issues
    training_args.dataloader_pin_memory = False

    # Create trainer with early stopping callback
    callbacks = []
    if early_stopping_patience > 0:
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience
        )
        callbacks.append(early_stopping_callback)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # Apply GPT-OSS specific training masking
    print("Applying GPT-OSS training masking...")
    from unsloth.chat_templates import train_on_responses_only
    
    gpt_oss_kwargs = dict(
        instruction_part="<|start|>user<|message|>",
        response_part="<|start|>assistant<|channel|>final<|message|>"
    )
    
    trainer = train_on_responses_only(trainer, **gpt_oss_kwargs)
    print("GPT-OSS training masking applied successfully!")

    # Start training
    print("Starting training...")
    print(f"Output directory: {training_args.output_dir}")

    try:
        if resume_from_checkpoint:
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print("Saving current checkpoint...")
        trainer.save_model()
        return

    # Save final model
    print("Training completed! Saving model...")

    # Save LoRA adapters
    model_short = model_name.split("/")[-1].replace("-", "_")
    lora_output_dir = f"outputs/{model_short}_gpt_oss_sft_lora"
    model.save_pretrained(lora_output_dir)
    tokenizer.save_pretrained(lora_output_dir)

    print(f"LoRA adapters saved to: {lora_output_dir}")

    # Save to W&B if enabled
    if not disable_wandb:
        wandb.save(f"{lora_output_dir}/*")
        wandb.finish()

    print("Training completed successfully!")
    print(f"Final model saved to: {lora_output_dir}")


if __name__ == "__main__":
    fire.Fire(main)
