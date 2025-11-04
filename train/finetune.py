"""https://colab.research.google.com/drive/1Kose-ucXO1IBaZq5BvbwWieuubP7hxvQ?usp=sharing
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional

from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import EarlyStoppingCallback
from dotenv import load_dotenv

load_dotenv()


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


def format_prompts(examples: Dict[str, List[str]], tokenizer):
    instructions = examples["instruction"]
    responses = examples["response"]
    texts = []

    for instruction, response in zip(instructions, responses):
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)

    return {"text": texts}


def main(
    model_name: str = "unsloth/Qwen2.5-72B-Instruct-bnb-4bit",
    resume_from_checkpoint: str = None,
    lora_rank: int = 64,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    max_seq_length: int = 512,
    num_epochs: float = 2,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 0.00014544944973479554,
    warmup_ratio: float = 0.05,
    weight_decay: float = 0.01,
    max_grad_norm: float = 0.3,
    lr_scheduler_type: str = "constant_with_warmup",
    hparams_path: str = "train_results/best_hparams.json",
    train_file: str = "data/train.json",
    val_file: str = "data/val.json",
    output_dir: str = None,
    use_wandb: bool = True,
    early_stopping_patience: int = 3,
    eval_steps: int = 100,
    save_steps: int = 100,
    logging_steps: int = 10,
    save_total_limit: int = 2,
):
    if hparams_path.is_file():
        with open(hparams_path, "r") as f:
            hparams = json.load(f)["params"]
        
        learning_rate = hparams["learning_rate"]
        batch_size = hparams["per_device_train_batch_size"]
        gradient_accumulation_steps = hparams["gradient_accumulation_steps"]
        lora_dropout = hparams["lora_dropout"]
        max_seq_length = hparams["max_seq_length"]

    train_dataset = load_dataset("json", data_files=train_file, split="train")
    eval_dataset = load_dataset("json", data_files=val_file, split="train")

    if use_wandb:
        os.environ["WANDB_PROJECT"] = "cyber-llm-instruct"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        max_lora_rank=lora_rank,
        attn_implementation="flash_attention_2",
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=DEFAULT_TARGET_MODULES,
        lora_alpha=lora_alpha or lora_rank,
        lora_dropout=lora_dropout,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="qwen25",
    )
    train_dataset = train_dataset.map(
        lambda examples: format_prompts(examples, tokenizer), batched=True
    )
    eval_dataset = eval_dataset.map(
        lambda examples: format_prompts(examples, tokenizer), batched=True
    )

    model_short = model_name.split("/")[-1].replace("-", "_")
    if not output_dir:
        output_dir = f"outputs/{model_short}_sft"

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
        optim=DEFAULT_OPTIMIZER,
        lr_scheduler_type=lr_scheduler_type,
        seed=3407,
        report_to="wandb" if use_wandb else "none",
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        packing=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,  # Disable to avoid Triton CUDA version issues
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    except KeyboardInterrupt:
        trainer.save_model()
        return

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
