#!/usr/bin/env python3
"""
train_llm_advanced.py

End-to-end training pipeline for a causal language model with:
- AdamW optimizer
- Cosine learning rate schedule with warmup
- Gradient accumulation, mixed precision (fp16/bf16) via accelerate
- Periodic checkpointing and best-checkpoint tracking on eval loss
- Resumable training
- Logging to W&B and TensorBoard

Usage (basic):
  python train_llm_advanced.py \
    --model_name_or_path gpt2 \
    --train_file data/train.jsonl \
    --eval_file data/eval.jsonl \
    --output_dir outputs/llm-exp1

This script assumes JSONL with fields: {"text": "..."}
"""
from __future__ import annotations
import os
import math
import json
import time
import random
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup,
)
from transformers.optimization import AdamW
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from datasets import load_dataset
from pathlib import Path

try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False

from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainArgs:
    model_name_or_path: str = "gpt2"
    train_file: str = ""
    eval_file: str = ""
    output_dir: str = "outputs/run"
    max_seq_len: int = 1024
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.1
    warmup_ratio: float = 0.03
    num_train_epochs: int = 3
    max_train_steps: Optional[int] = None
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 3
    seed: int = 42
    mixed_precision: Optional[str] = None  # 'fp16' or 'bf16' or None
    gradient_checkpointing: bool = True
    resume_from_checkpoint: Optional[str] = None
    use_wandb: bool = True
    wandb_project: str = "custom-llm"
    wandb_run_name: Optional[str] = None


def parse_args() -> TrainArgs:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--model_name_or_path', type=str, default=TrainArgs.model_name_or_path)
    p.add_argument('--train_file', type=str, required=True)
    p.add_argument('--eval_file', type=str, required=True)
    p.add_argument('--output_dir', type=str, default=TrainArgs.output_dir)
    p.add_argument('--max_seq_len', type=int, default=TrainArgs.max_seq_len)
    p.add_argument('--per_device_train_batch_size', type=int, default=TrainArgs.per_device_train_batch_size)
    p.add_argument('--per_device_eval_batch_size', type=int, default=TrainArgs.per_device_eval_batch_size)
    p.add_argument('--gradient_accumulation_steps', type=int, default=TrainArgs.gradient_accumulation_steps)
    p.add_argument('--learning_rate', type=float, default=TrainArgs.learning_rate)
    p.add_argument('--weight_decay', type=float, default=TrainArgs.weight_decay)
    p.add_argument('--warmup_ratio', type=float, default=TrainArgs.warmup_ratio)
    p.add_argument('--num_train_epochs', type=int, default=TrainArgs.num_train_epochs)
    p.add_argument('--max_train_steps', type=int, default=0)
    p.add_argument('--logging_steps', type=int, default=TrainArgs.logging_steps)
    p.add_argument('--eval_steps', type=int, default=TrainArgs.eval_steps)
    p.add_argument('--save_steps', type=int, default=TrainArgs.save_steps)
    p.add_argument('--save_total_limit', type=int, default=TrainArgs.save_total_limit)
    p.add_argument('--seed', type=int, default=TrainArgs.seed)
    p.add_argument('--mixed_precision', type=str, default="auto")
    p.add_argument('--gradient_checkpointing', action='store_true')
    p.add_argument('--no_wandb', action='store_true')
    p.add_argument('--resume_from_checkpoint', type=str, default=None)
    p.add_argument('--wandb_project', type=str, default=TrainArgs.wandb_project)
    p.add_argument('--wandb_run_name', type=str, default=None)
    args = p.parse_args()

    ta = TrainArgs(
        model_name_or_path=args.model_name_or_path,
        train_file=args.train_file,
        eval_file=args.eval_file,
        output_dir=args.output_dir,
        max_seq_len=args.max_seq_len,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        max_train_steps=(args.max_train_steps or None),
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        mixed_precision=(None if args.mixed_precision == 'none' else args.mixed_precision),
        gradient_checkpointing=args.gradient_checkpointing,
        resume_from_checkpoint=args.resume_from_checkpoint,
        use_wandb=(not args.no_wandb),
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
    return ta


def get_datasets(train_path: str, eval_path: str, tokenizer: AutoTokenizer, max_len: int):
    # Expect JSONL with {"text": str}
    def load_jsonl(path):
        records = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if 'text' in obj:
                    records.append(obj)
        return records

    class JsonlDataset(Dataset):
        def __init__(self, items):
            self.items = items
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx):
            return self.items[idx]['text']

    train_items = load_jsonl(train_path)
    eval_items = load_jsonl(eval_path)

    def tok_map(example_texts):
        return tokenizer(
            example_texts,
            truncation=True,
            max_length=max_len,
            padding=False,
            return_tensors='pt',
        )

    # We will rely on DataCollatorForLanguageModeling to create labels = shifted inputs
    return JsonlDataset(train_items), JsonlDataset(eval_items)


def create_optimizer_and_scheduler(model, learning_rate, weight_decay, num_warmup_steps, num_training_steps):
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return optimizer, lr_scheduler


def save_checkpoint(state, output_dir: str, tag: str, save_total_limit: int = 3):
    ckpt_dir = Path(output_dir) / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving checkpoint: {ckpt_dir}")
    state['model'].save_pretrained(ckpt_dir)
    state['tokenizer'].save_pretrained(ckpt_dir)
    torch.save({'optimizer': state['optimizer'].state_dict(),
                'lr_scheduler': state['lr_scheduler'].state_dict(),
                'step': state['step']}, ckpt_dir / 'trainer_state.pt')

    # cleanup old checkpoints
    ckpts = sorted([p for p in Path(output_dir).glob('checkpoint-*')], key=lambda p: p.stat().st_mtime, reverse=True)
    for old in ckpts[save_total_limit:]:
        try:
            import shutil
            shutil.rmtree(old)
            logger.info(f"Removed old checkpoint: {old}")
        except Exception as e:
            logger.warning(f"Failed to remove {old}: {e}")


def main():
    args = parse_args()
    set_seed(args.seed)

    project_config = ProjectConfiguration(project_dir=args.output_dir, automatic_checkpoint_naming=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=(None if args.mixed_precision == 'auto' else args.mixed_precision),
        project_config=project_config,
        log_with=(['wandb'] if args.use_wandb and WANDB_AVAILABLE else None),
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tb'))
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    else:
        writer = None

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    train_ds, eval_ds = get_datasets(args.train_file, args.eval_file, tokenizer, args.max_seq_len)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=lambda batch: data_collator(batch),
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_ds,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=lambda batch: data_collator(batch),
    )

    # Steps computation
    total_train_steps = args.max_train_steps
    if total_train_steps is None or total_train_steps <= 0:
        steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
        total_train_steps = steps_per_epoch * args.num_train_epochs
    warmup_steps = int(total_train_steps * args.warmup_ratio)

    optimizer, lr_scheduler = create_optimizer_and_scheduler(
        model, args.learning_rate, args.weight_decay, warmup_steps, total_train_steps
    )

    # Prepare with accelerator
    model, optimizer, train_loader, eval_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, lr_scheduler
    )

    start_step = 0
    if args.resume_from_checkpoint:
        ckpt = torch.load(os.path.join(args.resume_from_checkpoint, 'trainer_state.pt'), map_location='cpu')
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        start_step = ckpt.get('step', 0)
        logger.info(f"Resumed from {args.resume_from_checkpoint} at step {start_step}")

    best_eval_loss = float('inf')

    model.train()
    global_step = start_step
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if accelerator.is_main_process and global_step % args.logging_steps == 0:
                lr = lr_scheduler.get_last_lr()[0]
                logger.info(f"step {global_step} | loss {loss.item():.4f} | lr {lr:.2e}")
                if writer:
                    writer.add_scalar('train/loss', loss.item(), global_step)
                    writer.add_scalar('train/lr', lr, global_step)
                if args.use_wandb and WANDB_AVAILABLE:
                    wandb.log({'train/loss': loss.item(), 'train/lr': lr, 'step': global_step})

            if accelerator.is_main_process and global_step % args.eval_steps == 0:
                eval_loss = evaluate(model, eval_loader, accelerator)
                if writer:
                    writer.add_scalar('eval/loss', eval_loss, global_step)
                if args.use_wandb and WANDB_AVAILABLE:
                    wandb.log({'eval/loss': eval_loss, 'step': global_step})
                logger.info(f"Eval @ step {global_step}: loss {eval_loss:.4f}")
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    unwrapped = accelerator.unwrap_model(model)
                    save_checkpoint({'model': unwrapped, 'tokenizer': tokenizer, 'optimizer': optimizer,
                                     'lr_scheduler': lr_scheduler, 'step': global_step},
                                    args.output_dir, f"checkpoint-best", args.save_total_limit)

            if accelerator.is_main_process and global_step % args.save_steps == 0:
                unwrapped = accelerator.unwrap_model(model)
                save_checkpoint({'model': unwrapped, 'tokenizer': tokenizer, 'optimizer': optimizer,
                                 'lr_scheduler': lr_scheduler, 'step': global_step},
                                args.output_dir, f"checkpoint-{global_step}", args.save_total_limit)

            if global_step >= total_train_steps:
                break
        if global_step >= total_train_steps:
            break

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        save_checkpoint({'model': unwrapped, 'tokenizer': tokenizer, 'optimizer': optimizer,
                         'lr_scheduler': lr_scheduler, 'step': global_step},
                        args.output_dir, f"checkpoint-final", args.save_total_limit)
        if writer:
            writer.close()


def evaluate(model, eval_loader, accelerator: Accelerator):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in eval_loader:
            outputs = model(**batch)
            loss = accelerator.gather_for_metrics(outputs.loss.detach())
            losses.append(loss)
    model.train()
    # losses is a list of tensors; concatenate and average
    losses = torch.cat([x.reshape(-1) for x in losses])
    return losses.mean().item()


if __name__ == '__main__':
    main()
