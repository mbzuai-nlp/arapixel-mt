"""Restartable dialect fine-tuning for any prepared training mixture."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import torch
from datasets import DatasetDict, load_from_disk
from transformers import Trainer, TrainingArguments

from training.common import find_last_valid_checkpoint, write_json_atomic
from training.cli import (
    add_model_argument,
    add_prepared_dataset_arguments,
    prepared_dataset_for,
)

from finetuning.common import (
    build_training_data,
    load_bytes_model,
    load_decoder_model,
    load_dynamic_pixel_model,
    load_vocab_model,
)


def parse_args() -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(
        description="Restartable fine-tuning on one prepared dialect/MSA mixture."
    )
    add_model_argument(parser)
    parser.add_argument(
        "--model-checkpoint",
        type=Path,
        required=True,
        help="Final MSA checkpoint used to initialize fine-tuning.",
    )
    add_prepared_dataset_arguments(parser)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=None,
        help="Optional per-GPU microbatch override.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Optional gradient-accumulation override.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load the checkpoint/data and print the setup without training.",
    )
    return parser.parse_args(), parser


def get_train_split(dataset):
    if isinstance(dataset, DatasetDict):
        if "train" not in dataset:
            raise ValueError("Prepared DatasetDict has no train split")
        return dataset["train"]
    return dataset


def main() -> None:
    args, parser = parse_args()
    prepared_dataset = prepared_dataset_for(args, parser)
    if not args.model_checkpoint.is_file():
        raise FileNotFoundError(args.model_checkpoint)
    if not prepared_dataset.is_dir():
        raise FileNotFoundError(prepared_dataset)
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    for name in ("train_batch_size", "gradient_accumulation_steps"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(
                f"{name.replace('_', '-')} must be positive"
            )

    final_file = args.output_dir / "best_model.bin"
    completion_file = args.output_dir / "training_complete.json"
    has_weights = final_file.is_file() and final_file.stat().st_size > 0
    has_metadata = completion_file.is_file()
    if has_weights and has_metadata:
        print(f"Completed fine-tuning already exists; skipping: {final_file}")
        return
    if has_weights or has_metadata:
        raise RuntimeError(
            "Partial final fine-tuning state found. Refusing to overwrite it; "
            f"inspect {args.output_dir}"
        )

    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = get_train_split(load_from_disk(str(prepared_dataset)))
    loaders = {
        "pixel": load_dynamic_pixel_model,
        "bytes": load_bytes_model,
        "decoder": load_decoder_model,
        "vocab": load_vocab_model,
    }
    model, tokenizer = loaders[args.model](str(args.model_checkpoint))
    train_dataset, data_collator = build_training_data(
        args.model, dataset, tokenizer
    )

    if args.model == "pixel":
        batch_size, grad_accum = 16, 2
    else:
        batch_size, grad_accum = 8, 4
    if args.train_batch_size is not None:
        batch_size = args.train_batch_size
    if args.gradient_accumulation_steps is not None:
        grad_accum = args.gradient_accumulation_steps
    effective_batch = batch_size * grad_accum * world_size
    if rank == 0:
        print(
            f"Fine-tuning {args.model} on {len(dataset):,} rows for "
            f"{args.epochs} epoch(s)",
            flush=True,
        )
        print(
            f"Per-GPU microbatch={batch_size}, grad_accum={grad_accum}, "
            f"world_size={world_size}, effective_global_batch={effective_batch}",
            flush=True,
        )
        if effective_batch != 128:
            print(
                "WARNING: the reference fine-tuning configuration uses global batch 128; "
                f"this configuration uses {effective_batch}.",
                flush=True,
            )

    if args.dry_run:
        print("Dry run complete; no fine-tuning or checkpoint save was performed.")
        return

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        seed=args.seed,
        data_seed=args.seed,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_safetensors=False,
        logging_strategy="steps",
        logging_steps=10,
        report_to=["wandb"] if rank == 0 else [],
        run_name=args.run_name,
        dataloader_drop_last=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        bf16=True,
        fp16=False,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        label_names=["target_ids"],
        optim="adamw_hf",
        max_grad_norm=1.0,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    resume_checkpoint = find_last_valid_checkpoint(args.output_dir)
    if rank == 0:
        if resume_checkpoint is None:
            print("Starting fine-tuning from the MSA checkpoint.", flush=True)
        else:
            print(f"Resuming fine-tuning from {resume_checkpoint}", flush=True)

    result = trainer.train(
        resume_from_checkpoint=(
            str(resume_checkpoint) if resume_checkpoint is not None else None
        )
    )
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        temporary = args.output_dir / f".best_model.bin.tmp.{os.getpid()}"
        unwrapped = trainer.accelerator.unwrap_model(trainer.model)
        torch.save(unwrapped.state_dict(), temporary)
        os.replace(temporary, final_file)
        write_json_atomic(
            args.output_dir / "training_complete.json",
            {
                "model_type": args.model,
                "training_phase": "finetuning",
                "input_checkpoint": str(args.model_checkpoint.resolve()),
                "prepared_dataset": str(prepared_dataset.resolve()),
                "dataset_rows": len(dataset),
                "epochs": args.epochs,
                "seed": args.seed,
                "learning_rate": args.learning_rate,
                "warmup_ratio": args.warmup_ratio,
                "scheduler": "cosine",
                "global_step": trainer.state.global_step,
                "train_batch_size_per_gpu": batch_size,
                "gradient_accumulation_steps": grad_accum,
                "world_size": world_size,
                "effective_global_batch_size": effective_batch,
                "metrics": dict(result.metrics),
            },
        )
        print(f"Saved completed fine-tuned weights to {final_file}", flush=True)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
