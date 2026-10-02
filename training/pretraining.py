"""Common restartable MSA pretraining engine for all four model families.

One invocation runs one optimizer/scheduler stage. The public schedule runs an
optional alignment stage followed by two independent one-epoch full stages,
matching the reported experiments.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import torch
from datasets import DatasetDict, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from training.common import (
    find_last_valid_checkpoint,
    load_state_dict,
    set_seed,
    write_json_atomic,
)


BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VOCAB_TOKENIZER = Path(
    os.environ.get(
        "VOCAB_TOKENIZER_DIR",
        REPO_ROOT / "DecoderOnly" / "VocabPlus" / "merged_tokenizer_32000",
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one MSA curriculum stage with checkpoint resumption."
    )
    parser.add_argument(
        "--model",
        choices=("pixel", "bytes", "decoder", "vocab"),
        required=True,
    )
    parser.add_argument(
        "--stage",
        choices=("alignment", "full"),
        required=True,
    )
    parser.add_argument("--percentage", type=int, default=100)
    parser.add_argument(
        "--raw-dataset",
        type=Path,
        default=None,
        help="Raw MSA DatasetDict (required for decoder and vocab models).",
    )
    parser.add_argument("--pixel-prepared-dataset", type=Path, default=None)
    parser.add_argument("--bytes-prepared-dataset", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--input-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint used to initialize this single optimizer/scheduler stage.",
    )
    parser.add_argument(
        "--vocab-tokenizer", type=Path, default=DEFAULT_VOCAB_TOKENIZER
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-steps", type=int, default=2000)
    parser.add_argument("--eval-steps", type=int, default=2000)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=None,
        help="Optional per-GPU microbatch override.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Optional per-GPU evaluation-batch override.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Optional gradient-accumulation override.",
    )
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the selected model/data objects and print the setup without training.",
    )
    return parser.parse_args()



def read_original_vocab_size(tokenizer_dir: Path) -> int:
    metadata = tokenizer_dir / "vocab_plus_meta.txt"
    with metadata.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("original_vocab_size:"):
                return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"Could not find original_vocab_size in {metadata}")



def base_decoder_and_tokenizer(tokenizer_path: str = BASE_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    decoder = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    decoder.config.use_cache = False
    return decoder, tokenizer


def prepared_dataset_total_rows(
    model_type: str,
    prepared_path: Path,
    dataset: DatasetDict,
) -> int:
    """Read the original row count recorded during data preparation."""
    metadata_name = {
        "pixel": "pixel_metadata.json",
        "bytes": "byte_compact_metadata.json",
    }[model_type]
    metadata_path = prepared_path / metadata_name
    prepared_rows = len(dataset["train"])

    if not metadata_path.is_file():
        return prepared_rows

    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    selected_rows = int(metadata.get("selected_train_rows", prepared_rows))
    if selected_rows != prepared_rows:
        raise ValueError(
            f"{metadata_path} reports {selected_rows:,} selected rows, but "
            f"the prepared dataset contains {prepared_rows:,}"
        )
    return int(metadata.get("original_train_rows", prepared_rows))


def model_prepared_dataset(args: argparse.Namespace) -> Path | None:
    if args.model == "pixel":
        return args.pixel_prepared_dataset
    if args.model == "bytes":
        return args.bytes_prepared_dataset
    return None


def build_model_and_data(
    args: argparse.Namespace,
) -> Tuple[Any, ...]:
    """Return model, tokenizer, train dataset, eval dataset, and batch config."""
    if args.model in ("pixel", "bytes"):
        prepared_dataset = model_prepared_dataset(args)
        dataset = load_from_disk(str(prepared_dataset))
        if not isinstance(dataset, DatasetDict):
            raise ValueError(f"Expected a prepared DatasetDict at {prepared_dataset}")
        if "train" not in dataset or "validation" not in dataset:
            raise ValueError("The prepared dataset needs train and validation splits")
        total_rows = prepared_dataset_total_rows(
            args.model, prepared_dataset, dataset
        )
        subset_rows = max(1, total_rows * args.percentage // 100)
        if not 0 < subset_rows <= total_rows:
            raise ValueError(
                f"Invalid {args.percentage}% subset: {subset_rows} of "
                f"{total_rows} rows"
            )
        if len(dataset["train"]) < subset_rows:
            raise ValueError(
                f"Prepared data has {len(dataset['train']):,} rows, but "
                f"{args.percentage}% needs {subset_rows:,}"
            )
        train_subset = dataset["train"].select(range(subset_rows))
    else:
        raw_dataset = load_from_disk(str(args.raw_dataset))
        if not isinstance(raw_dataset, DatasetDict):
            raise ValueError(f"Expected a DatasetDict at {args.raw_dataset}")
        if "train" not in raw_dataset or "validation" not in raw_dataset:
            raise ValueError("The raw dataset needs train and validation splits")
        total_rows = len(raw_dataset["train"])
        if total_rows == 0:
            raise ValueError("The raw train split is empty")
        subset_rows = max(1, total_rows * args.percentage // 100)
        if not 0 < subset_rows <= total_rows:
            raise ValueError(
                f"Invalid {args.percentage}% subset: {subset_rows} of "
                f"{total_rows} rows"
            )
        dataset = raw_dataset
        train_subset = raw_dataset["train"].select(range(subset_rows))

    if args.model == "pixel":
        from PixelSmolLM.data import FixedPixelCollator, PixelTranslationDataset
        from PixelSmolLM.factory import build_model
        from PixelSmolLM.utils import TrainingMode

        mode = (
            TrainingMode.ENCODER_PROJECTION
            if args.stage == "alignment"
            else TrainingMode.BOTH
        )
        model, tokenizer = build_model(mode, dynamic=False)
        if args.input_checkpoint is not None:
            load_state_dict(model, args.input_checkpoint)
        model.set_training_mode(mode)
        eval_subset = dataset["validation"].select(
            range(min(2000, len(dataset["validation"])))
        )
        train_data = PixelTranslationDataset(train_subset, include_targets=True)
        eval_data = PixelTranslationDataset(eval_subset, include_targets=True)
        collator = FixedPixelCollator()
        return (
            model, tokenizer, train_data, eval_data, collator,
            64, 64, 2, total_rows, subset_rows,
        )

    if args.model == "bytes":
        from BytesSmolLM.model import ConfigurableTextViTTranslationModel
        from BytesSmolLM.text_vit import TextViTArchitecture, TextViTConfig
        from BytesSmolLM.utils import (
            TextViTDataCollator,
            TextViTTranslationDataset,
            TrainingMode,
        )

        mode = (
            TrainingMode.ENCODER_PROJECTION
            if args.stage == "alignment"
            else TrainingMode.BOTH
        )
        decoder, tokenizer = base_decoder_and_tokenizer()
        encoder = TextViTArchitecture(config=TextViTConfig())
        model = ConfigurableTextViTTranslationModel(
            text_vit_model=encoder,
            decoder_model=decoder,
            tokenizer=tokenizer,
            training_mode=mode,
        )
        if args.input_checkpoint is not None:
            load_state_dict(model, args.input_checkpoint)
        model.set_training_mode(mode)
        model.train()
        eval_subset = dataset["validation"].select(
            range(min(2000, len(dataset["validation"])))
        )
        train_data = TextViTTranslationDataset(train_subset)
        eval_data = TextViTTranslationDataset(eval_subset)
        collator = TextViTDataCollator()
        return (
            model,
            tokenizer,
            train_data,
            eval_data,
            collator,
            32,
            32,
            4,
            total_rows,
            subset_rows,
        )

    from DecoderOnly.utils import DecoderOnlyDataCollator, DecoderOnlyTranslationDataset

    eval_subset = dataset["validation"].select(
        range(min(1000, len(dataset["validation"])))
    )

    if args.model == "decoder":
        from DecoderOnly.model import DecoderOnlyTranslationModel

        decoder, tokenizer = base_decoder_and_tokenizer()
        model = DecoderOnlyTranslationModel(decoder, tokenizer)
        if args.input_checkpoint is not None:
            load_state_dict(model, args.input_checkpoint)
        for parameter in model.parameters():
            parameter.requires_grad = True
        model.train()
        train_data = DecoderOnlyTranslationDataset(train_subset, tokenizer)
        eval_data = DecoderOnlyTranslationDataset(eval_subset, tokenizer)
        collator = DecoderOnlyDataCollator()
        return (
            model,
            tokenizer,
            train_data,
            eval_data,
            collator,
            16,
            32,
            8,
            total_rows,
            subset_rows,
        )

    from DecoderOnly.VocabPlus.model import VocabPlusTranslationModel

    tokenizer_path = args.vocab_tokenizer.resolve()
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    tokenizer.pad_token = tokenizer.eos_token
    original_vocab_size = read_original_vocab_size(tokenizer_path)
    decoder = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    decoder.config.use_cache = False
    decoder.resize_token_embeddings(len(tokenizer))
    model = VocabPlusTranslationModel(
        decoder_model=decoder,
        tokenizer=tokenizer,
        original_vocab_size=original_vocab_size,
    )
    if args.input_checkpoint is not None:
        load_state_dict(model, args.input_checkpoint)
    if args.stage == "alignment":
        model.freeze_for_alignment()
        train_batch, eval_batch, grad_accum = 32, 32, 4
    else:
        model.unfreeze_all()
        train_batch, eval_batch, grad_accum = 16, 16, 8
    model.train()
    train_data = DecoderOnlyTranslationDataset(train_subset, tokenizer)
    eval_data = DecoderOnlyTranslationDataset(eval_subset, tokenizer)
    collator = DecoderOnlyDataCollator()
    return (
        model,
        tokenizer,
        train_data,
        eval_data,
        collator,
        train_batch,
        eval_batch,
        grad_accum,
        total_rows,
        subset_rows,
    )


def save_final_weights(
    trainer: Trainer,
    tokenizer: Any,
    args: argparse.Namespace,
    subset_rows: int,
    total_rows: int,
    train_metrics: Dict[str, Any],
    train_batch_size: int,
    eval_batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int,
) -> None:
    final_dir = args.output_dir / "final"
    final_file = final_dir / "pytorch_model.bin"
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        final_dir.mkdir(parents=True, exist_ok=True)
        temporary = final_dir / f".pytorch_model.bin.tmp.{os.getpid()}"
        unwrapped = trainer.accelerator.unwrap_model(trainer.model)
        torch.save(unwrapped.state_dict(), temporary)
        os.replace(temporary, final_file)
        if args.model == "vocab":
            tokenizer.save_pretrained(str(final_dir))
        write_json_atomic(
            final_dir / "training_complete.json",
            {
                "model_type": args.model,
                "stage": args.stage,
                "percentage": args.percentage,
                "subset_rows": subset_rows,
                "total_rows": total_rows,
                "epochs": args.epochs,
                "seed": args.seed,
                "learning_rate": args.learning_rate,
                "warmup_ratio": args.warmup_ratio,
                "scheduler": "cosine",
                "train_batch_size_per_gpu": train_batch_size,
                "eval_batch_size_per_gpu": eval_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "world_size": world_size,
                "effective_global_batch_size": (
                    train_batch_size
                    * gradient_accumulation_steps
                    * world_size
                ),
                "global_step": trainer.state.global_step,
                "metrics": train_metrics,
                "input_checkpoint": (
                    str(args.input_checkpoint.resolve())
                    if args.input_checkpoint is not None
                    else None
                ),
            },
        )
        print(f"Saved completed stage to {final_file}", flush=True)
    trainer.accelerator.wait_for_everyone()


def validate_args(args: argparse.Namespace) -> None:
    if not 0 < args.percentage <= 100:
        raise ValueError("percentage must be in (0, 100]")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    for name in (
        "train_batch_size",
        "eval_batch_size",
        "gradient_accumulation_steps",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.model == "decoder" and args.stage != "full":
        raise ValueError("DecoderOnly has no alignment-only stage")
    if args.stage == "alignment" and args.input_checkpoint is not None:
        raise ValueError("Alignment stages must start fresh, without incoming weights")
    if (
        args.stage == "full"
        and args.model in ("pixel", "bytes", "vocab")
        and args.input_checkpoint is None
    ):
        raise ValueError(
            f"{args.model} full training requires --input-checkpoint"
        )
    if args.model in ("pixel", "bytes"):
        prepared_dataset = model_prepared_dataset(args)
        if prepared_dataset is None or not prepared_dataset.is_dir():
            raise FileNotFoundError(prepared_dataset)
        other_prepared = (
            args.bytes_prepared_dataset
            if args.model == "pixel"
            else args.pixel_prepared_dataset
        )
        if other_prepared is not None:
            raise ValueError(
                f"{args.model} accepts only its model-specific prepared dataset"
            )
    elif args.raw_dataset is None or not args.raw_dataset.is_dir():
        raise FileNotFoundError(args.raw_dataset)
    if args.input_checkpoint is not None and not args.input_checkpoint.is_file():
        raise FileNotFoundError(args.input_checkpoint)


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    final_file = args.output_dir / "final" / "pytorch_model.bin"
    completion_file = args.output_dir / "final" / "training_complete.json"
    has_weights = final_file.is_file() and final_file.stat().st_size > 0
    has_metadata = completion_file.is_file()
    if has_weights and has_metadata:
        print(f"Completed stage already exists; skipping: {final_file}")
        return
    if has_weights or has_metadata:
        raise RuntimeError(
            "Partial final-stage state found. Refusing to overwrite it; "
            f"inspect {final_file.parent}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        data_collator,
        train_batch_size,
        eval_batch_size,
        grad_accum,
        total_rows,
        subset_rows,
    ) = build_model_and_data(args)

    if args.train_batch_size is not None:
        train_batch_size = args.train_batch_size
    if args.eval_batch_size is not None:
        eval_batch_size = args.eval_batch_size
    if args.gradient_accumulation_steps is not None:
        grad_accum = args.gradient_accumulation_steps

    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    effective_batch = train_batch_size * grad_accum * world_size
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_parameters = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(
            f"Model={args.model} stage={args.stage} rows={subset_rows:,}/"
            f"{total_rows:,} epochs={args.epochs}",
            flush=True,
        )
        print(
            f"Per-GPU microbatch={train_batch_size}, grad_accum={grad_accum}, "
            f"world_size={world_size}, effective_global_batch={effective_batch}",
            flush=True,
        )
        print(
            f"Trainable parameters={trainable:,}/{total_parameters:,}", flush=True
        )
        if effective_batch != 512:
            print(
                "WARNING: the reference pretraining configuration uses global batch 512; "
                f"this configuration uses {effective_batch}.",
                flush=True,
            )

    if args.dry_run:
        print("Dry run complete; no training or checkpoint save was performed.")
        return

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        logging_strategy="steps",
        logging_steps=10,
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        eval_accumulation_steps=8,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_safetensors=False,
        bf16=True,
        fp16=False,
        seed=args.seed,
        data_seed=args.seed,
        dataloader_drop_last=False,
        dataloader_num_workers=8 if args.model == "decoder" else 4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to=["wandb"] if rank == 0 else [],
        run_name=args.run_name,
        optim="adamw_hf",
        max_grad_norm=1.0,
        push_to_hub=False,
        ddp_find_unused_parameters=False,
        label_names=["target_ids"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    resume_checkpoint = find_last_valid_checkpoint(args.output_dir)
    if rank == 0:
        if resume_checkpoint is None:
            print("Starting this stage from its declared incoming weights.", flush=True)
        else:
            print(f"Resuming Trainer state from {resume_checkpoint}", flush=True)

    result = trainer.train(
        resume_from_checkpoint=(
            str(resume_checkpoint) if resume_checkpoint is not None else None
        )
    )
    save_final_weights(
        trainer,
        tokenizer,
        args,
        subset_rows,
        total_rows,
        dict(result.metrics),
        train_batch_size,
        eval_batch_size,
        grad_accum,
        world_size,
    )


if __name__ == "__main__":
    main()
