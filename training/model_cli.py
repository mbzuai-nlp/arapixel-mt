"""Shared CLI for model-specific MSA training entry points.

The low-level :mod:`training.pretraining` module runs one optimizer/scheduler
stage. These entry points add distributed launch, stable model-specific
arguments, and the reported two-stage full-training protocol.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


MODEL_MODULES = {
    "pixel": "PixelSmolLM",
    "bytes": "BytesSmolLM",
    "decoder": "DecoderOnly",
    "vocab": "DecoderOnly.VocabPlus",
}

DEFAULT_TRAIN_BATCH_SIZES = {
    ("pixel", "alignment"): 64,
    ("pixel", "full"): 64,
    ("bytes", "alignment"): 32,
    ("bytes", "full"): 32,
    ("vocab", "alignment"): 32,
    ("vocab", "full"): 16,
    ("decoder", "full"): 16,
}


def _add_data_argument(parser: argparse.ArgumentParser, model: str) -> None:
    if model == "pixel":
        parser.add_argument(
            "--pixel-prepared-dataset",
            dest="prepared_dataset",
            type=Path,
            required=True,
        )
    elif model == "bytes":
        parser.add_argument(
            "--bytes-prepared-dataset",
            dest="prepared_dataset",
            type=Path,
            required=True,
        )
    else:
        parser.add_argument("--raw-dataset", type=Path, required=True)

    if model == "vocab":
        parser.add_argument(
            "--vocab-tokenizer",
            type=Path,
            required=True,
            help="Merged Vocab+ tokenizer directory.",
        )


def _parser(model: str, phase: str) -> argparse.ArgumentParser:
    if model not in MODEL_MODULES:
        raise ValueError(f"Unsupported model: {model}")
    if phase not in ("alignment", "full"):
        raise ValueError(f"Unsupported phase: {phase}")
    if model == "decoder" and phase == "alignment":
        raise ValueError("Base has no alignment stage")

    description = (
        f"Run the {model} MSA alignment stage."
        if phase == "alignment"
        else f"Run both reported {model} full-training epochs."
    )
    parser = argparse.ArgumentParser(description=description)
    _add_data_argument(parser, model)
    if phase == "full" and model != "decoder":
        parser.add_argument(
            "--alignment-checkpoint",
            type=Path,
            required=True,
            help="Final checkpoint produced by this model's alignment stage.",
        )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Alignment-stage directory."
            if phase == "alignment"
            else "Model experiment directory; full_epoch1/ and full_epoch2/ are created here."
        ),
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--percentage", type=int, default=100)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--save-steps", type=int, default=2000)
    parser.add_argument("--eval-steps", type=int, default=2000)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=None,
        help="Per-GPU training microbatch; defaults to the reported setting.",
    )
    parser.add_argument(
        "--effective-batch-size",
        type=int,
        default=512,
        help=(
            "Target global batch. Gradient accumulation is derived from this, "
            "the per-GPU microbatch, and --num-gpus."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the distributed stage command(s) without running them.",
    )
    return parser


def _data_args(args: argparse.Namespace, model: str) -> list[str]:
    if model == "pixel":
        result = ["--pixel-prepared-dataset", str(args.prepared_dataset)]
    elif model == "bytes":
        result = ["--bytes-prepared-dataset", str(args.prepared_dataset)]
    else:
        result = ["--raw-dataset", str(args.raw_dataset)]
    if model == "vocab":
        result.extend(("--vocab-tokenizer", str(args.vocab_tokenizer)))
    return result


def _validate(args: argparse.Namespace, model: str, phase: str) -> None:
    if not 0 < args.percentage <= 100:
        raise ValueError("percentage must be in (0, 100]")
    for name in (
        "num_gpus",
        "effective_batch_size",
        "save_steps",
        "eval_steps",
        "save_total_limit",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.train_batch_size is not None and args.train_batch_size <= 0:
        raise ValueError("train-batch-size must be positive")
    _batch_configuration(args, model, phase)
    data_path = (
        args.prepared_dataset
        if model in ("pixel", "bytes")
        else args.raw_dataset
    )
    if not args.dry_run and not data_path.is_dir():
        raise FileNotFoundError(data_path)
    if model == "vocab" and not args.dry_run and not args.vocab_tokenizer.is_dir():
        raise FileNotFoundError(args.vocab_tokenizer)
    if phase == "full" and model != "decoder":
        if not args.dry_run and not args.alignment_checkpoint.is_file():
            raise FileNotFoundError(args.alignment_checkpoint)


def _batch_configuration(
    args: argparse.Namespace,
    model: str,
    stage: str,
) -> tuple[int, int]:
    batch_size = (
        args.train_batch_size
        if args.train_batch_size is not None
        else DEFAULT_TRAIN_BATCH_SIZES[(model, stage)]
    )
    denominator = batch_size * args.num_gpus
    if args.effective_batch_size % denominator:
        raise ValueError(
            f"effective-batch-size {args.effective_batch_size} is not divisible "
            f"by train-batch-size {batch_size} x num-gpus {args.num_gpus}"
        )
    return batch_size, args.effective_batch_size // denominator


def _stage_command(
    args: argparse.Namespace,
    model: str,
    stage: str,
    output_dir: Path,
    run_suffix: str,
    input_checkpoint: Path | None,
) -> list[str]:
    train_batch_size, gradient_accumulation_steps = _batch_configuration(
        args, model, stage
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={args.num_gpus}",
        "-m",
        "training.pretraining",
        "--model",
        model,
        "--stage",
        stage,
        "--percentage",
        str(args.percentage),
        "--output-dir",
        str(output_dir),
        "--epochs",
        "1",
        "--learning-rate",
        str(args.learning_rate),
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--seed",
        str(args.seed),
        "--save-steps",
        str(args.save_steps),
        "--eval-steps",
        str(args.eval_steps),
        "--save-total-limit",
        str(args.save_total_limit),
        "--train-batch-size",
        str(train_batch_size),
        "--gradient-accumulation-steps",
        str(gradient_accumulation_steps),
        "--run-name",
        f"{args.run_name}-{run_suffix}",
        *_data_args(args, model),
    ]
    if input_checkpoint is not None:
        command.extend(("--input-checkpoint", str(input_checkpoint)))
    return command


def _run(command: list[str], dry_run: bool) -> None:
    print("Running:", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def alignment_main(model: str) -> None:
    args = _parser(model, "alignment").parse_args()
    _validate(args, model, "alignment")
    command = _stage_command(
        args,
        model,
        "alignment",
        args.output_dir,
        "alignment",
        None,
    )
    _run(command, args.dry_run)


def full_main(model: str) -> None:
    args = _parser(model, "full").parse_args()
    _validate(args, model, "full")

    first = args.output_dir / "full_epoch1"
    second = args.output_dir / "full_epoch2"
    incoming = None if model == "decoder" else args.alignment_checkpoint

    first_command = _stage_command(
        args,
        model,
        "full",
        first,
        "full-epoch1",
        incoming,
    )
    _run(first_command, args.dry_run)

    first_checkpoint = first / "final" / "pytorch_model.bin"
    if not args.dry_run and not first_checkpoint.is_file():
        raise RuntimeError(
            f"Full epoch 1 did not produce its final checkpoint: {first_checkpoint}"
        )
    second_command = _stage_command(
        args,
        model,
        "full",
        second,
        "full-epoch2",
        first_checkpoint,
    )
    _run(second_command, args.dry_run)
