"""Run the reported MSA protocol for one or more model families.

This is an orchestration layer only. It delegates alignment and full training
to the model-specific entry points, so 100% MSA and curriculum runs use
exactly the same training implementation.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


MODELS = ("pixel", "bytes", "vocab", "decoder")
ALIGNMENT_MODELS = ("pixel", "bytes", "vocab")
MODULES = {
    "decoder": "DecoderOnly",
    "vocab": "DecoderOnly.VocabPlus",
    "bytes": "BytesSmolLM",
    "pixel": "PixelSmolLM",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=list(MODELS),
    )
    parser.add_argument("--raw-dataset", type=Path)
    parser.add_argument("--pixel-prepared-dataset", type=Path)
    parser.add_argument("--bytes-prepared-dataset", type=Path)
    parser.add_argument("--vocab-tokenizer", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root directory for model alignment/full checkpoints.",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--percentage", type=int, default=100)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument(
        "--alignment-batch-sizes",
        type=int,
        nargs=3,
        default=(64, 32, 32),
        metavar=("PIXEL", "BYTES", "VOCAB"),
        help="Per-GPU alignment microbatches in Pixel, Byte, Vocab+ order.",
    )
    parser.add_argument(
        "--full-batch-sizes",
        type=int,
        nargs=4,
        default=(64, 32, 16, 16),
        metavar=("PIXEL", "BYTES", "VOCAB", "BASE"),
        help="Per-GPU full-training microbatches in Pixel, Byte, Vocab+, Base order.",
    )
    parser.add_argument(
        "--effective-batch-size",
        type=int,
        default=512,
        help="Global batch retained by deriving gradient accumulation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--save-steps", type=int, default=2000)
    parser.add_argument("--eval-steps", type=int, default=2000)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _data_args(args: argparse.Namespace, model: str) -> list[str]:
    if model == "pixel":
        return [
            "--pixel-prepared-dataset",
            str(args.pixel_prepared_dataset),
        ]
    if model == "bytes":
        return [
            "--bytes-prepared-dataset",
            str(args.bytes_prepared_dataset),
        ]
    result = ["--raw-dataset", str(args.raw_dataset)]
    if model == "vocab":
        result.extend(("--vocab-tokenizer", str(args.vocab_tokenizer)))
    return result


def _common_args(
    args: argparse.Namespace,
    model: str,
    stage: str,
) -> list[str]:
    if stage == "alignment":
        batch_sizes = dict(zip(ALIGNMENT_MODELS, args.alignment_batch_sizes))
    else:
        batch_sizes = dict(zip(MODELS, args.full_batch_sizes))
    result = [
        *_data_args(args, model),
        "--percentage",
        str(args.percentage),
        "--num-gpus",
        str(args.num_gpus),
        "--train-batch-size",
        str(batch_sizes[model]),
        "--effective-batch-size",
        str(args.effective_batch_size),
        "--seed",
        str(args.seed),
        "--learning-rate",
        str(args.learning_rate),
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--save-steps",
        str(args.save_steps),
        "--eval-steps",
        str(args.eval_steps),
        "--save-total-limit",
        str(args.save_total_limit),
    ]
    if args.dry_run:
        result.append("--dry-run")
    return result


def _run(command: list[str]) -> None:
    print("Delegating:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _validate(args: argparse.Namespace) -> None:
    if not 0 < args.percentage <= 100:
        raise ValueError("percentage must be in (0, 100]")
    for name in ("num_gpus", "effective_batch_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if any(
        size <= 0
        for size in (*args.alignment_batch_sizes, *args.full_batch_sizes)
    ):
        raise ValueError("batch sizes must be positive")

    alignment_batches = dict(zip(ALIGNMENT_MODELS, args.alignment_batch_sizes))
    full_batches = dict(zip(MODELS, args.full_batch_sizes))
    for model in args.models:
        stages = [("full", full_batches[model])]
        if model != "decoder":
            stages.insert(0, ("alignment", alignment_batches[model]))
        for stage, batch_size in stages:
            denominator = batch_size * args.num_gpus
            if args.effective_batch_size % denominator:
                raise ValueError(
                    f"{model} {stage}: effective-batch-size {args.effective_batch_size} "
                    f"is not divisible by batch-size {batch_size} x "
                    f"num-gpus {args.num_gpus}"
                )

    selected = set(args.models)
    requirements = []
    if selected & {"decoder", "vocab"}:
        requirements.append(("--raw-dataset", args.raw_dataset, "directory"))
    if "pixel" in selected:
        requirements.append(
            ("--pixel-prepared-dataset", args.pixel_prepared_dataset, "directory")
        )
    if "bytes" in selected:
        requirements.append(
            ("--bytes-prepared-dataset", args.bytes_prepared_dataset, "directory")
        )
    if "vocab" in selected:
        requirements.append(
            ("--vocab-tokenizer", args.vocab_tokenizer, "directory")
        )

    for flag, path, kind in requirements:
        if path is None:
            raise ValueError(f"{flag} is required for the selected models")
        if not args.dry_run and kind == "directory" and not path.is_dir():
            raise FileNotFoundError(path)


def main() -> None:
    args = parse_args()
    _validate(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        model_root = args.output_dir / model
        module_root = MODULES[model]

        alignment_checkpoint = None
        if model != "decoder":
            alignment_dir = model_root / "alignment"
            alignment_args = _common_args(args, model, "alignment")
            _run(
                [
                    sys.executable,
                    "-m",
                    f"{module_root}.train_alignment",
                    *alignment_args,
                    "--output-dir",
                    str(alignment_dir),
                    "--run-name",
                    f"{args.run_name}-{model}",
                ]
            )
            alignment_checkpoint = (
                alignment_dir / "final" / "pytorch_model.bin"
            )

        full_args = _common_args(args, model, "full")
        full_command = [
            sys.executable,
            "-m",
            f"{module_root}.train_full",
            *full_args,
            "--output-dir",
            str(model_root),
            "--run-name",
            f"{args.run_name}-{model}",
        ]
        if alignment_checkpoint is not None:
            full_command.extend(
                ("--alignment-checkpoint", str(alignment_checkpoint))
            )
        _run(full_command)


if __name__ == "__main__":
    main()
