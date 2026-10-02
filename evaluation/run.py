"""Launch distributed evaluation with a normal Python command."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from evaluation.source_layout import SOURCE_LAYOUTS
from training.cli import (
    PREPARED_DATASET_FLAGS,
    add_model_argument,
    add_prepared_dataset_arguments,
    prepared_dataset_for,
)


def parse_args() -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_argument(parser)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--raw-dataset",
        type=Path,
        required=True,
        help="Raw DatasetDict whose named splits will all be scored.",
    )
    add_prepared_dataset_arguments(parser)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size per GPU.",
    )
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-proc", type=int, default=8)
    parser.add_argument(
        "--source-layout",
        choices=SOURCE_LAYOUTS,
        default="auto",
        help=(
            "auto reads training_complete.json beside the checkpoint; msa "
            "uses the model's fixed MSA training length; dynamic pads each "
            "batch to its longest source."
        ),
    )
    parser.add_argument("--skip-comet", action="store_true")
    return parser.parse_args(), parser


def main() -> None:
    args, parser = parse_args()
    prepared_dataset = prepared_dataset_for(args, parser, required=False)
    for label, value in (
        ("num-gpus", args.num_gpus),
        ("batch-size", args.batch_size),
        ("num-beams", args.num_beams),
        ("max-new-tokens", args.max_new_tokens),
        ("num-proc", args.num_proc),
    ):
        if value <= 0:
            parser.error(f"--{label} must be positive")

    command = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        f"--num_processes={args.num_gpus}",
        "-m",
        "evaluation.evaluate",
        "--model",
        args.model,
        "--model-checkpoint",
        str(args.model_checkpoint),
        "--raw-dataset",
        str(args.raw_dataset),
        "--output-dir",
        str(args.output_dir),
        "--batch-size",
        str(args.batch_size),
        "--num-beams",
        str(args.num_beams),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--num-proc",
        str(args.num_proc),
        "--source-layout",
        args.source_layout,
    ]
    if prepared_dataset is not None:
        command.extend(
            (PREPARED_DATASET_FLAGS[args.model], str(prepared_dataset))
        )
    if args.skip_comet:
        command.append("--skip-comet")

    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
