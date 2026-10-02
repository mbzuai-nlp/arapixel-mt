"""Fine-tune one prepared model dataset and evaluate its checkpoint."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from training.cli import (
    PREPARED_DATASET_FLAGS,
    add_model_argument,
    add_prepared_dataset_arguments,
    prepared_dataset_for,
)

DEFAULT_FINETUNE_BATCH_SIZES = {
    "pixel": 16,
    "bytes": 8,
    "decoder": 8,
    "vocab": 8,
}


def parse_args() -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_argument(parser)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    add_prepared_dataset_arguments(parser)
    parser.add_argument(
        "--raw-dataset",
        type=Path,
        required=True,
        help="Raw evaluation DatasetDict; all named splits are scored.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Root for the fine-tuned checkpoint, prepared eval cache, and results.",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=None,
        help="Per-GPU microbatch; defaults to 16 for Pixel and 8 otherwise.",
    )
    parser.add_argument(
        "--effective-batch-size",
        type=int,
        default=128,
        help="Global batch retained by deriving gradient accumulation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument(
        "--evaluate-pretrained",
        action="store_true",
        help="Evaluate the incoming MSA checkpoint before fine-tuning.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--skip-comet", action="store_true")
    parser.add_argument(
        "--post-label",
        default="post",
        help="Post-fine-tuning result directory name.",
    )
    return parser.parse_args(), parser


def _batch_configuration(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> tuple[int, int]:
    batch_size = (
        args.train_batch_size
        if args.train_batch_size is not None
        else DEFAULT_FINETUNE_BATCH_SIZES[args.model]
    )
    for label, value in (
        ("num-gpus", args.num_gpus),
        ("train-batch-size", batch_size),
        ("effective-batch-size", args.effective_batch_size),
    ):
        if value <= 0:
            parser.error(f"--{label} must be positive")
    denominator = batch_size * args.num_gpus
    if args.effective_batch_size % denominator:
        parser.error(
            f"--effective-batch-size {args.effective_batch_size} is not divisible "
            f"by --train-batch-size {batch_size} x --num-gpus {args.num_gpus}"
        )
    return batch_size, args.effective_batch_size // denominator


def run(command: list[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def validated(path: Path) -> bool:
    validation = path / "validation.json"
    if not validation.is_file():
        return False
    try:
        with validation.open(encoding="utf-8") as handle:
            return bool(json.load(handle).get("valid"))
    except (OSError, ValueError):
        return False


def train(
    args: argparse.Namespace,
    prepared_train: Path,
    output_dir: Path,
) -> Path:
    final = output_dir / "best_model.bin"
    completion = output_dir / "training_complete.json"
    if final.is_file() and final.stat().st_size > 0 and completion.is_file():
        print(f"[train] complete; skipping {final}")
        return final

    run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={args.num_gpus}",
            "-m",
            "finetuning.train",
            "--model",
            args.model,
            "--model-checkpoint",
            str(args.model_checkpoint),
            PREPARED_DATASET_FLAGS[args.model],
            str(prepared_train),
            "--output-dir",
            str(output_dir),
            "--epochs",
            str(args.epochs),
            "--learning-rate",
            str(args.learning_rate),
            "--seed",
            str(args.seed),
            "--train-batch-size",
            str(args.train_batch_size),
            "--gradient-accumulation-steps",
            str(args.gradient_accumulation_steps),
            "--run-name",
            args.run_name,
        ]
    )
    if not final.is_file() or not completion.is_file():
        raise RuntimeError(
            "Fine-tuning exited without a complete checkpoint/metadata pair: "
            f"{output_dir}"
        )
    return final


def evaluate(
    args: argparse.Namespace,
    checkpoint: Path,
    prepared_eval: Path,
    result_dir: Path,
    source_layout: str,
) -> None:
    if validated(result_dir):
        print(f"[evaluate] complete; skipping {result_dir}")
        return
    command = [
        sys.executable,
        "-m",
        "evaluation.run",
        "--num-gpus",
        str(args.num_gpus),
        "--model",
        args.model,
        "--model-checkpoint",
        str(checkpoint),
        "--raw-dataset",
        str(args.raw_dataset),
        PREPARED_DATASET_FLAGS[args.model],
        str(prepared_eval),
        "--output-dir",
        str(result_dir),
        "--source-layout",
        source_layout,
        "--batch-size",
        str(args.eval_batch_size),
    ]
    if args.skip_comet:
        command.append("--skip-comet")
    run(command)


def main() -> None:
    args, parser = parse_args()
    prepared_train = prepared_dataset_for(args, parser)
    if args.epochs <= 0 or args.eval_batch_size <= 0:
        parser.error("--epochs and --eval-batch-size must be positive")
    args.train_batch_size, args.gradient_accumulation_steps = (
        _batch_configuration(args, parser)
    )
    for path in (
        args.model_checkpoint,
        prepared_train,
        args.raw_dataset,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    prepared_eval = (
        args.experiment_dir
        / "prepared"
        / f"{args.model}-prepared-dataset"
    )
    checkpoint_dir = args.experiment_dir / "checkpoints" / args.model
    result_root = args.experiment_dir / "results" / args.model

    if args.evaluate_pretrained:
        evaluate(
            args,
            args.model_checkpoint,
            prepared_eval,
            result_root / "pre",
            "msa",
        )
    fine_tuned_checkpoint = train(args, prepared_train, checkpoint_dir)
    evaluate(
        args,
        fine_tuned_checkpoint,
        prepared_eval,
        result_root / args.post_label,
        "dynamic",
    )


if __name__ == "__main__":
    main()
