"""Shared public CLI vocabulary for model and prepared-dataset arguments."""

from __future__ import annotations

import argparse
from pathlib import Path


MODEL_CHOICES = ("pixel", "bytes", "vocab", "decoder")
PREPARED_DATASET_FLAGS = {
    "pixel": "--pixel-prepared-dataset",
    "bytes": "--bytes-prepared-dataset",
    "decoder": "--decoder-prepared-dataset",
    "vocab": "--vocab-prepared-dataset",
}


def add_model_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)


def add_prepared_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    for model, flag in PREPARED_DATASET_FLAGS.items():
        parser.add_argument(
            flag,
            dest=f"{model}_prepared_dataset",
            type=Path,
            help=f"Prepared {model} DatasetDict path.",
        )


def prepared_dataset_for(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser | None = None,
    *,
    required: bool = True,
) -> Path | None:
    selected = getattr(args, f"{args.model}_prepared_dataset")
    supplied = {
        model: getattr(args, f"{model}_prepared_dataset")
        for model in MODEL_CHOICES
        if getattr(args, f"{model}_prepared_dataset") is not None
    }
    message = None
    if selected is None and required:
        message = (
            f"{PREPARED_DATASET_FLAGS[args.model]} is required when "
            f"--model {args.model} is selected"
        )
    elif selected is None and supplied:
        extras = ", ".join(
            PREPARED_DATASET_FLAGS[model] for model in supplied
        )
        message = (
            f"Prepared-dataset argument does not match --model {args.model}: "
            f"{extras}"
        )
    elif selected is not None and set(supplied) != {args.model}:
        extras = ", ".join(
            PREPARED_DATASET_FLAGS[model]
            for model in supplied
            if model != args.model
        )
        message = (
            f"Prepared-dataset argument does not match --model {args.model}: "
            f"{extras}"
        )
    if message is not None:
        if parser is not None:
            parser.error(message)
        raise ValueError(message)
    return selected
