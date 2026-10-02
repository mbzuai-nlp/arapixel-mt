"""Prepare one model's nested MSA prefix through its public preparer."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("pixel", "bytes", "decoder", "vocab"),
        required=True,
    )
    parser.add_argument("--percentage", type=int, required=True)
    parser.add_argument("--raw-dataset", type=Path, required=True)
    parser.add_argument("--pixel-prepared-dataset", type=Path)
    parser.add_argument("--bytes-prepared-dataset", type=Path)
    parser.add_argument("--num-proc", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.percentage <= 100:
        raise ValueError("percentage must be in (0, 100]")
    if not args.raw_dataset.is_dir():
        raise FileNotFoundError(args.raw_dataset)

    if args.model in ("decoder", "vocab"):
        if (
            args.pixel_prepared_dataset is not None
            or args.bytes_prepared_dataset is not None
        ):
            raise ValueError(
                "Base and Vocab+ consume --raw-dataset directly; "
                "omit prepared-dataset arguments."
            )
        print(f"{args.model} needs no model-specific preparation.")
        return

    if args.model == "pixel":
        if args.pixel_prepared_dataset is None:
            raise ValueError(
                "--pixel-prepared-dataset is required for --model pixel"
            )
        if args.bytes_prepared_dataset is not None:
            raise ValueError(
                "--bytes-prepared-dataset does not match --model pixel"
            )
        from PixelSmolLM.prepare import prepare_dataset
        from PixelSmolLM.rendering.config import (
            PRETRAIN_SOURCE_WORDS,
            TARGET_TOKENS,
        )

        prepare_dataset(
            dataset_path=args.raw_dataset,
            save_path=args.pixel_prepared_dataset,
            mode="msa",
            percentage=args.percentage,
            max_source_words=PRETRAIN_SOURCE_WORDS,
            max_target_tokens=TARGET_TOKENS,
            num_proc=args.num_proc,
            batch_size=args.batch_size,
        )
        return

    if args.bytes_prepared_dataset is None:
        raise ValueError(
            "--bytes-prepared-dataset is required for --model bytes"
        )
    if args.pixel_prepared_dataset is not None:
        raise ValueError(
            "--pixel-prepared-dataset does not match --model bytes"
        )
    from BytesSmolLM.prepare import prepare_dataset

    prepare_dataset(
        raw_dataset=args.raw_dataset,
        save_path=args.bytes_prepared_dataset,
        percentage=args.percentage,
        num_proc=args.num_proc,
        batch_size=max(args.batch_size, 1_000),
    )


if __name__ == "__main__":
    main()
