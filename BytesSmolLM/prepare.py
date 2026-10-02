"""Build the image-free prepared MSA dataset used by the Byte model."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path

from datasets import DatasetDict, Features, Sequence, Value, load_from_disk
from transformers import AutoTokenizer

from dataset_processing.utils.tokenize_dataset import tokenize_batch


BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
DEFAULT_PERCENTAGE = 100
VALIDATION_ROWS = 2_000
_tokenizer = None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dataset", type=Path, required=True)
    parser.add_argument(
        "--bytes-prepared-dataset",
        dest="save_path",
        type=Path,
        required=True,
        help="Output path for the prepared Byte DatasetDict.",
    )
    parser.add_argument(
        "--percentage",
        type=int,
        default=DEFAULT_PERCENTAGE,
        help="Nested MSA prefix to prepare (default: 100).",
    )
    parser.add_argument("--validation-rows", type=int, default=VALIDATION_ROWS)
    parser.add_argument("--num-proc", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=7_000)
    return parser.parse_args()


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        _tokenizer.pad_token = _tokenizer.eos_token
    return _tokenizer


def encode_batch(examples):
    return tokenize_batch(examples, get_tokenizer())


def compact_features():
    return Features(
        {
            "source": Value("string"),
            "target": Value("string"),
            "word_byte_ids": Sequence(Sequence(Value("int16"))),
            "byte_word_counts": Value("int32"),
            "target_ids": Sequence(Value("int32")),
            "target_mask": Sequence(Value("int8")),
        }
    )


def write_json_atomic(path: Path, payload):
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def prepare_dataset(
    raw_dataset: Path | str,
    save_path: Path | str,
    percentage: int = DEFAULT_PERCENTAGE,
    validation_rows: int = VALIDATION_ROWS,
    num_proc: int = 32,
    batch_size: int = 7_000,
):
    args = argparse.Namespace(
        raw_dataset=Path(raw_dataset),
        save_path=Path(save_path),
        percentage=percentage,
        validation_rows=validation_rows,
        num_proc=num_proc,
        batch_size=batch_size,
    )
    if not 0 < args.percentage <= 100:
        raise ValueError("percentage must be in (0, 100]")
    if args.num_proc <= 0 or args.batch_size <= 0 or args.validation_rows <= 0:
        raise ValueError("num-proc, batch-size, and validation-rows must be positive")
    if not args.raw_dataset.is_dir():
        raise FileNotFoundError(args.raw_dataset)
    if args.save_path.exists():
        raise FileExistsError(f"Refusing to overwrite {args.save_path}")

    raw = load_from_disk(str(args.raw_dataset))
    if not isinstance(raw, DatasetDict):
        raise ValueError("Raw MSA input must be a DatasetDict")
    if "train" not in raw or "validation" not in raw:
        raise ValueError("Raw MSA data needs train and validation splits")
    total_train_rows = len(raw["train"])
    if total_train_rows == 0:
        raise ValueError("Raw MSA train split is empty")
    selected_train_rows = max(1, total_train_rows * args.percentage // 100)
    selected = DatasetDict(
        {
            "train": raw["train"].select(range(selected_train_rows)),
            "validation": raw["validation"].select(
                range(min(args.validation_rows, len(raw["validation"])))
            ),
        }
    )

    map_cache = args.save_path.with_name(f"{args.save_path.name}.map-cache")
    if map_cache.exists():
        raise FileExistsError(f"Refusing existing map cache: {map_cache}")
    map_cache.mkdir(parents=True)

    encoded = DatasetDict()
    try:
        for split_name, split in selected.items():
            print(
                f"[byte-compact] Encoding {split_name}: {len(split):,} rows "
                f"with num_proc={args.num_proc}",
                flush=True,
            )
            encoded[split_name] = split.map(
                encode_batch,
                batched=True,
                batch_size=args.batch_size,
                num_proc=args.num_proc,
                features=compact_features(),
                cache_file_name=str(map_cache / f"{split_name}.arrow"),
                desc=f"Compact Byte encoding ({split_name})",
            )

        encoded.save_to_disk(str(args.save_path), num_proc=args.num_proc)
        write_json_atomic(
            args.save_path / "byte_compact_metadata.json",
            {
                "schema_version": 1,
                "source_dataset": str(args.raw_dataset.resolve()),
                "selection": "prefix_of_input_training_split",
                "original_train_rows": total_train_rows,
                "percentage": args.percentage,
                "selected_train_rows": selected_train_rows,
                "selected_validation_rows": len(selected["validation"]),
                "source_words": 32,
                "bytes_per_word": 32,
                "target_tokens": 64,
                "byte_pad_token_id": 256,
                "columns": list(compact_features().keys()),
            },
        )
    finally:
        del encoded
        gc.collect()
        try:
            shutil.rmtree(map_cache)
        except OSError as error:
            print(
                f"Warning: could not remove temporary map cache {map_cache}: {error}",
                file=sys.stderr,
                flush=True,
            )

    print(f"Saved compact Byte dataset to {args.save_path}", flush=True)
    return args.save_path


def main():
    args = parse_args()
    prepare_dataset(**vars(args))


if __name__ == "__main__":
    main()
