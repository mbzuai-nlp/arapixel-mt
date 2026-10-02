"""Prepare a raw evaluation DatasetDict for one model family."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer

from training.cli import (
    add_model_argument,
    add_prepared_dataset_arguments,
    prepared_dataset_for,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VOCAB_TOKENIZER = Path(
    os.environ.get(
        "VOCAB_TOKENIZER_DIR",
        REPO_ROOT / "DecoderOnly" / "VocabPlus" / "merged_tokenizer_32000",
    )
)


def parse_args() -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_argument(parser)
    parser.add_argument(
        "--raw-dataset",
        type=Path,
        required=True,
        help="Raw DatasetDict whose splits contain source and target columns.",
    )
    add_prepared_dataset_arguments(parser)
    parser.add_argument(
        "--max-source-words",
        type=int,
        help="Optional Pixel/Byte preparation cap; default keeps all words.",
    )
    parser.add_argument(
        "--max-source-tokens",
        type=int,
        help="Optional Base/Vocab+ preparation cap; default keeps all tokens.",
    )
    parser.add_argument("--num-proc", type=int, default=8)
    return parser.parse_args(), parser


def prepare_pixel(
    raw_dataset,
    pixel_prepared_dataset,
    max_source_words=None,
    num_proc=8,
):
    from PixelSmolLM.prepare import prepare_dataset

    return prepare_dataset(
        dataset_path=raw_dataset,
        save_path=pixel_prepared_dataset,
        mode="eval",
        max_source_words=max_source_words,
        num_proc=num_proc,
    )


def prepare_bytes(
    raw_dataset,
    bytes_prepared_dataset,
    max_source_words=None,
    num_proc=8,
):
    from BytesSmolLM.utils import MAX_WORD_LENGTH, SimpleWordProcessor

    print(f"Preparing dynamic Byte evaluation data from {raw_dataset}...")
    dataset = load_from_disk(str(raw_dataset))
    processor = SimpleWordProcessor(max_word_length=MAX_WORD_LENGTH)

    def map_bytes(batch):
        ids, masks, counts = [], [], []
        for source in batch["source"]:
            tensor = processor.process_text(str(source) if source else "")
            if max_source_words is not None and tensor.numel() > 0:
                tensor = tensor[:max_source_words]
            word_ids = tensor.tolist() if tensor.numel() > 0 else []
            ids.append(word_ids)
            masks.append([1] * len(word_ids))
            counts.append(len(word_ids))
        return {
            "input_ids": ids,
            "attention_mask": masks,
            "num_words": counts,
        }

    encoded = dataset.map(
        map_bytes,
        batched=True,
        num_proc=num_proc,
        desc="Tokenizing bytes for evaluation",
    )
    encoded.save_to_disk(str(bytes_prepared_dataset))
    print(f"Saved to {bytes_prepared_dataset}")


def _prepare_token_dataset(
    raw_dataset,
    prepared_dataset,
    tokenizer_name_or_path,
    max_source_tokens=None,
    num_proc=8,
):
    dataset = load_from_disk(str(raw_dataset))
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_name_or_path))
    tokenizer.pad_token = tokenizer.eos_token

    def map_batch(batch):
        sources = [str(source) if source else "" for source in batch["source"]]
        kwargs = {
            "truncation": max_source_tokens is not None,
            "padding": False,
            "add_special_tokens": False,
        }
        if max_source_tokens is not None:
            kwargs["max_length"] = max_source_tokens
        encoded = tokenizer(sources, **kwargs)
        return {
            "source_ids": encoded["input_ids"],
            "source_mask": encoded["attention_mask"],
        }

    encoded = dataset.map(
        map_batch,
        batched=True,
        num_proc=num_proc,
        desc="Tokenizing sources for evaluation",
    )
    encoded.save_to_disk(str(prepared_dataset))
    print(f"Saved to {prepared_dataset}")


def prepare_decoder(
    raw_dataset,
    decoder_prepared_dataset,
    max_source_tokens=None,
    num_proc=8,
):
    _prepare_token_dataset(
        raw_dataset,
        decoder_prepared_dataset,
        "HuggingFaceTB/SmolLM2-360M-Instruct",
        max_source_tokens=max_source_tokens,
        num_proc=num_proc,
    )


def prepare_vocab(
    raw_dataset,
    vocab_prepared_dataset,
    max_source_tokens=None,
    num_proc=8,
):
    _prepare_token_dataset(
        raw_dataset,
        vocab_prepared_dataset,
        DEFAULT_VOCAB_TOKENIZER,
        max_source_tokens=max_source_tokens,
        num_proc=num_proc,
    )


def prepare_for_model(
    model: str,
    raw_dataset: Path,
    prepared_dataset: Path,
    *,
    max_source_words: int | None = None,
    max_source_tokens: int | None = None,
    num_proc: int = 8,
) -> None:
    if not raw_dataset.is_dir():
        raise FileNotFoundError(raw_dataset)
    if prepared_dataset.exists():
        raise FileExistsError(
            f"Refusing to overwrite prepared dataset: {prepared_dataset}"
        )
    if num_proc <= 0:
        raise ValueError("num-proc must be positive")
    for value, name in (
        (max_source_words, "max-source-words"),
        (max_source_tokens, "max-source-tokens"),
    ):
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive")

    functions = {
        "pixel": prepare_pixel,
        "bytes": prepare_bytes,
        "decoder": prepare_decoder,
        "vocab": prepare_vocab,
    }
    kwargs = {"num_proc": num_proc}
    if model in ("pixel", "bytes"):
        kwargs["max_source_words"] = max_source_words
    else:
        kwargs["max_source_tokens"] = max_source_tokens
    functions[model](raw_dataset, prepared_dataset, **kwargs)


def main() -> None:
    args, parser = parse_args()
    prepared_dataset = prepared_dataset_for(args, parser)
    prepare_for_model(
        args.model,
        args.raw_dataset,
        prepared_dataset,
        max_source_words=args.max_source_words,
        max_source_tokens=args.max_source_tokens,
        num_proc=args.num_proc,
    )


if __name__ == "__main__":
    main()
