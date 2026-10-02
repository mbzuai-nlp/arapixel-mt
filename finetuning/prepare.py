"""Prepare one raw training mixture for model-specific dynamic fine-tuning."""

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
        help="Raw Dataset or DatasetDict with source and target columns.",
    )
    add_prepared_dataset_arguments(parser)
    parser.add_argument(
        "--max-source-words",
        type=int,
        help="Pixel/Byte source cap (default: model fine-tuning constant, 64).",
    )
    parser.add_argument(
        "--max-source-tokens",
        type=int,
        help="Base/Vocab+ source cap (default: model fine-tuning constant, 256).",
    )
    parser.add_argument(
        "--max-target-tokens",
        type=int,
        help="Optional English target cap; default keeps all target tokens.",
    )
    parser.add_argument("--num-proc", type=int, default=8)
    return parser.parse_args(), parser


def tokenize_targets(tokenizer, targets, max_target_tokens=None):
    texts = [
        str(target) + tokenizer.eos_token
        if target
        else tokenizer.eos_token
        for target in targets
    ]
    kwargs = {
        "padding": False,
        "add_special_tokens": False,
        "truncation": max_target_tokens is not None,
    }
    if max_target_tokens is not None:
        kwargs["max_length"] = max_target_tokens
    return tokenizer(texts, **kwargs)


def prepare_pixel(
    raw_dataset,
    pixel_prepared_dataset,
    max_source_words=None,
    max_target_tokens=None,
    num_proc=8,
):
    from PixelSmolLM.prepare import prepare_dataset
    from PixelSmolLM.rendering.config import FINETUNE_SOURCE_WORDS

    return prepare_dataset(
        dataset_path=raw_dataset,
        save_path=pixel_prepared_dataset,
        mode="finetune",
        max_source_words=(
            FINETUNE_SOURCE_WORDS
            if max_source_words is None
            else max_source_words
        ),
        max_target_tokens=max_target_tokens,
        num_proc=num_proc,
    )


def prepare_bytes(
    raw_dataset,
    bytes_prepared_dataset,
    max_source_words=None,
    max_target_tokens=None,
    num_proc=8,
):
    from BytesSmolLM.utils import (
        FINETUNE_SOURCE_WORDS,
        MAX_WORD_LENGTH,
        SimpleWordProcessor,
    )

    source_limit = (
        FINETUNE_SOURCE_WORDS
        if max_source_words is None
        else max_source_words
    )
    print(f"Preparing dynamic Byte dataset from {raw_dataset}...")
    dataset = load_from_disk(str(raw_dataset))
    word_processor = SimpleWordProcessor(max_word_length=MAX_WORD_LENGTH)
    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolLM2-360M-Instruct"
    )
    tokenizer.pad_token = tokenizer.eos_token

    def map_bytes(batch):
        all_word_byte_ids = []
        all_attention_masks = []
        all_num_words = []
        for source in batch["source"]:
            tensor = word_processor.process_text(str(source) if source else "")
            if tensor.numel() > 0:
                tensor = tensor[:source_limit]
                word_ids = tensor.tolist()
            else:
                word_ids = []
            all_word_byte_ids.append(word_ids)
            all_attention_masks.append([1] * len(word_ids))
            all_num_words.append(len(word_ids))

        targets = tokenize_targets(
            tokenizer,
            batch["target"],
            max_target_tokens=max_target_tokens,
        )
        return {
            "input_ids": all_word_byte_ids,
            "attention_mask": all_attention_masks,
            "num_words": all_num_words,
            "target_ids": targets["input_ids"],
            "target_mask": targets["attention_mask"],
        }

    encoded = dataset.map(
        map_bytes,
        batched=True,
        num_proc=num_proc,
        desc="Tokenizing bytes for dynamic training",
    )
    encoded.save_to_disk(str(bytes_prepared_dataset))
    print(f"Saved to {bytes_prepared_dataset}")


def _prepare_token_dataset(
    raw_dataset,
    prepared_dataset,
    tokenizer_name_or_path,
    max_source_tokens,
    max_target_tokens,
    num_proc,
):
    dataset = load_from_disk(str(raw_dataset))
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_name_or_path))
    tokenizer.pad_token = tokenizer.eos_token

    def map_batch(batch):
        sources = [str(source) if source else "" for source in batch["source"]]
        source_kwargs = {
            "padding": False,
            "add_special_tokens": False,
            "truncation": max_source_tokens is not None,
        }
        if max_source_tokens is not None:
            source_kwargs["max_length"] = max_source_tokens
        source_encoding = tokenizer(sources, **source_kwargs)
        target_encoding = tokenize_targets(
            tokenizer,
            batch["target"],
            max_target_tokens=max_target_tokens,
        )
        return {
            "source_ids": source_encoding["input_ids"],
            "source_mask": source_encoding["attention_mask"],
            "target_ids": target_encoding["input_ids"],
            "target_mask": target_encoding["attention_mask"],
        }

    encoded = dataset.map(
        map_batch,
        batched=True,
        num_proc=num_proc,
        desc="Tokenizing dynamic training dataset",
    )
    encoded.save_to_disk(str(prepared_dataset))
    print(f"Saved to {prepared_dataset}")


def prepare_decoder(
    raw_dataset,
    decoder_prepared_dataset,
    max_source_tokens=None,
    max_target_tokens=None,
    num_proc=8,
):
    from DecoderOnly.utils import FINETUNE_SOURCE_TOKENS

    _prepare_token_dataset(
        raw_dataset,
        decoder_prepared_dataset,
        "HuggingFaceTB/SmolLM2-360M-Instruct",
        (
            FINETUNE_SOURCE_TOKENS
            if max_source_tokens is None
            else max_source_tokens
        ),
        max_target_tokens,
        num_proc,
    )


def prepare_vocab(
    raw_dataset,
    vocab_prepared_dataset,
    max_source_tokens=None,
    max_target_tokens=None,
    num_proc=8,
):
    from DecoderOnly.utils import FINETUNE_SOURCE_TOKENS

    _prepare_token_dataset(
        raw_dataset,
        vocab_prepared_dataset,
        DEFAULT_VOCAB_TOKENIZER,
        (
            FINETUNE_SOURCE_TOKENS
            if max_source_tokens is None
            else max_source_tokens
        ),
        max_target_tokens,
        num_proc,
    )


def prepare_for_model(
    model: str,
    raw_dataset: Path,
    prepared_dataset: Path,
    *,
    max_source_words: int | None = None,
    max_source_tokens: int | None = None,
    max_target_tokens: int | None = None,
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
        (max_target_tokens, "max-target-tokens"),
    ):
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive")

    functions = {
        "pixel": prepare_pixel,
        "bytes": prepare_bytes,
        "decoder": prepare_decoder,
        "vocab": prepare_vocab,
    }
    kwargs = {
        "max_target_tokens": max_target_tokens,
        "num_proc": num_proc,
    }
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
        max_target_tokens=args.max_target_tokens,
        num_proc=args.num_proc,
    )


if __name__ == "__main__":
    main()
