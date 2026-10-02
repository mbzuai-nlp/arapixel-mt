"""Create datasets for the contextual Pixel renderer."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image as HFImage
from datasets import Sequence, Value, load_from_disk
from PIL import Image
from transformers import AutoTokenizer

from .rendering.config import (
    PATCHES_PER_WORD,
    PRETRAIN_SOURCE_WORDS,
    TARGET_TOKENS,
    RENDERER_SPEC,
)


BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
_renderer = None
_tokenizer = None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dataset", dest="dataset_path", type=Path, required=True)
    parser.add_argument(
        "--pixel-prepared-dataset",
        dest="save_path",
        type=Path,
        required=True,
        help="Output path for the prepared contextual Pixel DatasetDict.",
    )
    parser.add_argument(
        "--percentage",
        type=int,
        default=100,
        help="Nested MSA prefix to prepare (default: 100).",
    )
    parser.add_argument(
        "--max-source-words",
        type=int,
        default=None,
        help="Maximum source words (default: the MSA training limit, 32).",
    )
    parser.add_argument("--max-target-tokens", type=int, default=None)
    parser.add_argument("--validation-rows", type=int, default=2000)
    parser.add_argument("--num-proc", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def get_renderer():
    global _renderer
    if _renderer is None:
        from .rendering.contextual import ContextualBigramRenderer

        _renderer = ContextualBigramRenderer()
    return _renderer


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        _tokenizer.pad_token = _tokenizer.eos_token
    return _tokenizer


def render_source(text: str, max_source_words: int | None):
    from .rendering.contextual import pack_word_patches

    renderer = get_renderer()
    _, _, per_word_patches = renderer.render_text(str(text) if text else "")
    images, masks = [], []
    selected_words = (
        per_word_patches
        if max_source_words is None
        else per_word_patches[:max_source_words]
    )
    for patches in selected_words:
        if not patches:
            continue
        array, mask = pack_word_patches(patches, PATCHES_PER_WORD)
        images.append(Image.fromarray(array))
        masks.append(mask)
    return images, masks


def target_encoding(tokenizer, targets, mode: str, max_target_tokens):
    texts = [
        (str(target) if target else "") + tokenizer.eos_token
        for target in targets
    ]
    if mode == "msa":
        limit = max_target_tokens or TARGET_TOKENS
        return tokenizer(
            texts,
            truncation=True,
            max_length=limit,
            padding="max_length",
        )
    kwargs = {
        "truncation": max_target_tokens is not None,
        "padding": False,
        "add_special_tokens": False,
    }
    if max_target_tokens is not None:
        kwargs["max_length"] = max_target_tokens
    return tokenizer(texts, **kwargs)


def add_features(original, include_targets: bool):
    values = dict(original)
    values.update(
        {
            "word_images": Sequence(HFImage()),
            "patch_masks": Sequence(Sequence(Value("int8"))),
            "num_words": Value("int32"),
        }
    )
    if include_targets:
        values.update(
            {
                "target_ids": Sequence(Value("int32")),
                "target_mask": Sequence(Value("int8")),
            }
        )
    return Features(values)


def prepare_split(
    split: Dataset,
    mode: str,
    max_source_words: int | None,
    max_target_tokens,
    num_proc: int,
    batch_size: int,
    cache_file: Path,
):
    include_targets = mode != "eval"
    # Construct once in the parent and pickle it into map workers. Calling
    # from_pretrained independently in every worker causes unnecessary Hub
    # metadata checks and can stall on compute nodes without network access.
    tokenizer = get_tokenizer() if include_targets else None

    def map_batch(examples):
        all_images, all_masks, all_counts = [], [], []
        for source in examples["source"]:
            images, masks = render_source(source, max_source_words)
            all_images.append(images)
            all_masks.append(masks)
            all_counts.append(len(images))
        result = {
            "word_images": all_images,
            "patch_masks": all_masks,
            "num_words": all_counts,
        }
        if include_targets:
            encoded = target_encoding(
                tokenizer, examples["target"], mode, max_target_tokens
            )
            result["target_ids"] = encoded["input_ids"]
            result["target_mask"] = encoded["attention_mask"]
        return result

    workers = max(1, min(num_proc, len(split)))
    map_kwargs = {}
    if workers > 1:
        map_kwargs["num_proc"] = workers
    return split.map(
        map_batch,
        batched=True,
        batch_size=batch_size,
        writer_batch_size=batch_size,
        features=add_features(split.features, include_targets),
        desc=f"contextual Pixel rendering ({mode})",
        cache_file_name=str(cache_file),
        **map_kwargs,
    )


def select_input(dataset, args):
    if args.mode != "msa":
        return dataset, {}
    if not isinstance(dataset, DatasetDict):
        raise ValueError("MSA input must be a DatasetDict")
    if "train" not in dataset or "validation" not in dataset:
        raise ValueError("MSA input needs train and validation splits")
    total = len(dataset["train"])
    if total == 0:
        raise ValueError("MSA train split is empty")
    subset_rows = max(1, total * args.percentage // 100)
    selected = DatasetDict(
        {
            "train": dataset["train"].select(range(subset_rows)),
            "validation": dataset["validation"].select(
                range(min(args.validation_rows, len(dataset["validation"])))
            ),
        }
    )
    return selected, {
        "selection": "prefix_of_input_training_split",
        "percentage": args.percentage,
        "original_train_rows": total,
        "selected_train_rows": subset_rows,
        "selected_validation_rows": len(selected["validation"]),
    }


def write_metadata(path: Path, payload):
    metadata = path / "pixel_metadata.json"
    temporary = path / f".pixel_metadata.json.tmp.{os.getpid()}"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, metadata)


def prepare_dataset(
    dataset_path: Path | str,
    save_path: Path | str,
    mode: str,
    max_source_words: int,
    max_target_tokens: int | None = None,
    percentage: int = 100,
    validation_rows: int = 2000,
    num_proc: int = 32,
    batch_size: int = 128,
):
    args = argparse.Namespace(
        dataset_path=Path(dataset_path),
        save_path=Path(save_path),
        mode=mode,
        max_source_words=max_source_words,
        max_target_tokens=max_target_tokens,
        percentage=percentage,
        validation_rows=validation_rows,
        num_proc=num_proc,
        batch_size=batch_size,
    )
    if not args.dataset_path.is_dir():
        raise FileNotFoundError(args.dataset_path)
    if args.save_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing path: {args.save_path}"
        )
    if args.max_source_words is not None and args.max_source_words <= 0:
        raise ValueError("max-source-words must be positive when provided")
    if args.mode == "msa" and args.max_source_words is None:
        raise ValueError("MSA preparation requires max-source-words")
    if args.num_proc <= 0 or args.batch_size <= 0:
        raise ValueError("num-proc and batch-size must be positive")
    if args.mode == "msa" and not 0 < args.percentage <= 100:
        raise ValueError("MSA percentage must be in (0, 100]")

    source = load_from_disk(str(args.dataset_path))
    selected, selection = select_input(source, args)
    map_cache = args.save_path.with_name(f"{args.save_path.name}.map-cache")
    if map_cache.exists():
        raise FileExistsError(f"Refusing existing map cache: {map_cache}")
    map_cache.mkdir(parents=True)
    if isinstance(selected, DatasetDict):
        rendered = DatasetDict(
            {
                name: prepare_split(
                    split,
                    args.mode,
                    args.max_source_words,
                    args.max_target_tokens,
                    args.num_proc,
                    args.batch_size,
                    map_cache / f"{name}.arrow",
                )
                for name, split in selected.items()
            }
        )
    else:
        rendered = prepare_split(
            selected,
            args.mode,
            args.max_source_words,
            args.max_target_tokens,
            args.num_proc,
            args.batch_size,
            map_cache / "dataset.arrow",
        )

    save_kwargs = {}
    if args.num_proc > 1:
        save_kwargs["num_proc"] = args.num_proc
    rendered.save_to_disk(str(args.save_path), **save_kwargs)
    lengths = (
        {name: len(split) for name, split in rendered.items()}
        if isinstance(rendered, DatasetDict)
        else {"dataset": len(rendered)}
    )
    write_metadata(
        args.save_path,
        {
            "schema_version": 1,
            "mode": args.mode,
            "source_dataset": str(args.dataset_path.resolve()),
            "max_source_words": args.max_source_words,
            "max_target_tokens": args.max_target_tokens,
            "lengths": lengths,
            "renderer": RENDERER_SPEC,
            **selection,
        },
    )
    # The saved dataset is self-contained. Release Arrow memory maps before
    # removing the temporary shards used during preparation.
    del rendered
    gc.collect()
    try:
        shutil.rmtree(map_cache)
    except OSError as error:
        # A cleanup failure only leaves a disposable cache; the prepared
        # dataset itself is already complete.
        print(
            f"Warning: could not remove temporary map cache {map_cache}: "
            f"{error}",
            file=sys.stderr,
            flush=True,
        )
    print(f"Saved complete contextual Pixel dataset to {args.save_path}", flush=True)
    return args.save_path


def main():
    args = parse_args()
    if args.max_source_words is None:
        args.max_source_words = PRETRAIN_SOURCE_WORDS
    if args.max_target_tokens is None:
        args.max_target_tokens = TARGET_TOKENS
    args.mode = "msa"
    prepare_dataset(**vars(args))


if __name__ == "__main__":
    main()
