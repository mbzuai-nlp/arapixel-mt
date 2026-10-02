"""Dataset wrappers and collators for 24 x 24 character-bigram patches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .rendering.config import (
    FINETUNE_SOURCE_WORDS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    PATCHES_PER_WORD,
    PRETRAIN_SOURCE_WORDS,
)


REQUIRED_IMAGE_COLUMNS = (
    "word_images",
    "patch_masks",
    "num_words",
    "source",
    "target",
)


def validate_columns(dataset, target_required: bool) -> None:
    required = set(REQUIRED_IMAGE_COLUMNS)
    if target_required:
        required.update(("target_ids", "target_mask"))
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"Pixel dataset is missing columns: {sorted(missing)}")


class PixelTranslationDataset(Dataset):
    def __init__(self, dataset, include_targets: bool = True):
        validate_columns(dataset, target_required=include_targets)
        self.dataset = dataset
        self.include_targets = include_targets

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        item = self.dataset[index]
        num_words = int(item.get("num_words", 0))
        images = []
        masks = []
        for word_index in range(num_words):
            array = np.asarray(
                item["word_images"][word_index], dtype=np.float32
            )
            if array.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
                raise ValueError(
                    f"Unexpected Pixel image shape {array.shape}; expected "
                    f"{(IMAGE_HEIGHT, IMAGE_WIDTH)}"
                )
            rgb = np.stack((array, array, array), axis=0) / 255.0
            images.append(torch.from_numpy(rgb))
            masks.append(
                torch.tensor(item["patch_masks"][word_index], dtype=torch.bool)
            )

        result = {
            "word_images": images,
            "patch_masks": masks,
            "num_words": num_words,
            "sample_index": int(item.get("_eval_index", index)),
            "arabic_texts": str(item.get("source", "")),
            "english_texts": str(item.get("target", "")),
        }
        if self.include_targets:
            result["target_ids"] = torch.tensor(
                item["target_ids"], dtype=torch.long
            )
            result["target_mask"] = torch.tensor(
                item["target_mask"], dtype=torch.long
            )
        return result


def _empty_images():
    return torch.empty(0, 3, IMAGE_HEIGHT, IMAGE_WIDTH)


def _empty_masks():
    return torch.empty(0, PATCHES_PER_WORD, dtype=torch.bool)


@dataclass
class FixedPixelCollator:
    max_source_words: int = PRETRAIN_SOURCE_WORDS

    def __call__(self, features):
        valid = [
            feature
            for feature in features
            if feature["num_words"] > 0 and feature["english_texts"]
        ]
        if not valid:
            return {
                "word_images": _empty_images(),
                "patch_masks": _empty_masks(),
                "word_counts": [],
                "target_ids": torch.empty(0, 0, dtype=torch.long),
                "target_mask": torch.empty(0, 0, dtype=torch.long),
                "arabic_texts": [],
                "english_texts": [],
            }

        all_images, all_masks, counts = [], [], []
        for feature in valid:
            count = min(feature["num_words"], self.max_source_words)
            all_images.extend(feature["word_images"][:count])
            all_masks.extend(feature["patch_masks"][:count])
            counts.append(count)
        return {
            "word_images": torch.stack(all_images),
            "patch_masks": torch.stack(all_masks),
            "word_counts": counts,
            "target_ids": torch.stack([f["target_ids"] for f in valid]),
            "target_mask": torch.stack([f["target_mask"] for f in valid]),
            "arabic_texts": [f["arabic_texts"] for f in valid],
            "english_texts": [f["english_texts"] for f in valid],
        }


@dataclass
class DynamicPixelTrainCollator:
    pad_token_id: int
    max_source_words: int = FINETUNE_SOURCE_WORDS

    def __call__(self, features):
        valid = [
            feature
            for feature in features
            if feature["num_words"] > 0 and feature["english_texts"]
        ]
        if not valid:
            return {
                "word_images": _empty_images(),
                "patch_masks": _empty_masks(),
                "word_counts": [],
                "target_ids": torch.empty(0, 0, dtype=torch.long),
                "target_mask": torch.empty(0, 0, dtype=torch.long),
                "arabic_texts": [],
                "english_texts": [],
            }

        all_images, all_masks, counts = [], [], []
        for feature in valid:
            count = min(feature["num_words"], self.max_source_words)
            all_images.extend(feature["word_images"][:count])
            all_masks.extend(feature["patch_masks"][:count])
            counts.append(count)
        return {
            "word_images": torch.stack(all_images),
            "patch_masks": torch.stack(all_masks),
            "word_counts": counts,
            "target_ids": pad_sequence(
                [f["target_ids"] for f in valid],
                batch_first=True,
                padding_value=self.pad_token_id,
            ),
            "target_mask": pad_sequence(
                [f["target_mask"] for f in valid],
                batch_first=True,
                padding_value=0,
            ),
            "arabic_texts": [f["arabic_texts"] for f in valid],
            "english_texts": [f["english_texts"] for f in valid],
        }


@dataclass
class PixelEvalCollator:
    fixed_source_length: Optional[int] = None

    def __call__(self, features):
        valid = [
            feature
            for feature in features
            if feature["num_words"] > 0 and feature["english_texts"].strip()
        ]
        if not valid:
            return {
                "word_images": _empty_images(),
                "patch_masks": _empty_masks(),
                "word_counts": [],
                "sample_indices": [],
                "arabic_texts": [],
                "english_texts": [],
            }

        all_images, all_masks, counts = [], [], []
        for feature in valid:
            count = feature["num_words"]
            if self.fixed_source_length is not None:
                count = min(count, self.fixed_source_length)
            all_images.extend(feature["word_images"][:count])
            all_masks.extend(feature["patch_masks"][:count])
            counts.append(count)
        return {
            "word_images": torch.stack(all_images),
            "patch_masks": torch.stack(all_masks),
            "word_counts": counts,
            "sample_indices": [f["sample_index"] for f in valid],
            "arabic_texts": [f["arabic_texts"] for f in valid],
            "english_texts": [f["english_texts"] for f in valid],
        }


def build_eval_dataloader(
    model_type,
    dataset,
    tokenizer,
    batch_size,
    fixed_source_length=None,
):
    if model_type != "pixel":
        raise ValueError("The Pixel model only supports model_type=pixel")
    if fixed_source_length is not None and fixed_source_length <= 0:
        raise ValueError("fixed_source_length must be positive")
    return DataLoader(
        PixelTranslationDataset(dataset, include_targets=False),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=PixelEvalCollator(fixed_source_length),
        num_workers=4,
        pin_memory=True,
    )

