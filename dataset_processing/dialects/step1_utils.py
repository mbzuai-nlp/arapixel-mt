from __future__ import annotations

import csv
import os
from functools import lru_cache
from pathlib import Path

from datasets import Dataset, DatasetDict

from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import (
    normalize_alef_ar,
    normalize_alef_maksura_ar,
    normalize_teh_marbuta_ar,
)

DEFAULT_DATA_ROOT = Path("data")
DATA_ROOT = Path(os.environ.get("DATA_ROOT", DEFAULT_DATA_ROOT))
DIALECTS_ROOT = DATA_ROOT / "dialects"
RAW_ROOT = DIALECTS_ROOT / "raw"
PROCESSED_ROOT = DIALECTS_ROOT / "processed"

SPLITS = ("train", "dev", "test")


@lru_cache(maxsize=250_000)
def normalize_arabic_text(text: str) -> str:
    text = normalize_alef_ar(text)
    text = normalize_teh_marbuta_ar(text)
    text = normalize_alef_maksura_ar(text)
    text = dediac_ar(text)

    numeral_map = {
        "٠": "0",
        "١": "1",
        "٢": "2",
        "٣": "3",
        "٤": "4",
        "٥": "5",
        "٦": "6",
        "٧": "7",
        "٨": "8",
        "٩": "9",
    }
    for arabic, western in numeral_map.items():
        text = text.replace(arabic, western)

    return " ".join(simple_word_tokenize(text))


def split_key(value: str) -> str:
    return "_".join(value.strip().split())


def format_source_text(text: str) -> str:
    return normalize_arabic_text(text.strip())


def dialect_output_path(name: str, root: Path | None = None) -> Path:
    return (root or PROCESSED_ROOT) / name


def hf_disk_path(path: Path) -> str:
    if path.is_absolute():
        return path.as_uri()
    return str(path)


def read_tsv_with_header(path: Path, expected_header: list[str]) -> list[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader, None)
        if header != expected_header:
            raise ValueError(f"{path} has header {header!r}; expected {expected_header!r}")
        return [row for row in reader]


def dataset_dict_from_rows(rows_by_split: dict[str, list[dict[str, str]] | Dataset]) -> DatasetDict:
    return DatasetDict(
        {
            split_name: rows if isinstance(rows, Dataset) else Dataset.from_list(rows)
            for split_name, rows in rows_by_split.items()
        }
    )


def save_dataset_dict(rows_by_split: dict[str, list[dict[str, str]] | Dataset], output_path: Path) -> DatasetDict:
    dataset_dict = dataset_dict_from_rows(rows_by_split)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(hf_disk_path(output_path))
    print(f"Saved {output_path}")
    print(dataset_dict)
    return dataset_dict
