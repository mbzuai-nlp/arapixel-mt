from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from dataset_processing.dialects.step1_utils import (
    RAW_ROOT,
    SPLITS,
    dialect_output_path,
    format_source_text,
    read_tsv_with_header,
    save_dataset_dict,
    split_key,
)

ALEXANDRIA_RAW_ROOT = RAW_ROOT / "Alexandria"
ALEXANDRIA_HEADER = ["Dialect", "Text"]


def load_alexandria_file(path: Path) -> list[tuple[str, str]]:
    rows = read_tsv_with_header(path, ALEXANDRIA_HEADER)
    records: list[tuple[str, str]] = []
    for row in rows:
        if len(row) != len(ALEXANDRIA_HEADER):
            raise ValueError(f"{path} has malformed row: {row!r}")
        label, text = row
        records.append((label.strip(), text.strip()))
    return records


def build_split(
    split: str,
    raw_root: Path = ALEXANDRIA_RAW_ROOT,
) -> dict[str, list[dict[str, str]]]:
    arabic_rows = load_alexandria_file(raw_root / f"{split}.ar.tsv")
    english_rows = load_alexandria_file(raw_root / f"{split}.en.tsv")
    if len(arabic_rows) != len(english_rows):
        raise ValueError(f"Alexandria {split} ar/en length mismatch: {len(arabic_rows)} != {len(english_rows)}")

    rows_by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for index, ((label, source), (_english_label, target)) in enumerate(zip(arabic_rows, english_rows), start=1):
        if not label or not source or not target:
            continue
        bucket = split_key(label)
        rows_by_label[bucket].append(
            {
                "id": f"Alexandria_{split}_{bucket}_{index:06d}",
                "source": format_source_text(source),
                "target": target,
            }
        )

    return dict(sorted(rows_by_label.items()))


def build_all(save_to_disk: bool = True) -> dict[str, object]:
    artifacts: dict[str, object] = {}
    for split in SPLITS:
        rows_by_label = build_split(split)
        output_path = dialect_output_path(f"Alexandria_{split}")
        artifacts[split] = save_dataset_dict(rows_by_label, output_path) if save_to_disk else rows_by_label
    return artifacts


if __name__ == "__main__":
    build_all(save_to_disk=True)
