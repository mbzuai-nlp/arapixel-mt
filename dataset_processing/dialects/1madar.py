from __future__ import annotations

from pathlib import Path

from dataset_processing.dialects.step1_utils import (
    RAW_ROOT,
    SPLITS,
    dialect_output_path,
    format_source_text,
    read_tsv_with_header,
    save_dataset_dict,
)

MADAR_RAW_ROOT = RAW_ROOT / "MADAR"
MADAR_HEADER = ["sentID.BTEC", "split", "lang", "sent"]


def load_madar_file(path: Path) -> dict[str, str]:
    rows = read_tsv_with_header(path, MADAR_HEADER)
    records: dict[str, str] = {}
    for row in rows:
        if len(row) != len(MADAR_HEADER):
            raise ValueError(f"{path} has malformed row: {row!r}")
        sent_id, _split_name, _lang, sentence = row
        sent_id = sent_id.strip()
        sentence = sentence.strip()
        if sent_id and sentence:
            records[sent_id] = sentence
    return records


def city_from_path(path: Path, split: str) -> str:
    prefix = "MADAR."
    suffix = f".{split}.tsv"
    if not path.name.startswith(prefix) or not path.name.endswith(suffix):
        raise ValueError(f"Unexpected MADAR filename: {path.name}")
    return path.name[len(prefix) : -len(suffix)]


def build_split(
    split: str,
    raw_root: Path = MADAR_RAW_ROOT,
) -> dict[str, list[dict[str, str]]]:
    english_path = raw_root / f"MADAR.English.{split}.tsv"
    english_by_id = load_madar_file(english_path)
    rows_by_city: dict[str, list[dict[str, str]]] = {}

    for path in sorted(raw_root.glob(f"MADAR.*.{split}.tsv")):
        city = city_from_path(path, split)
        if city == "English":
            continue

        source_by_id = load_madar_file(path)
        missing_targets = sorted(set(source_by_id) - set(english_by_id))
        if missing_targets:
            preview = ", ".join(missing_targets[:5])
            raise ValueError(f"{path} has {len(missing_targets)} ids missing from English: {preview}")

        rows: list[dict[str, str]] = []
        for sent_id in sorted(source_by_id, key=int):
            source = source_by_id[sent_id].strip()
            target = english_by_id[sent_id].strip()
            if not source or not target:
                continue
            rows.append(
                {
                    "id": f"MADAR_{split}_{city}_{sent_id}",
                    "source": format_source_text(source),
                    "target": target,
                }
            )
        rows_by_city[city] = rows

    return rows_by_city


def build_all(save_to_disk: bool = True) -> dict[str, object]:
    artifacts: dict[str, object] = {}
    for split in SPLITS:
        rows_by_city = build_split(split)
        output_path = dialect_output_path(f"MADAR_{split}")
        artifacts[split] = save_dataset_dict(rows_by_city, output_path) if save_to_disk else rows_by_city
    return artifacts


if __name__ == "__main__":
    build_all(save_to_disk=True)
