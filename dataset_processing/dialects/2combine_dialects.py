from __future__ import annotations

from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk

from dataset_processing.dialects.step1_utils import (
    PROCESSED_ROOT,
    SPLITS,
    dialect_output_path,
    hf_disk_path,
    split_key,
)

SKIP_INPUTS = {
    "dialects_train",
    "dialects_dev",
    "dialects_test",
    "dialects_dev_noisy",
    "dialects_test_noisy",
}


def _step1_split(path: Path) -> str | None:
    name = path.name

    if name in SKIP_INPUTS:
        return None
    if not (path / "dataset_dict.json").exists():
        return None

    for split in SPLITS:
        if name.endswith(f"_{split}"):
            return split
    return None


def _normalize_dataset_columns(dataset: Dataset, source_path: Path, split_name: str) -> Dataset:
    required = {"id", "source", "target"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"{source_path}/{split_name} is missing required columns: {sorted(missing)}")

    extra_columns = [column for column in dataset.column_names if column not in required]
    if extra_columns:
        dataset = dataset.remove_columns(extra_columns)
    return dataset


def _combined_split_name(
    source_path: Path,
    step1_split: str,
    split_name: str,
) -> str:
    source_name = source_path.name
    split_suffix = f"_{step1_split}"
    dataset_name = source_name[: -len(split_suffix)] if source_name.endswith(split_suffix) else source_name
    return f"{split_key(dataset_name)}_{split_key(split_name)}"


def build_combined_datasets(
    processed_root: Path | None = None,
) -> dict[str, DatasetDict]:
    processed_root = processed_root or PROCESSED_ROOT
    combined_by_split = {split: DatasetDict() for split in SPLITS}

    for source_path in sorted(processed_root.iterdir()):
        if not source_path.is_dir():
            continue

        step1_split = _step1_split(source_path)
        if step1_split is None:
            continue

        dataset_dict = load_from_disk(hf_disk_path(source_path))
        combined = combined_by_split[step1_split]
        for split_name, dataset in dataset_dict.items():
            combined_name = _combined_split_name(
                source_path,
                step1_split,
                split_name,
            )
            if combined_name in combined:
                raise ValueError(f"Duplicate combined split name: {combined_name}")
            combined[combined_name] = _normalize_dataset_columns(dataset, source_path, split_name)

    empty = [split for split, dataset_dict in combined_by_split.items() if not dataset_dict]
    if empty:
        raise ValueError(f"No Step 1 DatasetDict artifacts found for split(s): {', '.join(empty)}")

    return combined_by_split


def main() -> None:
    combined_by_split = build_combined_datasets()
    for split, dataset_dict in combined_by_split.items():
        output_path = dialect_output_path(f"dialects_{split}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving {split} dataset to {output_path}...")
        dataset_dict.save_to_disk(hf_disk_path(output_path))
        print(dataset_dict)


if __name__ == "__main__":
    main()
