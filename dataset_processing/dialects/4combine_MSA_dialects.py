import os
from pathlib import Path

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

from dataset_processing.dialects.step1_utils import DATA_ROOT, PROCESSED_ROOT, hf_disk_path

MSA_DATASET_PATH = Path(os.environ.get("MSA_DATASET_PATH", DATA_ROOT / "pixel_translation_MSA_filtered_32"))
DIALECT_TRAIN_DATASETDICT_PATHS = [
    PROCESSED_ROOT / "arabench_train",
    PROCESSED_ROOT / "MADAR_train",
    PROCESSED_ROOT / "Alexandria_train",
]


def _to_source_target(dataset: Dataset) -> Dataset:
    extra_columns = [column for column in dataset.column_names if column not in {"source", "target"}]
    if extra_columns:
        dataset = dataset.remove_columns(extra_columns)
    return dataset


def _flatten_dataset_dict(dataset_dict: DatasetDict) -> Dataset:
    return concatenate_datasets([_to_source_target(dataset) for dataset in dataset_dict.values()])


def load_dialect_train() -> Dataset:
    datasets = [
        _flatten_dataset_dict(load_from_disk(hf_disk_path(path)))
        for path in DIALECT_TRAIN_DATASETDICT_PATHS
    ]
    return concatenate_datasets(datasets).shuffle(seed=42)


def main() -> None:
    dialect_train = load_dialect_train()
    dialect_output_path = PROCESSED_ROOT / "Dialect_Combined"
    dialect_train.save_to_disk(hf_disk_path(dialect_output_path))
    print(f"Saved dialect-only train dataset to {dialect_output_path}")
    print(dialect_train)

    MSA = load_from_disk(hf_disk_path(MSA_DATASET_PATH))
    msa_train = MSA["train"].shuffle(seed=42)

    ratios = [1, 2, 4]
    dialect_size = len(dialect_train)

    for ratio in ratios:
        size = dialect_size * ratio
        MSA_sampled = msa_train.select(range(size))

        combined_dataset = concatenate_datasets([dialect_train, MSA_sampled])
        combined_dataset = combined_dataset.shuffle(seed=42)

        output_path = PROCESSED_ROOT / f"MSA_Dialect_Combined_{ratio}x"
        combined_dataset.save_to_disk(hf_disk_path(output_path))
        print(combined_dataset)


if __name__ == "__main__":
    main()
