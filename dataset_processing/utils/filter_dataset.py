import re
import os
from pathlib import Path

from datasets import load_from_disk, DatasetDict

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))

dataset = load_from_disk(DATA_ROOT / "pixel_translation_MSA")

dataset_splits = DatasetDict(dataset)

# Filter length
def count_words(s: str) -> int:
    return len(re.findall(r"\S+", str(s).strip()))

def filter_len(batch):
    return [
        (count_words(src) <= 32) and (count_words(tgt) <= 32)
        for src, tgt in zip(batch["source"], batch["target"])
    ]

filtered_splits = DatasetDict()
for split in dataset_splits.keys():
    filtered_splits[split] = dataset_splits[split].filter(filter_len, num_proc=6, batched=True, batch_size=5000,
)

filtered_splits.save_to_disk(DATA_ROOT / "pixel_translation_MSA_filtered_32")
