# Diacritics and Dotless datasets generator
from __future__ import annotations

import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from datasets import Dataset
from datasets import DatasetDict, load_from_disk

from dataset_processing.dialects.step1_utils import dialect_output_path, hf_disk_path

from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.morphology.database import MorphologyDB
from camel_tools.tokenizers.word import simple_word_tokenize

DEFAULT_INPUT_SPLITS = ("dev", "test")
DEFAULT_NUM_WORKERS = max(1, min(4, os.cpu_count() or 1))
LOCAL_DB_DIR = Path(__file__).resolve().parent
BERT_MODEL_BY_DB = {
    "calima-msa-s31": "msa",
    "calima-egy-c044": "egy",
}
LOCAL_DB_PATH_BY_DB = {
    db_name: LOCAL_DB_DIR / f"{db_name}.db"
    for db_name in BERT_MODEL_BY_DB
}

REMOVE_DOTS_RE = re.compile(r'\w+|[^\w]')


def remove_dots(text: str) -> str:
    """Returns a versiona of a given string with dots removed.

    Args:
        input_str (str): The string to remove dots from.

    Returns:
        str: The string with dots removed.
    """

    # Account for problematic word-final letters followed by punctuation
    words = REMOVE_DOTS_RE.findall(text)
    return ''.join(remove_dots_arabic_word(word) for word in words)


# Dotless equivalents (Unicode approximations or best-fit mappings)
DOTLESS_MAP = {
    'ب': 'ٮ',
    'ت': 'ٮ',
    'ث': 'ٮ',
    'ج': 'ح',
    'خ': 'ح',
    'ذ': 'د',
    'ز': 'ر',
    'ش': 'س',
    'ض': 'ص',
    'ظ': 'ط',
    'غ': 'ع',
    'ف': 'ٯ',
    'ق': 'ٯ',
    'ن': 'ٮ',
    'ي': 'ٮ',
    'ك': 'ک',
    'ة': 'ه',
    'ئ': 'ٮ',
    'أ': 'ا',
    'إ': 'ا',
    'آ': 'ا',
    'ؤ': 'و',
}


def remove_dots_arabic_word(word: str) -> str:
    if not word:
        return ""
    if word:
        # Handle all but the final character
        core = word[:-1]
        last = word[-1]

        mapped = [DOTLESS_MAP.get(c, c) for c in core]

        # Special handling for final ن or ي
        if last == 'ن':
            mapped.append('ں')  # Final form of dotless noon
        elif last == 'ف':
            mapped.append('ڡ')  # Final form of dotless fa
        elif last in ('ي', 'ئ'):
            mapped.append('ى')  # Final form of ya (alif maqsura)
        else:
            mapped.append(DOTLESS_MAP.get(last, last))

    return ''.join(mapped)




def get_db_for_split(split_name):
    """
    Determine the camel_diac database based on the split name.
    """
    s = split_name.lower()
    if 'msa' in s or 'modern standard arabic' in s:
        return 'calima-msa-s31'
    elif 'egypt' in s or 'cairo' in s:
        return 'calima-egy-c044'
    else:

        return 'calima-egy-c044'

class SafeDiacritizer:
    def __init__(self):
        self.models = {}

    def get_model(self, db_name):
        if db_name not in self.models:
            bert_model_name = BERT_MODEL_BY_DB[db_name]
            print(
                "Loading camel_tools BERTUnfactoredDisambiguator "
                f"'{bert_model_name}' with {db_name} analyzer..."
            )
            bert = BERTUnfactoredDisambiguator.pretrained(bert_model_name)
            db_path = LOCAL_DB_PATH_BY_DB[db_name]
            if db_path.is_file():
                db = MorphologyDB(str(db_path))
            else:
                print(
                    f"Local morphology DB not found at {db_path}; "
                    f"using CAMeL Tools database {db_name}."
                )
                db = MorphologyDB.builtin_db(db_name, flags="a")
            bert._analyzer = Analyzer(db)
            self.models[db_name] = bert
        return self.models[db_name]

    def preload(self, db_names):
        for db_name in sorted(set(db_names)):
            self.get_model(db_name)

    def add_diacritics(self, text, db_name):
        if not text:
            return ""
        bert = self.get_model(db_name)
        words = simple_word_tokenize(text)
        disambigged = bert.disambiguate(words)
        diacritized_words = []
        for d in disambigged:
            if d.analyses and len(d.analyses) > 0:
                diacritized_words.append(d.analyses[0].analysis.get('diac', d.word))
            else:
                diacritized_words.append(d.word)
        return ' '.join(diacritized_words)

    def add_diacritics_batch(self, batch, db_name):
        return {"source": [self.add_diacritics(text, db_name) for text in batch["source"]]}


_PROCESS_DIACRITIZER: SafeDiacritizer | None = None


def _get_worker_diacritizer() -> SafeDiacritizer:
    """Reuse CAMeL BERT models across split tasks in each worker process."""
    global _PROCESS_DIACRITIZER
    if _PROCESS_DIACRITIZER is None:
        _PROCESS_DIACRITIZER = SafeDiacritizer()
    return _PROCESS_DIACRITIZER


def _build_noisy_splits_for_split(args: tuple[str, Dataset]) -> tuple[str, Dataset, Dataset]:
    split_name, split_ds = args
    db_name = get_db_for_split(split_name)
    diacritizer = _get_worker_diacritizer()

    size = min(len(split_ds), 500)
    split_ds_500 = split_ds.shuffle(seed=42).select(range(size))

    print(f"\nProcessing split: {split_name}")
    print(f"  -> Creating diacritics_{split_name} (using cached DB: {db_name}, size: {size})")
    diacritics_split = split_ds_500.map(
        lambda batch: diacritizer.add_diacritics_batch(batch, db_name),
        batched=True,
        batch_size=32,
        desc=f"Adding diacritics to {split_name}",
    )

    print(f"  -> Creating dotless_{split_name} (size: {size})")
    dotless_split = split_ds_500.map(
        lambda x: {"source": remove_dots(x["source"])},
        desc=f"Removing dots from {split_name}",
    )

    return split_name, diacritics_split, dotless_split


def build_noisy_dataset(
    ds: DatasetDict,
    diacritizer: SafeDiacritizer | None = None,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> DatasetDict:
    if num_workers < 1:
        raise ValueError("--num-workers must be at least 1")

    new_ds_dict = DatasetDict()
    split_names = list(ds.keys())

    if num_workers == 1:
        diacritizer = diacritizer or SafeDiacritizer()
        diacritizer.preload(get_db_for_split(split_name) for split_name in split_names)

        for split_name in split_names:
            split_name, diacritics_split, dotless_split = _build_noisy_splits_for_split_with_diacritizer(
                split_name,
                ds[split_name],
                diacritizer,
            )
            new_ds_dict[split_name] = ds[split_name]
            new_ds_dict[f"diacritics_{split_name}"] = diacritics_split
            new_ds_dict[f"dotless_{split_name}"] = dotless_split

        return new_ds_dict

    max_workers = min(num_workers, len(split_names))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_build_noisy_splits_for_split, [(name, ds[name]) for name in split_names]))

    by_split = {
        split_name: (diacritics_split, dotless_split)
        for split_name, diacritics_split, dotless_split in results
    }
    for split_name in split_names:
        diacritics_split, dotless_split = by_split[split_name]
        new_ds_dict[split_name] = ds[split_name]
        new_ds_dict[f"diacritics_{split_name}"] = diacritics_split
        new_ds_dict[f"dotless_{split_name}"] = dotless_split

    return new_ds_dict


def _build_noisy_splits_for_split_with_diacritizer(
    split_name: str,
    split_ds: Dataset,
    diacritizer: SafeDiacritizer,
) -> tuple[str, Dataset, Dataset]:
    db_name = get_db_for_split(split_name)
    size = min(len(split_ds), 500)
    split_ds_500 = split_ds.shuffle(seed=42).select(range(size))

    print(f"\nProcessing split: {split_name}")
    print(f"  -> Creating diacritics_{split_name} (using cached DB: {db_name}, size: {size})")
    diacritics_split = split_ds_500.map(
        lambda batch: diacritizer.add_diacritics_batch(batch, db_name),
        batched=True,
        batch_size=32,
        desc=f"Adding diacritics to {split_name}",
    )

    print(f"  -> Creating dotless_{split_name} (size: {size})")
    dotless_split = split_ds_500.map(
        lambda x: {"source": remove_dots(x["source"])},
        desc=f"Removing dots from {split_name}",
    )

    return split_name, diacritics_split, dotless_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add noisy variants to dialect dev/test datasets.")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "dev", "test"),
        default=list(DEFAULT_INPUT_SPLITS),
        help="Normalized dataset partitions to process (default: dev test).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Split-level worker processes for noise generation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    for split in args.splits:
        ds_path = dialect_output_path(f"dialects_{split}")
        output_path = dialect_output_path(f"dialects_{split}_noisy")

        print(f"Loading dataset from {ds_path}...")
        ds = load_from_disk(hf_disk_path(ds_path))
        noisy_ds = build_noisy_dataset(ds, num_workers=args.num_workers)

        print(f"\nSaving noisy {split} dataset to {output_path}...")
        noisy_ds.save_to_disk(hf_disk_path(output_path))

    print("Done!")

if __name__ == "__main__":
    main()
