from __future__ import annotations

from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk

from dataset_processing.dialects.step1_utils import (
    RAW_ROOT,
    dialect_output_path,
    format_source_text,
    hf_disk_path,
)


FLORES_RAW_ROOT = RAW_ROOT / "FLORES"

# Arabic variants to download/read (Arabizi excluded).
ARABIC_VARIANTS = [
    "aeb_Arab",           # Tunisian
    "apc_Arab_nort3139",  # North Levantine
    "apc_Arab_sout3123",  # South Levantine
    "arb_Arab",           # MSA
    "ars_Arab",           # Najdi / Gulf
    "ary_Arab",           # Moroccan
    "arz_Arab",           # Egyptian
    "acm_Arab",           # Iraqi
    "acq_Arab",           # Yemeni
]

ENGLISH_VARIANT = "eng_Latn"

VARIANT_TO_DIALECT = {
    "aeb_Arab": "Tunisi",
    "apc_Arab_nort3139": "NorthLevantine",
    "apc_Arab_sout3123": "SouthLevantine",
    "arb_Arab": "MSA",
    "ars_Arab": "Najdi",
    "ary_Arab": "Moroccan",
    "arz_Arab": "Egyptian",
    "acm_Arab": "Mesopotamian",
    "acq_Arab": "Taizzi-Adeni",
}

# FLORES splits: "dev" -> FLORES_dev, "devtest" -> FLORES_test.
SPLIT_MAP = {
    "dev": "FLORES_dev",
    "devtest": "FLORES_test",
}


def all_variants() -> list[str]:
    return [*ARABIC_VARIANTS, ENGLISH_VARIANT]


def download_flores_raw(raw_root: Path = FLORES_RAW_ROOT) -> None:
    from datasets import load_dataset

    raw_root.mkdir(parents=True, exist_ok=True)
    for variant in all_variants():
        output_path = raw_root / variant
        if output_path.exists():
            print(f"Skipping existing FLORES variant: {output_path}")
            continue

        print(f"Downloading FLORES+ {variant} to {output_path}...")
        dataset = load_dataset("openlanguagedata/flores_plus", variant)
        dataset.save_to_disk(hf_disk_path(output_path))


def load_local_flores(raw_root: Path = FLORES_RAW_ROOT) -> dict[str, DatasetDict]:
    missing = [variant for variant in all_variants() if not (raw_root / variant).exists()]
    if missing:
        formatted = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing local FLORES raw variant(s) under {raw_root}: {formatted}. "
            "Run the one-time FLORES raw download before processing."
        )

    return {
        variant: load_from_disk(hf_disk_path(raw_root / variant))
        for variant in all_variants()
    }


def build_all(
    save_to_disk: bool = True,
    raw_root: Path = FLORES_RAW_ROOT,
) -> dict[str, object]:
    raw = load_local_flores(raw_root)
    accumulated: dict[str, dict[str, list[dict[str, str]]]] = {
        flores_split: {}
        for flores_split in SPLIT_MAP
    }

    eng_data = raw[ENGLISH_VARIANT]
    for variant, dialect in VARIANT_TO_DIALECT.items():
        ar_data = raw[variant]

        for flores_split in SPLIT_MAP:
            ar_rows = ar_data[flores_split]
            eng_rows = eng_data[flores_split]
            eng_lookup = {row["id"]: row["text"] for row in eng_rows}

            rows = []
            for row in ar_rows:
                orig_id = row["id"]
                rows.append(
                    {
                        "id": f"{orig_id}_{variant}",
                        "source": format_source_text(row["text"]),
                        "target": eng_lookup[orig_id],
                    }
                )

            accumulated[flores_split].setdefault(dialect, []).extend(rows)

    artifacts: dict[str, object] = {}
    for flores_split, save_name in SPLIT_MAP.items():
        dataset_dict = DatasetDict(
            {
                dialect: Dataset.from_list(rows)
                for dialect, rows in accumulated[flores_split].items()
            }
        )
        if save_to_disk:
            save_path = dialect_output_path(save_name)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            dataset_dict.save_to_disk(hf_disk_path(save_path))
            print(f"\nSaved {save_name} to {save_path}")
            print(dataset_dict)
            artifacts[flores_split] = dataset_dict
        else:
            artifacts[flores_split] = dataset_dict

    return artifacts


def main() -> None:
    build_all(save_to_disk=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
