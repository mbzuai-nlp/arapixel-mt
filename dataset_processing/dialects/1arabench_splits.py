from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple

from datasets import Dataset, DatasetDict

from dataset_processing.dialects.step1_utils import (
    RAW_ROOT,
    dialect_output_path,
    format_source_text,
    hf_disk_path,
    save_dataset_dict,
)

ARABENCH_RAW_ROOT = RAW_ROOT / "arabench"
ARABENCH_DIR = ARABENCH_RAW_ROOT / "AraBench_dataset"
XML_PATH = ARABENCH_RAW_ROOT / "LDC.xml"

REGION_MAP = {
    "nil": "Nile Basin",
    "glf": "Gulf",
    "lev": "Levantine",
    "msa": "MSA",
    "mgr": "Maghreb",
}

COUNTRY_MAP = {
    "eg": "Egypt",
    "sd": "Sudan",
    "iq": "Iraq",
    "om": "Oman",
    "qa": "Qatar",
    "sa": "Saudi Arabia",
    "jo": "Jordan",
    "lb": "Lebanon",
    "ps": "Palestine",
    "pa": "Palestine",
    "sy": "Syria",
    "ms": "MSA",
    "dz": "Algeria",
    "ly": "Libya",
    "ma": "Morocco",
    "tn": "Tunisia",
    "ye": "Yemen",
    "lv": "Levant",
}

DIALECT_BY_COUNTRY = {
    "Algeria": "Maghreb",
    "Egypt": "Nile Basin",
    "Iraq": "Gulf",
    "Jordan": "Levantine",
    "Lebanon": "Levantine",
    "Levant": "Levantine",
    "Libya": "Maghreb",
    "Morocco": "Maghreb",
    "MSA": "MSA",
    "Oman": "Gulf",
    "Palestine": "Levantine",
    "Qatar": "Gulf",
    "Saudi Arabia": "Gulf",
    "Sudan": "Nile Basin",
    "Syria": "Levantine",
    "Tunisia": "Maghreb",
    "Yemen": "Yemen",
}

DATASET_NAME_MAP = {
    "QAraC": "qarac",
    "ldc_web_eg": "apt",
    "ldc_web_lv": "apt",
}

APT_CANONICAL_METADATA = {
    "ldc_web_eg": {"country": "Egypt", "country_code": "eg", "region_code": "nil"},
    "ldc_web_lv": {"country": "Levant", "country_code": "lv", "region_code": "lev"},
}

TRAIN_DIALECT_TO_SPLIT = {
    "EGYPTIAN": "eg",
    "LEVANTINE": "lev",
}


class FileMeta(NamedTuple):
    corpus: str
    split: str
    region_code: str
    idx: str
    country_code: str
    lang: str
    has_ids: bool


def _slugify(value: str) -> str:
    return value.lower().replace(" ", "_")


def build_split_name(dataset: str, country: str) -> str:
    return _slugify(f"{dataset}_{country}")


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def parse_filename(filename: str) -> FileMeta | None:
    parts = filename.split(".")
    if len(parts) < 6:
        return None

    has_ids = parts[-1] == "ids"
    if has_ids:
        if len(parts) < 7:
            return None
        lang = parts[-2]
        country = parts[-3]
        idx = parts[-4]
        region = parts[-5]
        split = parts[-6]
        corpus = ".".join(parts[:-6])
    else:
        lang = parts[-1]
        country = parts[-2]
        idx = parts[-3]
        region = parts[-4]
        split = parts[-5]
        corpus = ".".join(parts[:-5])

    if lang not in {"ar", "en"} or split not in {"dev", "test"}:
        return None

    if corpus in APT_CANONICAL_METADATA:
        canonical = APT_CANONICAL_METADATA[corpus]
        region = canonical["region_code"]
        country = canonical["country_code"]

    return FileMeta(
        corpus=corpus,
        split=split,
        region_code=region,
        idx=idx,
        country_code=country,
        lang=lang,
        has_ids=has_ids,
    )


def load_xml_map(xml_path: Path = XML_PATH) -> dict[str, tuple[str, str]]:
    root = ET.parse(xml_path).getroot()

    xml_map: dict[str, tuple[str, str]] = {}
    for segment in root.iter("SEGMENT"):
        source_guid = segment.findtext("SOURCE_GUID", default="").strip()
        source = segment.findtext("SOURCE", default="").strip()
        target = segment.findtext("TARGET", default="").strip()
        if source_guid:
            xml_map[source_guid] = (source, target)
    return xml_map


def iter_xml_segments(xml_path: Path = XML_PATH) -> list[dict[str, str]]:
    root = ET.parse(xml_path).getroot()

    segments: list[dict[str, str]] = []
    for segment in root.iter("SEGMENT"):
        segments.append(
            {
                "dialect": segment.findtext("DIALECT", default="").strip(),
                "source_guid": segment.findtext("SOURCE_GUID", default="").strip(),
                "source": segment.findtext("SOURCE", default="").strip(),
                "target": segment.findtext("TARGET", default="").strip(),
            }
        )
    return segments


def resolve_region_country_dialect(region_code: str, country_code: str) -> tuple[str, str, str]:
    region = REGION_MAP.get(region_code, region_code)
    country = COUNTRY_MAP.get(country_code, country_code)
    dialect = DIALECT_BY_COUNTRY.get(country, region)
    return region, country, dialect


def resolve_dataset_name(corpus: str, idx: str) -> str:
    dataset_name = DATASET_NAME_MAP.get(corpus, corpus)
    if dataset_name == "bible":
        return f"bible_{idx}"
    return dataset_name


def collect_grouped_eval_files(
    data_dir: Path = ARABENCH_DIR,
) -> dict[tuple[Any, ...], dict[str, list[Path]]]:
    grouped: dict[tuple[Any, ...], dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))

    for path in sorted(data_dir.iterdir()):
        if not path.is_file() or path.name.lower().startswith("readme") or path.name.startswith("."):
            continue

        meta = parse_filename(path.name)
        if meta is None:
            raise ValueError(f"Unexpected AraBench filename: {path.name}")

        key = (
            meta.corpus,
            meta.split,
            meta.region_code,
            meta.idx,
            meta.country_code,
            meta.has_ids,
        )
        grouped[key][meta.lang].append(path)

    return grouped


def load_group_pairs(
    ar_path: Path,
    en_path: Path,
    has_ids: bool,
    xml_map: dict[str, tuple[str, str]],
) -> list[tuple[str, str, str]]:
    pairs: list[tuple[str, str, str]] = []

    ar_lines = read_lines(ar_path)
    en_lines = read_lines(en_path)
    if len(ar_lines) != len(en_lines):
        kind = "ID" if has_ids else "plain"
        raise ValueError(f"{kind} length mismatch: {ar_path} has {len(ar_lines)}, {en_path} has {len(en_lines)}")

    for index, (left, right) in enumerate(zip(ar_lines, en_lines), start=1):
        left = left.strip()
        right = right.strip()

        if has_ids:
            if not left or not right:
                continue
            if left != right:
                raise ValueError(f"ID pair mismatch in {ar_path} and {en_path}: {left!r} != {right!r}")
            if left not in xml_map:
                raise ValueError(f"{left!r} from {ar_path} is missing from {XML_PATH}")

            source, target = xml_map[left]
            pairs.append((left, source, target))
            continue

        pairs.append((f"{ar_path.stem}_{index:06d}", left, right))

    return pairs


def process_eval_group(
    group_key: tuple[Any, ...],
    lang_paths: dict[str, list[Path]],
    xml_map: dict[str, tuple[str, str]],
) -> tuple[str, str, list[dict[str, str]]]:
    corpus, split, region_code, idx, country_code, has_ids = group_key
    ar_paths = sorted(lang_paths.get("ar", []))
    en_paths = sorted(lang_paths.get("en", []))
    if not ar_paths or not en_paths:
        raise ValueError(f"Missing Arabic/English pair for AraBench group: {group_key}")

    raw_pairs = load_group_pairs(ar_paths[0], en_paths[0], has_ids, xml_map)
    dataset_name = resolve_dataset_name(corpus, idx)
    if corpus in APT_CANONICAL_METADATA:
        country = APT_CANONICAL_METADATA[corpus]["country"]
    else:
        _region, country, _dialect = resolve_region_country_dialect(region_code, country_code)

    split_name = build_split_name(dataset_name, country)
    rows: list[dict[str, str]] = []
    for raw_id, source, target in raw_pairs:
        raw_source = source.strip()
        raw_target = target.strip()
        if not raw_source or not raw_target:
            continue

        rows.append(
            {
                "id": f"arabench_{split}_{split_name}_{raw_id}",
                "source": format_source_text(raw_source),
                "target": raw_target,
            }
        )

    return split, split_name, rows


def build_eval_artifacts() -> dict[str, DatasetDict]:
    grouped = collect_grouped_eval_files()
    xml_map = load_xml_map()

    rows_by_split_name: dict[str, dict[str, list[dict[str, str]]]] = {
        "dev": defaultdict(list),
        "test": defaultdict(list),
    }

    for group_key, lang_paths in grouped.items():
        split, split_name, rows = process_eval_group(
            group_key,
            lang_paths,
            xml_map,
        )
        rows_by_split_name[split][split_name].extend(rows)

    return {
        split: DatasetDict(
            {
                split_name: Dataset.from_list(rows)
                for split_name, rows in sorted(rows_by_split_name[split].items())
            }
        )
        for split in ("dev", "test")
    }


def collect_excluded_eval_ids(data_dir: Path = ARABENCH_DIR) -> set[str]:
    excluded_ids: set[str] = set()

    for path in sorted(data_dir.glob("*.ids")):
        for line in read_lines(path):
            line = line.strip()
            if line:
                excluded_ids.add(line)

    return excluded_ids


def build_train_dataset() -> DatasetDict:
    excluded_ids = collect_excluded_eval_ids()
    rows_by_split: dict[str, list[dict[str, str]]] = {"eg": [], "lev": []}

    for segment in iter_xml_segments():
        source_guid = segment["source_guid"]
        raw_source = segment["source"].strip()
        raw_target = segment["target"].strip()
        dialect = segment["dialect"].strip().upper()

        if not source_guid or not raw_source or not raw_target:
            continue
        if source_guid in excluded_ids:
            continue
        if dialect not in TRAIN_DIALECT_TO_SPLIT:
            raise ValueError(f"Unexpected LDC train dialect label: {segment['dialect']!r}")

        split_name = TRAIN_DIALECT_TO_SPLIT[dialect]
        rows_by_split[split_name].append(
            {
                "id": f"arabench_train_{split_name}_{source_guid}",
                "source": format_source_text(raw_source),
                "target": raw_target,
            }
        )

    return DatasetDict(
        {
            split_name: Dataset.from_list(rows)
            for split_name, rows in rows_by_split.items()
        }
    )


def build_all(save_to_disk: bool = True) -> dict[str, DatasetDict]:
    artifacts = build_eval_artifacts()
    artifacts["train"] = build_train_dataset()

    if save_to_disk:
        save_dataset_dict(dict(artifacts["dev"]), dialect_output_path("arabench_dev"))
        save_dataset_dict(dict(artifacts["test"]), dialect_output_path("arabench_test"))

        train_output_path = dialect_output_path("arabench_train")
        train_output_path.parent.mkdir(parents=True, exist_ok=True)
        artifacts["train"].save_to_disk(hf_disk_path(train_output_path))
        print(f"Saved {train_output_path}")
        print(artifacts["train"])

    return artifacts


if __name__ == "__main__":
    build_all(save_to_disk=True)
