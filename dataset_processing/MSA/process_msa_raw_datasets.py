from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
from dataclasses import asdict, dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]

from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import (
    normalize_alef_ar,
    normalize_alef_maksura_ar,
    normalize_teh_marbuta_ar,
)
import cld3
from datasets import load_dataset



DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))
RAW_ROOT = DATA_ROOT / "pixel_translation_MSA_raw"
DEFAULT_STAGE_DIR = REPO_ROOT / "dataset_processing" / "artifacts" / "pixel_translation_msa_cleaned_filtered_32"
DEFAULT_REPORT_PATH = DEFAULT_STAGE_DIR / "stats.json"
DEFAULT_WORKSPACE_DATASET_PATH = DEFAULT_STAGE_DIR / "hf_dataset"
WORD_RE = re.compile(r"\S+")
PROGRESS_EVERY = 100_000
DEFAULT_NUM_WORKERS = max(1, min(os.cpu_count() or 1, 24))
DEFAULT_BATCH_SIZE = 2_000

DATASET_FILES = {
    "opus": (
        RAW_ROOT / "opus" / "OpenSubtitles.ar-en.ar",
        RAW_ROOT / "opus" / "OpenSubtitles.ar-en.en",
    ),
    "qed": (
        RAW_ROOT / "qed" / "QED.ar-en.ar",
        RAW_ROOT / "qed" / "QED.ar-en.en",
    ),
    "ted": (
        RAW_ROOT / "ted" / "NeuLab-TedTalks.ar-en.ar",
        RAW_ROOT / "ted" / "NeuLab-TedTalks.ar-en.en",
    ),
    "unpc": (
        RAW_ROOT / "unpc" / "UNPC.ar-en.ar",
        RAW_ROOT / "unpc" / "UNPC.ar-en.en",
    )
}


@dataclass
class StageStats:
    rows: int = 0
    source_tokens: int = 0
    target_tokens: int = 0


@dataclass
class DatasetReport:
    raw: StageStats
    cleaned: StageStats
    filtered: StageStats
    dropped_during_cleaning: int
    dropped_by_length_filter: int
    source_path: str
    target_path: str
    shard_path: str


@dataclass
class BatchResult:
    raw: StageStats
    cleaned: StageStats
    filtered: StageStats
    dropped_during_cleaning: int
    dropped_by_length_filter: int
    serialized_rows: str


@dataclass(frozen=True)
class ProcessingConfig:
    tokenize_arabic: bool = True
    strip_tatweel: bool = False
    save_stage: str = "filtered"


def count_words(text: str) -> int:
    return len(WORD_RE.findall(str(text).strip()))


def filter_len(source: str, target: str) -> bool:
    return (count_words(source) <= 32) and (count_words(target) <= 32)


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


def normalize_arabic_text_no_tokenize(text: str) -> str:
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

    return text


def maybe_strip_tatweel(text: str, strip_tatweel: bool) -> str:
    if not strip_tatweel:
        return text
    return text.replace("ـ", "")


def check_text(text: str, lang: str) -> bool:
    try:
        out = cld3.get_language(text)
        if (
            out is None
            or len(text.split()) > 100
            or len(text.split()) == 0
            or out.language != lang
            or out.is_reliable is False
        ):
            return False
    except Exception:
        return False
    return True


def prepare_cleaned_pair(
    raw_source: str,
    raw_target: str,
    config: ProcessingConfig,
) -> tuple[str, str] | None:
    if not (check_text(raw_source, "ar") and check_text(raw_target, "en")):
        return None
    if config.tokenize_arabic:
        cleaned_source = normalize_arabic_text(raw_source)
    else:
        cleaned_source = normalize_arabic_text_no_tokenize(raw_source)
    cleaned_source = maybe_strip_tatweel(
        cleaned_source, config.strip_tatweel
    ).strip()
    return cleaned_source, raw_target.strip()


def add_stage_stats(stats: StageStats, source: str, target: str) -> None:
    stats.rows += 1
    stats.source_tokens += count_words(source)
    stats.target_tokens += count_words(target)


def merge_stage_stats(target: StageStats, source: StageStats) -> None:
    target.rows += source.rows
    target.source_tokens += source.source_tokens
    target.target_tokens += source.target_tokens


def iter_aligned_lines(source_path: Path, target_path: Path) -> Iterable[tuple[int, str, str]]:
    with source_path.open("r", encoding="utf-8") as src_handle, target_path.open(
        "r", encoding="utf-8"
    ) as tgt_handle:
        for idx, (source, target) in enumerate(
            zip_longest(src_handle, tgt_handle, fillvalue=None),
            start=1,
        ):
            if source is None or target is None:
                raise ValueError(
                    f"Line mismatch between {source_path} and {target_path} at line {idx}"
                )
            yield idx, source.rstrip("\n"), target.rstrip("\n")


def batched_aligned_lines(
    source_path: Path, target_path: Path, batch_size: int
) -> Iterable[list[tuple[str, str]]]:
    batch: list[tuple[str, str]] = []
    for _, raw_source, raw_target in iter_aligned_lines(source_path, target_path):
        batch.append((raw_source, raw_target))
        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def process_batch(task: tuple[str, list[tuple[str, str]], ProcessingConfig]) -> BatchResult:
    dataset_name, rows, config = task
    raw = StageStats()
    cleaned = StageStats()
    filtered = StageStats()
    dropped_during_cleaning = 0
    dropped_by_length_filter = 0
    serialized_rows: list[str] = []

    for raw_source, raw_target in rows:
        add_stage_stats(raw, raw_source, raw_target)

        prepared_pair = prepare_cleaned_pair(raw_source, raw_target, config)
        if prepared_pair is None:
            dropped_during_cleaning += 1
            continue
        cleaned_source, cleaned_target = prepared_pair

        add_stage_stats(cleaned, cleaned_source, cleaned_target)

        if config.save_stage == "cleaned":
            serialized_rows.append(
                json.dumps(
                    {
                        "source": cleaned_source,
                        "target": cleaned_target,
                        "dataset": dataset_name,
                    },
                    ensure_ascii=False,
                )
            )

        if not filter_len(cleaned_source, cleaned_target):
            dropped_by_length_filter += 1
            continue

        add_stage_stats(filtered, cleaned_source, cleaned_target)
        if config.save_stage == "filtered":
            serialized_rows.append(
                json.dumps(
                    {
                        "source": cleaned_source,
                        "target": cleaned_target,
                        "dataset": dataset_name,
                    },
                    ensure_ascii=False,
                )
            )

    return BatchResult(
        raw=raw,
        cleaned=cleaned,
        filtered=filtered,
        dropped_during_cleaning=dropped_during_cleaning,
        dropped_by_length_filter=dropped_by_length_filter,
        serialized_rows="\n".join(serialized_rows) + ("\n" if serialized_rows else ""),
    )


def process_dataset(
    name: str,
    source_path: Path,
    target_path: Path,
    stage_dir: Path,
    num_workers: int,
    batch_size: int,
    config: ProcessingConfig,
) -> DatasetReport:
    shard_path = stage_dir / f"{name}.jsonl"
    raw = StageStats()
    cleaned = StageStats()
    filtered = StageStats()
    dropped_during_cleaning = 0
    dropped_by_length_filter = 0
    processed_rows = 0
    next_progress = PROGRESS_EVERY

    with shard_path.open("w", encoding="utf-8") as shard_handle, mp.Pool(
        processes=num_workers,
        maxtasksperchild=250,
    ) as pool:
        tasks = (
            (name, batch, config)
            for batch in batched_aligned_lines(source_path, target_path, batch_size)
        )
        for result in pool.imap_unordered(process_batch, tasks, chunksize=1):
            merge_stage_stats(raw, result.raw)
            merge_stage_stats(cleaned, result.cleaned)
            merge_stage_stats(filtered, result.filtered)
            dropped_during_cleaning += result.dropped_during_cleaning
            dropped_by_length_filter += result.dropped_by_length_filter
            processed_rows += result.raw.rows

            if result.serialized_rows:
                shard_handle.write(result.serialized_rows)

            while processed_rows >= next_progress:
                print(f"[{name}] processed {processed_rows:,} raw rows", flush=True)
                next_progress += PROGRESS_EVERY

    return DatasetReport(
        raw=raw,
        cleaned=cleaned,
        filtered=filtered,
        dropped_during_cleaning=dropped_during_cleaning,
        dropped_by_length_filter=dropped_by_length_filter,
        source_path=str(source_path),
        target_path=str(target_path),
        shard_path=str(shard_path),
    )


def summarize_reports(reports: dict[str, DatasetReport]) -> DatasetReport:
    total = DatasetReport(
        raw=StageStats(),
        cleaned=StageStats(),
        filtered=StageStats(),
        dropped_during_cleaning=0,
        dropped_by_length_filter=0,
        source_path="",
        target_path="",
        shard_path="",
    )
    for report in reports.values():
        total.raw.rows += report.raw.rows
        total.raw.source_tokens += report.raw.source_tokens
        total.raw.target_tokens += report.raw.target_tokens

        total.cleaned.rows += report.cleaned.rows
        total.cleaned.source_tokens += report.cleaned.source_tokens
        total.cleaned.target_tokens += report.cleaned.target_tokens

        total.filtered.rows += report.filtered.rows
        total.filtered.source_tokens += report.filtered.source_tokens
        total.filtered.target_tokens += report.filtered.target_tokens

        total.dropped_during_cleaning += report.dropped_during_cleaning
        total.dropped_by_length_filter += report.dropped_by_length_filter
    return total


def save_report(report_path: Path, reports: dict[str, DatasetReport]) -> None:
    payload = {
        "datasets": {name: asdict(report) for name, report in reports.items()},
        "total": asdict(summarize_reports(reports)),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_hf_dataset(stage_dir: Path, output_path: Path, dataset_names: list[str]) -> int:
    shard_paths = [str(stage_dir / f"{name}.jsonl") for name in dataset_names]
    cache_dir = stage_dir / ".hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        "json",
        data_files={"train": shard_paths},
        split="train",
        cache_dir=str(cache_dir),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_path))
    return dataset.num_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean, filter, merge, and report stats for the raw MSA translation corpora."
    )
    parser.add_argument(
        "--stage-dir",
        type=Path,
        default=DEFAULT_STAGE_DIR,
        help="Directory used for intermediate JSONL shards and stats.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Where to save the stats JSON report.",
    )
    parser.add_argument(
        "--workspace-output-path",
        type=Path,
        default=DEFAULT_WORKSPACE_DATASET_PATH,
        help="Where to save the merged Hugging Face dataset inside the writable workspace.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Worker processes used for batched normalization and filtering.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Aligned sentence pairs per worker batch.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_FILES),
        default=list(DATASET_FILES),
        help="Which raw corpora to process.",
    )
    parser.add_argument(
        "--skip-arabic-tokenization",
        action="store_true",
        help="Keep Arabic punctuation attached instead of spacing it with simple_word_tokenize.",
    )
    parser.add_argument(
        "--strip-tatweel",
        action="store_true",
        help="Remove Arabic tatweel/kashida characters (ـ) after normalization.",
    )
    parser.add_argument(
        "--save-stage",
        choices=["cleaned", "filtered"],
        default="filtered",
        help="Which stage to materialize into JSONL shards and the merged HF dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.stage_dir.mkdir(parents=True, exist_ok=True)
    selected_datasets = args.datasets
    config = ProcessingConfig(
        tokenize_arabic=not args.skip_arabic_tokenization,
        strip_tatweel=args.strip_tatweel,
        save_stage=args.save_stage,
    )

    reports: dict[str, DatasetReport] = {}
    for name in selected_datasets:
        source_path, target_path = DATASET_FILES[name]
        print(f"Processing {name}...", flush=True)
        reports[name] = process_dataset(
            name,
            source_path,
            target_path,
            args.stage_dir,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            config=config,
        )
        print(
            f"[{name}] raw={reports[name].raw.rows:,} cleaned={reports[name].cleaned.rows:,} "
            f"filtered={reports[name].filtered.rows:,}",
            flush=True,
        )

    save_report(args.report_path, reports)
    merged_rows = build_hf_dataset(args.stage_dir, args.workspace_output_path, selected_datasets)
    total = summarize_reports(reports)

    print("Processing complete.", flush=True)
    print(f"Report: {args.report_path}", flush=True)
    print(f"Workspace dataset: {args.workspace_output_path}", flush=True)
    print(f"Merged rows: {merged_rows:,}", flush=True)
    print(
        "Totals: "
        f"raw={total.raw.rows:,}, cleaned={total.cleaned.rows:,}, filtered={total.filtered.rows:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
