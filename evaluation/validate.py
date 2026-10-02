"""Validate score and prediction files against their prepared DatasetDict."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from datasets import DatasetDict, load_from_disk

from training.cli import (
    add_model_argument,
    add_prepared_dataset_arguments,
    prepared_dataset_for,
)


def parse_args() -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    add_model_argument(parser)
    add_prepared_dataset_arguments(parser)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(), parser


def _same_checkpoint(value: str, expected: Path) -> bool:
    return Path(value).resolve() == expected.resolve()


def _write_json_atomic(path: Path, payload) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def validate_outputs(
    *,
    results: Path,
    predictions: Path,
    prepared_dataset: Path,
    model: str,
    checkpoint: Path,
    output_json: Path,
) -> dict:
    for path in (results, predictions):
        if not path.is_file():
            raise FileNotFoundError(path)

    dataset = load_from_disk(str(prepared_dataset))
    if not isinstance(dataset, DatasetDict):
        raise ValueError(f"Expected DatasetDict: {prepared_dataset}")
    expected_splits = list(dataset.keys())

    scores = pd.read_csv(results, keep_default_na=False)
    required_score_columns = {
        "checkpoint",
        "model_type",
        "eval_split",
        "bleu",
        "spbleu",
        "chrf++",
    }
    missing = required_score_columns - set(scores.columns)
    if missing:
        raise RuntimeError(f"Score CSV missing columns: {sorted(missing)}")
    if len(scores) != len(expected_splits):
        raise RuntimeError(
            f"Expected {len(expected_splits)} score rows, got {len(scores)}"
        )
    if scores["eval_split"].duplicated().any():
        raise RuntimeError("Score CSV contains duplicated eval_split rows")
    if set(scores["eval_split"]) != set(expected_splits):
        raise RuntimeError("Score CSV split names do not match the evaluation dataset")
    if set(scores["model_type"]) != {model}:
        raise RuntimeError("Score CSV contains an unexpected model_type")
    if not all(
        _same_checkpoint(value, checkpoint)
        for value in scores["checkpoint"].astype(str)
    ):
        raise RuntimeError("Score CSV contains an unexpected checkpoint")
    if scores[["bleu", "spbleu", "chrf++"]].isna().any().any():
        raise RuntimeError("Score CSV contains missing BLEU/chrF values")

    prediction_frame = pd.read_csv(predictions, keep_default_na=False)
    required_prediction_columns = {
        "checkpoint",
        "model_type",
        "eval_split",
        "sample_index",
        "source",
        "reference",
        "prediction",
    }
    missing = required_prediction_columns - set(prediction_frame.columns)
    if missing:
        raise RuntimeError(f"Prediction CSV missing columns: {sorted(missing)}")
    if set(prediction_frame["eval_split"]) != set(expected_splits):
        raise RuntimeError(
            "Prediction CSV split names do not match the evaluation dataset"
        )
    if set(prediction_frame["model_type"]) != {model}:
        raise RuntimeError("Prediction CSV contains an unexpected model_type")
    if not all(
        _same_checkpoint(value, checkpoint)
        for value in prediction_frame["checkpoint"].astype(str)
    ):
        raise RuntimeError("Prediction CSV contains an unexpected checkpoint")

    expected_total = 0
    counts = {}
    for split in expected_splits:
        expected_count = len(dataset[split])
        expected_total += expected_count
        split_rows = prediction_frame[prediction_frame["eval_split"] == split]
        if len(split_rows) != expected_count:
            raise RuntimeError(
                f"{split}: expected {expected_count} predictions, "
                f"got {len(split_rows)}"
            )
        sample_indices = split_rows["sample_index"].astype(int).tolist()
        if sample_indices != list(range(expected_count)):
            raise RuntimeError(
                f"{split}: sample_index is duplicated, missing, or out of order"
            )
        expected_sources = [str(value or "") for value in dataset[split]["source"]]
        expected_references = [
            str(value or "") for value in dataset[split]["target"]
        ]
        if split_rows["source"].astype(str).tolist() != expected_sources:
            raise RuntimeError(f"{split}: prediction sources are reordered or changed")
        if split_rows["reference"].astype(str).tolist() != expected_references:
            raise RuntimeError(
                f"{split}: prediction references are reordered or changed"
            )
        counts[split] = len(split_rows)

    if len(prediction_frame) != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} total predictions, "
            f"got {len(prediction_frame)}"
        )

    payload = {
        "valid": True,
        "model_type": model,
        "checkpoint": str(checkpoint.resolve()),
        "prepared_dataset": str(prepared_dataset.resolve()),
        "split_count": len(expected_splits),
        "score_rows": len(scores),
        "prediction_rows": len(prediction_frame),
        "predictions_per_split": counts,
    }
    _write_json_atomic(output_json, payload)
    print(
        f"Validated {len(scores)} score rows and {len(prediction_frame):,} "
        f"predictions across {len(expected_splits)} splits."
    )
    return payload


def main() -> None:
    args, parser = parse_args()
    prepared_dataset = prepared_dataset_for(args, parser)
    validate_outputs(
        results=args.results,
        predictions=args.predictions,
        prepared_dataset=prepared_dataset,
        model=args.model,
        checkpoint=args.model_checkpoint,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
