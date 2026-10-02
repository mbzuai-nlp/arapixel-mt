"""Evaluate one checkpoint on every split of a raw evaluation DatasetDict."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import torch
from accelerate import Accelerator
from datasets import DatasetDict, load_from_disk

from evaluation.prepare import prepare_for_model
from evaluation.source_layout import SOURCE_LAYOUTS, configure_source_layout
from evaluation.validate import validate_outputs
from finetuning.common import (
    build_eval_dataloader,
    load_bytes_model,
    load_decoder_model,
    load_dynamic_pixel_model,
    load_vocab_model,
)
from training.cli import (
    add_model_argument,
    add_prepared_dataset_arguments,
    prepared_dataset_for,
)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
accelerator = Accelerator()


def parse_args() -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_argument(parser)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--raw-dataset",
        type=Path,
        required=True,
        help=(
            "Raw DatasetDict. Every split must contain source and target columns; "
            "every split is evaluated."
        ),
    )
    add_prepared_dataset_arguments(parser)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for results.csv, predictions.csv, and validation.json.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-proc", type=int, default=8)
    parser.add_argument(
        "--source-layout",
        choices=SOURCE_LAYOUTS,
        default="auto",
        help=(
            "auto reads training_complete.json beside the checkpoint; msa "
            "uses the model's fixed MSA training length; dynamic pads each "
            "batch to its longest source."
        ),
    )
    parser.add_argument(
        "--skip-comet",
        action="store_true",
        help="Compute BLEU/spBLEU/chrF++ without loading COMET.",
    )
    return parser.parse_args(), parser


def _prepared_dataset_exists(path: Path) -> bool:
    return path.is_dir() and (
        (path / "dataset_dict.json").is_file()
        or (
            (path / "state.json").is_file()
            and (path / "dataset_info.json").is_file()
        )
    )


def _completed(output_dir: Path) -> bool:
    validation = output_dir / "validation.json"
    if not validation.is_file():
        return False
    try:
        with validation.open(encoding="utf-8") as handle:
            return bool(json.load(handle).get("valid"))
    except (OSError, ValueError):
        return False


def load_comet_model():
    from comet import download_model, load_from_checkpoint

    return load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))


def generate_for_split(
    model,
    unwrapped,
    tokenizer,
    dataset,
    model_type,
    batch_size,
    num_beams,
    max_new_tokens,
    fixed_source_length,
):
    """Generate one translation per row and restore source order."""
    gathered = []
    if "_eval_index" in dataset.column_names:
        dataset = dataset.remove_columns("_eval_index")
    indexed_dataset = dataset.add_column(
        "_eval_index", list(range(len(dataset)))
    )
    dataloader = accelerator.prepare(
        build_eval_dataloader(
            model_type,
            indexed_dataset,
            tokenizer,
            batch_size,
            fixed_source_length=fixed_source_length,
        )
    )

    with torch.no_grad():
        for batch in dataloader:
            if not batch["arabic_texts"]:
                continue
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=torch.cuda.is_available(),
            ):
                if model_type == "bytes":
                    predictions = unwrapped.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        tokenizer=tokenizer,
                        max_new_tokens=max_new_tokens,
                        num_beams=num_beams,
                        use_bad_words_filtering=False,
                    )
                elif model_type in ("decoder", "vocab"):
                    predictions = unwrapped.generate(
                        source_ids=batch["source_ids"],
                        source_mask=batch["source_mask"],
                        tokenizer=tokenizer,
                        max_new_tokens=max_new_tokens,
                        num_beams=num_beams,
                        use_bad_words_filtering=False,
                    )
                else:
                    predictions = unwrapped.generate(
                        word_images=batch["word_images"],
                        patch_masks=batch["patch_masks"],
                        word_counts=batch["word_counts"],
                        tokenizer=tokenizer,
                        max_new_tokens=max_new_tokens,
                        num_beams=num_beams,
                        use_bad_words_filtering=False,
                    )

            rows = list(
                zip(
                    batch["sample_indices"],
                    [prediction.strip() for prediction in predictions],
                    batch["english_texts"],
                    batch["arabic_texts"],
                )
            )
            gathered_rows = accelerator.gather_for_metrics(
                rows,
                use_gather_object=True,
            )
            if accelerator.is_main_process:
                gathered.extend(gathered_rows)

    if not accelerator.is_main_process:
        return [], [], []
    if len(gathered) != len(dataset):
        raise RuntimeError(
            f"Expected {len(dataset)} predictions, got {len(gathered)}"
        )
    indices = [int(row[0]) for row in gathered]
    if sorted(indices) != list(range(len(dataset))):
        raise RuntimeError(
            "Distributed generation returned duplicated or missing source indices"
        )
    ordered = sorted(gathered, key=lambda row: int(row[0]))
    return (
        [row[1] for row in ordered],
        [row[2] for row in ordered],
        [row[3] for row in ordered],
    )


def evaluate_main() -> None:
    args, parser = parse_args()
    from evaluation.scoring import compute_bleu_spbleu_chrf
    if not args.model_checkpoint.is_file():
        raise FileNotFoundError(args.model_checkpoint)
    if not args.raw_dataset.is_dir():
        raise FileNotFoundError(args.raw_dataset)
    for value, name in (
        (args.batch_size, "batch-size"),
        (args.num_beams, "num-beams"),
        (args.max_new_tokens, "max-new-tokens"),
        (args.num_proc, "num-proc"),
    ):
        if value <= 0:
            parser.error(f"--{name} must be positive")

    supplied_prepared = prepared_dataset_for(args, parser, required=False)
    prepared_dataset = (
        supplied_prepared
        if supplied_prepared is not None
        else args.output_dir / f"{args.model}-prepared-dataset"
    )

    if _completed(args.output_dir):
        if accelerator.is_main_process:
            print(f"Validated evaluation already exists; skipping {args.output_dir}")
        return

    scores_path = args.output_dir / "results.csv"
    predictions_path = args.output_dir / "predictions.csv"
    validation_path = args.output_dir / "validation.json"

    if accelerator.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for path in (scores_path, predictions_path, validation_path):
            if path.exists():
                raise FileExistsError(
                    f"Partial evaluation output exists; inspect {path}"
                )
        if not _prepared_dataset_exists(prepared_dataset):
            if prepared_dataset.exists():
                raise FileExistsError(
                    f"Incomplete prepared evaluation data: {prepared_dataset}"
                )
            prepare_for_model(
                args.model,
                args.raw_dataset,
                prepared_dataset,
                num_proc=args.num_proc,
            )
    accelerator.wait_for_everyone()

    dataset = load_from_disk(str(prepared_dataset))
    if not isinstance(dataset, DatasetDict):
        raise ValueError(f"Expected an evaluation DatasetDict: {prepared_dataset}")
    if not dataset:
        raise ValueError("Evaluation DatasetDict contains no splits")
    for split, rows in dataset.items():
        missing = {"source", "target"} - set(rows.column_names)
        if missing:
            raise ValueError(
                f"Split {split!r} is missing columns: {sorted(missing)}"
            )

    loaders = {
        "pixel": load_dynamic_pixel_model,
        "bytes": load_bytes_model,
        "decoder": load_decoder_model,
        "vocab": load_vocab_model,
    }
    model, tokenizer = loaders[args.model](str(args.model_checkpoint))
    _, fixed_source_length = configure_source_layout(
        args.model,
        model,
        args.model_checkpoint,
        args.source_layout,
    )

    if accelerator.is_main_process:
        if fixed_source_length is None:
            print("Source layout: dynamic batch maximum")
        else:
            unit = "tokens" if args.model in ("decoder", "vocab") else "words"
            print(
                f"Source layout: fixed MSA layout at "
                f"{fixed_source_length} {unit}"
            )

    model = accelerator.prepare(model)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.eval()

    comet_model = None
    if accelerator.is_main_process and not args.skip_comet:
        print("Loading COMET model (Unbabel/wmt22-comet-da)...")
        comet_model = load_comet_model()
    elif accelerator.is_main_process:
        print("COMET scoring disabled.")

    split_results = {}
    if accelerator.is_main_process:
        print(f"\n=== Generation phase: {len(dataset)} splits ===")
    generation_start = time.time()
    for split in dataset:
        if accelerator.is_main_process:
            split_start = time.time()
            print(f"  Generating [{split}] ({len(dataset[split])} samples)...")
        predictions, references, sources = generate_for_split(
            model,
            unwrapped,
            tokenizer,
            dataset[split],
            args.model,
            args.batch_size,
            args.num_beams,
            args.max_new_tokens,
            fixed_source_length,
        )
        if accelerator.is_main_process:
            print(f"Split done in {time.time() - split_start:.2f}s")
            split_results[split] = (predictions, references, sources)

    if not accelerator.is_main_process:
        return

    print(f"Generation phase done in {time.time() - generation_start:.2f}s")
    print("\n=== Scoring phase ===")

    all_predictions, all_references, all_sources = [], [], []
    boundaries = {}
    for split, (predictions, references, sources) in split_results.items():
        start = len(all_predictions)
        all_predictions.extend(predictions)
        all_references.extend(references)
        all_sources.extend(sources)
        boundaries[split] = (start, len(all_predictions))

    comet_scores = None
    if comet_model is not None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Running COMET on {len(all_predictions)} samples...")
        comet_start = time.time()
        comet_data = [
            {"src": source, "mt": prediction, "ref": reference}
            for source, prediction, reference in zip(
                all_sources,
                all_predictions,
                all_references,
            )
        ]
        comet_output = comet_model.predict(
            comet_data,
            batch_size=256,
            gpus=1,
        )
        comet_scores = comet_output.scores
        print(f"COMET done in {time.time() - comet_start:.2f}s")

    with scores_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "checkpoint",
                "model_type",
                "eval_split",
                "bleu",
                "spbleu",
                "chrf++",
                "comet",
            ]
        )
        for split, (predictions, references, _) in split_results.items():
            start, end = boundaries[split]
            bleu, spbleu, chrf = compute_bleu_spbleu_chrf(
                predictions,
                references,
            )
            comet = ""
            if comet_scores is not None:
                comet = sum(comet_scores[start:end]) / (end - start)
            display = f"{comet:.4f}" if comet_scores is not None else "skipped"
            print(
                f"Results [{split}] — BLEU: {bleu:.2f}, "
                f"spBLEU: {spbleu:.2f}, chrF++: {chrf:.2f}, "
                f"COMET: {display}"
            )
            writer.writerow(
                [
                    args.model_checkpoint,
                    args.model,
                    split,
                    bleu,
                    spbleu,
                    chrf,
                    comet,
                ]
            )

    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "checkpoint",
                "model_type",
                "eval_split",
                "sample_index",
                "source",
                "reference",
                "prediction",
                "comet",
            ]
        )
        for split, (predictions, references, sources) in split_results.items():
            start, _ = boundaries[split]
            for index, (source, reference, prediction) in enumerate(
                zip(sources, references, predictions)
            ):
                comet = (
                    ""
                    if comet_scores is None
                    else comet_scores[start + index]
                )
                writer.writerow(
                    [
                        args.model_checkpoint,
                        args.model,
                        split,
                        index,
                        source,
                        reference,
                        prediction,
                        comet,
                    ]
                )

    validate_outputs(
        results=scores_path,
        predictions=predictions_path,
        prepared_dataset=prepared_dataset,
        model=args.model,
        checkpoint=args.model_checkpoint,
        output_json=validation_path,
    )
    print(f"Evaluation complete: {args.output_dir}")


if __name__ == "__main__":
    evaluate_main()
