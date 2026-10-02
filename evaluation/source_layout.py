"""Resolve and apply the source layout used by a trained checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.position_ids import use_physical_generation_positions


SOURCE_LAYOUTS = ("auto", "msa", "dynamic")


def msa_source_length(model_type: str) -> int:
    """Return the fixed source length used during MSA pretraining."""
    if model_type == "pixel":
        from PixelSmolLM.rendering.config import PRETRAIN_SOURCE_WORDS

        return PRETRAIN_SOURCE_WORDS
    if model_type == "bytes":
        from BytesSmolLM.utils import PRETRAIN_SOURCE_WORDS

        return PRETRAIN_SOURCE_WORDS
    if model_type in ("decoder", "vocab"):
        from DecoderOnly.utils import PRETRAIN_SOURCE_TOKENS

        return PRETRAIN_SOURCE_TOKENS
    raise ValueError(f"Unknown model type: {model_type!r}")


def infer_source_layout(checkpoint: Path | str) -> str:
    """Infer ``msa`` or ``dynamic`` from checkpoint metadata or filename."""
    checkpoint = Path(checkpoint)
    metadata = checkpoint.parent / "training_complete.json"
    if metadata.is_file():
        with metadata.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("stage") in ("alignment", "full"):
            return "msa"
        if (
            payload.get("training_phase") == "finetuning"
            or "prepared_dataset" in payload
            or "dataset_path" in payload
        ):
            return "dynamic"

    if checkpoint.name == "best_model.bin":
        print(
            "Warning: no usable checkpoint metadata; inferring dynamic source "
            "layout from best_model.bin.",
            flush=True,
        )
        return "dynamic"
    if checkpoint.name == "pytorch_model.bin":
        print(
            "Warning: no usable checkpoint metadata; inferring fixed MSA "
            "source layout from pytorch_model.bin.",
            flush=True,
        )
        return "msa"
    raise ValueError(
        "Cannot infer source layout from this checkpoint. Set source_layout "
        "to 'msa' or 'dynamic' explicitly."
    )


def resolve_source_layout(
    model_type: str,
    checkpoint: Path | str,
    source_layout: str = "auto",
) -> tuple[str, int | None]:
    """Resolve a layout and its optional fixed source length."""
    if source_layout not in SOURCE_LAYOUTS:
        raise ValueError(
            f"source_layout must be one of {SOURCE_LAYOUTS}, got "
            f"{source_layout!r}"
        )
    resolved = (
        infer_source_layout(checkpoint)
        if source_layout == "auto"
        else source_layout
    )
    fixed = msa_source_length(model_type) if resolved == "msa" else None
    return resolved, fixed


def configure_source_layout(
    model_type: str,
    model,
    checkpoint: Path | str,
    source_layout: str = "auto",
) -> tuple[str, int | None]:
    """Configure a loaded model for fixed-MSA or dynamic generation.

    The returned fixed length is ``None`` for dynamic checkpoints. Callers
    that tokenize or pad source inputs themselves should use this value when
    constructing the input batch.
    """
    resolved, fixed = resolve_source_layout(
        model_type,
        checkpoint,
        source_layout,
    )
    if model_type == "pixel":
        model.fixed_source_length = fixed
    if fixed is not None:
        use_physical_generation_positions(model)
    return resolved, fixed
