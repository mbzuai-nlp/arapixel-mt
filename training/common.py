"""Small shared helpers for restartable distributed training."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_state_dict(model: torch.nn.Module, weights: Path) -> None:
    if not weights.is_file():
        raise FileNotFoundError(weights)
    print(f"Loading incoming weights from {weights}", flush=True)
    state_dict = torch.load(str(weights), map_location="cpu")
    model.load_state_dict(state_dict)
    del state_dict


def find_last_valid_checkpoint(output_dir: Path) -> Optional[Path]:
    """Return the newest complete Trainer checkpoint, skipping partial writes."""

    candidates = []
    for path in output_dir.glob("checkpoint-*") if output_dir.is_dir() else ():
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        candidates.append((step, path))

    required = (
        "trainer_state.json",
        "training_args.bin",
        "optimizer.pt",
        "scheduler.pt",
    )
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rng_files = (
        ["rng_state.pth"]
        if world_size == 1
        else [f"rng_state_{rank}.pth" for rank in range(world_size)]
    )
    for _, path in sorted(candidates, reverse=True):
        has_model = (path / "pytorch_model.bin").is_file() or (
            path / "model.safetensors"
        ).is_file()
        if (
            has_model
            and all((path / filename).is_file() for filename in required)
            and all((path / filename).is_file() for filename in rng_files)
        ):
            return path
        print(f"Ignoring incomplete Trainer checkpoint: {path}", flush=True)
    return None


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)
