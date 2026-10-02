"""Constants and helpers shared by the Pixel model."""

from enum import Enum

import torch

from PixelSmolLM.data import FixedPixelCollator
from PixelSmolLM.rendering.config import (
    PATCH_SIZE,
    PRETRAIN_SOURCE_WORDS,
)


MAX_AR_WORDS = PRETRAIN_SOURCE_WORDS
pixels_per_patch = PATCH_SIZE


class TrainingMode(Enum):
    ENCODER_PROJECTION = "encoder_projection"
    BOTH = "both"


def build_safe_words_ids(headers, tokenizer):
    """
    Returns a deconflicted bad_words_ids list:
    - keeps single-token bans,
    - drops multi-token sequences whose last token collides with any single-token ban,
    - ensures remaining multi-token sequences have unique final tokens.
    """
    seqs = []
    for h in headers:
        ids = tokenizer.encode(h, add_special_tokens=False)
        if len(ids) > 0:
            seqs.append(tuple(ids))

    single_last_ids = set(s[-1] for s in seqs if len(s) == 1)
    multis = [s for s in seqs if len(s) > 1 and s[-1] not in single_last_ids]

    seen_last = set()
    multis_unique = []
    for s in multis:
        last = s[-1]
        if last not in seen_last:
            seen_last.add(last)
            multis_unique.append(s)

    bad = [[i] for i in sorted(single_last_ids)]
    bad += [list(s) for s in multis_unique]
    return bad


def sanity_check(model, dataset, tokenizer, K: int = 2, max_new_tokens: int = 64):
    """
    Quick sanity check: trainable param summary, determinism, generation.
    """
    print("\n" + "=" * 60)
    print("SANITY CHECK")
    print("=" * 60)

    # ── 1. Trainable parameters ──
    print("\n[1] Parameter Summary")
    stats = {}
    for name, param in model.named_parameters():
        top = name.split('.')[0]
        if top not in stats:
            stats[top] = {"total": 0, "trainable": 0}
        stats[top]["total"] += param.numel()
        if param.requires_grad:
            stats[top]["trainable"] += param.numel()

    for module_name, counts in stats.items():
        pct = 100 * counts["trainable"] / counts["total"] if counts["total"] > 0 else 0
        status = "TRAIN" if counts["trainable"] > 0 else "FROZEN"
        print(f"  {module_name:30s}: {counts['trainable']:>12,} / {counts['total']:>12,} ({pct:5.1f}%) {status}")

    total = sum(s["total"] for s in stats.values())
    trainable = sum(s["trainable"] for s in stats.values())
    print(f"  {'TOTAL':30s}: {trainable:>12,} / {total:>12,} ({100*trainable/total:.1f}%)")

    # ── 2. Determinism + Forward ──
    print("\n[2] Determinism & Forward")
    collator = FixedPixelCollator()
    dl = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=collator,
    )
    it = iter(dl)

    model.train()
    batch = next(it)

    px1 = model._pixel_embed_batch(
        batch["word_images"], batch["patch_masks"], batch["word_counts"]
    )
    px2 = model._pixel_embed_batch(
        batch["word_images"], batch["patch_masks"], batch["word_counts"]
    )
    y1, y2 = px1[0], px2[0]
    diff = float((y1 - y2).abs().mean().item()) if y1.numel() and y2.numel() else 0.0
    print(f"  Encoder determinism diff: {diff:.4e}")

    # ── 3. Generation ──
    print("\n[3] Generation Samples")
    model.eval()
    dl = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=collator,
    )
    it = iter(dl)

    with torch.no_grad():
        for i in range(min(K, len(dataset))):
            batch = next(it)
            ar = batch["arabic_texts"][0]
            en = batch["english_texts"][0]

            hyps = model.generate(
                word_images=batch["word_images"],
                patch_masks=batch["patch_masks"],
                word_counts=batch["word_counts"],
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
            )

            print(f"\n  Sample {i}: encoder diff = {diff:.4e}")
            print(f"    AR:  {ar[:120]}")
            print(f"    REF: {en[:120]}")
            print(f"    HYP: {hyps[0][:200]}")

    print("\n" + "=" * 60)
    print("SANITY CHECK DONE")
    print("=" * 60 + "\n")
