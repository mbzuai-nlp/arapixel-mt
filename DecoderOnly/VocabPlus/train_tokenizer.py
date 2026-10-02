"""
Train a BPE tokenizer on Arabic source data, then merge the new
vocabulary into the base SmolLM2 tokenizer:  V+ = V_base ∪ V_new.

Usage:
    python -m DecoderOnly.VocabPlus.train_tokenizer \
        --raw-dataset /path/to/raw_msa \
        --output-dir /path/to/merged_tokenizer_32000
"""

import argparse
from pathlib import Path

from datasets import load_from_disk
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
from transformers import AutoTokenizer


# ══════════════════════════════════════════════════════════════
# Tokenizer training
# ══════════════════════════════════════════════════════════════

def collect_arabic_texts(dataset):
    """Return a list of all non-empty Arabic source texts from the train split."""
    texts = []
    for example in dataset["train"]:
        text = example.get("source", "")
        if text and text.strip():
            texts.append(text)
    return texts


def estimate_vocab_size(texts, target_compression=3.0):
    """
    Heuristic: pick a vocab size so that the average Arabic sentence
    is approximately `target_compression` times shorter than character-level.
    Falls back to 32 000 if the heuristic gives an unreasonable value.
    """
    avg_chars = sum(len(t) for t in texts) / max(len(texts), 1)
    # rough: vocab_size ≈ unique_chars * (avg_word_len / target_compression)
    unique_chars = len(set(c for t in texts for c in t))
    est = int(unique_chars * (avg_chars / target_compression))
    est = max(4000, min(est, 64000))  # clamp to [4k, 64k]
    return est


def train_bpe(texts, vocab_size):
    """Train a Unicode BPE tokenizer on the given texts."""
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


# ══════════════════════════════════════════════════════════════
# Merge
# ══════════════════════════════════════════════════════════════

def merge_into_base(base_tokenizer, new_tokenizer, filter_ascii=True):
    """
    Add tokens from `new_tokenizer` that do not exist in `base_tokenizer`.

    Args:
        filter_ascii: if True, skip tokens that are purely ASCII
                      (these are already well-covered by the base).
    Returns:
        (base_tokenizer, num_added, added_tokens)
    """
    base_vocab = set(base_tokenizer.get_vocab().keys())
    new_vocab = set(new_tokenizer.get_vocab().keys())
    candidates = sorted(new_vocab - base_vocab)

    if filter_ascii:
        candidates = [t for t in candidates if any(ord(c) > 127 for c in t)]

    num_added = base_tokenizer.add_tokens(candidates)
    return base_tokenizer, num_added, candidates[:num_added]


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer on Arabic and merge into SmolLM2 tokenizer."
    )
    parser.add_argument(
        "--raw-dataset",
        type=Path,
        required=True,
        help="Raw MSA DatasetDict containing a train split with source text.",
    )
    parser.add_argument(
        "--base-model",
        default="HuggingFaceTB/SmolLM2-360M-Instruct",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="New tokenizer vocab size; use 0 for automatic sizing.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for the merged tokenizer.",
    )
    parser.add_argument(
        "--no-filter-ascii",
        action="store_true",
        help="Also add purely ASCII entries from the Arabic-trained BPE.",
    )
    args = parser.parse_args()

    # ── Collect Arabic texts ──
    print(f"Loading dataset from {args.raw_dataset} ...")
    dataset = load_from_disk(args.raw_dataset)
    arabic_texts = collect_arabic_texts(dataset)
    print(f"  Collected {len(arabic_texts):,} Arabic sentences")

    # ── Vocab size ──
    vocab_size = args.vocab_size
    if vocab_size <= 0:
        vocab_size = estimate_vocab_size(arabic_texts)
        print(f"  Auto-estimated vocab_size = {vocab_size:,}")
    else:
        print(f"  Using vocab_size = {vocab_size:,}")

    # ── Train new BPE ──
    print("Training BPE tokenizer on Arabic source text ...")
    new_tokenizer = train_bpe(arabic_texts, vocab_size)
    print(f"  New tokenizer vocab size: {new_tokenizer.get_vocab_size()}")

    # ── Load base tokenizer ──
    print(f"Loading base tokenizer from {args.base_model} ...")
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    original_vocab_size = len(base_tokenizer)
    print(f"  Base vocab size: {original_vocab_size}")

    # ── Merge ──
    merged, num_added, added_tokens = merge_into_base(
        base_tokenizer, new_tokenizer,
        filter_ascii=not args.no_filter_ascii,
    )
    print(f"  Added {num_added:,} new tokens  →  merged vocab size: {len(merged):,}")

    # ── Save ──
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.output_dir))

    meta_path = args.output_dir / "vocab_plus_meta.txt"
    with open(meta_path, "w") as f:
        f.write(f"original_vocab_size: {original_vocab_size}\n")
        f.write(f"new_tokens_added: {num_added}\n")
        f.write(f"merged_vocab_size: {len(merged)}\n")
        f.write(f"new_bpe_vocab_size: {vocab_size}\n")
        f.write(f"base_model: {args.base_model}\n")
        f.write(f"dataset: {args.raw_dataset}\n")

    print(f"Merged tokenizer saved to {args.output_dir}")

    # ── Compression stats ──
    print("\n── Compression comparison ──")
    base_tok = AutoTokenizer.from_pretrained(args.base_model)
    sample = arabic_texts[:5000]

    base_lens = [len(base_tok.encode(t, add_special_tokens=False)) for t in sample]
    merged_lens = [len(merged.encode(t, add_special_tokens=False)) for t in sample]

    avg_base = sum(base_lens) / len(base_lens)
    avg_merged = sum(merged_lens) / len(merged_lens)
    print(f"  Avg tokens/sentence (base):   {avg_base:.1f}")
    print(f"  Avg tokens/sentence (merged): {avg_merged:.1f}")
    print(f"  Compression ratio:            {avg_base / avg_merged:.2f}x")

    # ── Quick decode sanity check ──
    print("\n── Sanity check ──")
    test_texts = arabic_texts[:3]
    for t in test_texts:
        b_ids = base_tok.encode(t, add_special_tokens=False)
        m_ids = merged.encode(t, add_special_tokens=False)
        print(f"  Text:   {t[:80]}")
        print(f"  Base:   {len(b_ids)} tokens  →  {base_tok.decode(b_ids)[:80]}")
        print(f"  Merged: {len(m_ids)} tokens  →  {merged.decode(m_ids)[:80]}")
        print()


if __name__ == "__main__":
    main()
