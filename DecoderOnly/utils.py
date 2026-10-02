import torch
from torch.utils.data import Dataset
from dataclasses import dataclass


MAX_SEQ_LEN = 64         # Max token length for English target
PRETRAIN_SOURCE_TOKENS = 256
FINETUNE_SOURCE_TOKENS = 256
MAX_AR_SEQ_LEN = PRETRAIN_SOURCE_TOKENS


class DecoderOnlyTranslationDataset(Dataset):
    """
    Dataset that tokenizes raw source/target text on-the-fly
    using the decoder tokenizer for both Arabic and English.
    """
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        arabic_text = str(item["source"]) if item.get("source") else ""
        english_text = str(item["target"]) if item.get("target") else ""

        # ── Tokenize Arabic source (no special tokens) ──
        ar_enc = self.tokenizer(
            arabic_text,
            truncation=True,
            max_length=MAX_AR_SEQ_LEN,
            padding="max_length",
            add_special_tokens=False,
        )
        source_ids = torch.tensor(ar_enc["input_ids"], dtype=torch.long)
        source_mask = torch.tensor(ar_enc["attention_mask"], dtype=torch.long)

        # ── Tokenize English target (append EOS, no special tokens) ──
        tgt_text = english_text + self.tokenizer.eos_token
        en_enc = self.tokenizer(
            tgt_text,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
            add_special_tokens=False,
        )
        target_ids = torch.tensor(en_enc["input_ids"], dtype=torch.long)
        target_mask = torch.tensor(en_enc["attention_mask"], dtype=torch.long)

        return {
            "source_ids": source_ids,          # [MAX_AR_SEQ_LEN]
            "source_mask": source_mask,        # [MAX_AR_SEQ_LEN]
            "target_ids": target_ids,          # [MAX_SEQ_LEN]
            "target_mask": target_mask,        # [MAX_SEQ_LEN]
            "arabic_text": arabic_text,
            "english_text": english_text,
        }


@dataclass
class DecoderOnlyDataCollator:
    def __call__(self, features):
        # Filter out empty samples
        valid = [f for f in features if f["english_text"].strip()]
        if not valid:
            valid = features  # fallback

        return {
            "source_ids": torch.stack([f["source_ids"] for f in valid]),
            "source_mask": torch.stack([f["source_mask"] for f in valid]),
            "target_ids": torch.stack([f["target_ids"] for f in valid]),
            "target_mask": torch.stack([f["target_mask"] for f in valid]),
            "arabic_texts": [f["arabic_text"] for f in valid],
            "english_texts": [f["english_text"] for f in valid],
        }


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
    Quick sanity check: trainable param summary, forward pass, generation.
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

    # ── 2. Forward + Backward ──
    print("\n[2] Forward & Backward")
    collator = DecoderOnlyDataCollator()
    dl = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=False, collate_fn=collator,
    )
    batch = next(iter(dl))

    dev = model._current_device()
    model.train()
    try:
        outputs = model(
            source_ids=batch["source_ids"].to(dev),
            source_mask=batch["source_mask"].to(dev),
            target_ids=batch["target_ids"].to(dev),
            target_mask=batch["target_mask"].to(dev),
        )
        if torch.cuda.is_available():
            print("MEMORY After FORWARD:")
            print(torch.cuda.memory_summary())
        print(f"  Loss: {outputs['loss'].item():.4f}")
        outputs['loss'].backward()
        print("  Backward: OK")
        model.zero_grad()
        if torch.cuda.is_available():
            print("MEMORY AFTER BACKWARD:")
            print(torch.cuda.memory_summary())
    except Exception as e:
        print(f"  Forward FAILED: {e}")
        raise

    # ── 3. Generation ──
    print("\n[3] Generation Samples")
    model.eval()
    with torch.no_grad():
        for i in range(min(K, len(dataset))):
            sample_batch = collator([dataset[i]])
            ar = sample_batch.get("arabic_texts", [""])[0]
            en = sample_batch.get("english_texts", [""])[0]

            hyps = model.generate(
                source_ids=sample_batch["source_ids"].to(dev),
                source_mask=sample_batch["source_mask"].to(dev),
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
            )

            print(f"\n  Sample {i}:")
            if ar:
                print(f"    AR:  {ar[:120]}")
            if en:
                print(f"    REF: {en[:120]}")
            print(f"    HYP: {hyps[0][:200]}")

    print("\n" + "=" * 60)
    print("SANITY CHECK DONE")
    print("=" * 60 + "\n")
