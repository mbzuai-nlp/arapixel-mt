import torch 
import re
from enum import Enum
from torch.utils.data import Dataset
from dataclasses import dataclass


MAX_SEQ_LEN = 64        # Max sequence length for decoder
PRETRAIN_SOURCE_WORDS = 32
FINETUNE_SOURCE_WORDS = 64
MAX_AR_WORDS = PRETRAIN_SOURCE_WORDS
MAX_WORD_LENGTH = 32    # Max bytes per word

VOCAB_SIZE = 257        # 256 byte values + 1 PAD token
PAD_TOKEN_ID = 256      # The ID for padding

class TrainingMode(Enum):
    ENCODER_PROJECTION = "encoder_projection"
    BOTH = "both"


class SimpleWordProcessor:
    """
    Process words individually.
    Converts text -> words -> UTF-8 Bytes -> Fixed Length Vectors.
    """
    def __init__(self, max_word_length=MAX_WORD_LENGTH):
        self.max_word_length = max_word_length
        self.pad_token_id = PAD_TOKEN_ID

    def process_word(self, word: str):
        ids = list(word.encode('utf-8'))
        if len(ids) > self.max_word_length:
            ids = ids[:self.max_word_length]
        padding = [self.pad_token_id] * (self.max_word_length - len(ids))
        ids = ids + padding
        return ids

    def process_text(self, text: str):
        words = re.findall(r"\S+", str(text).strip())
        if not words:
            return torch.empty(0, self.max_word_length, dtype=torch.long)
        word_ids = [self.process_word(w) for w in words]
        return torch.tensor(word_ids, dtype=torch.long)

       
class TextViTTranslationDataset(Dataset):
    """Dataset that reads pre-tokenized byte IDs and target IDs."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Pre-tokenized Arabic bytes: already [MAX_AR_WORDS, MAX_WORD_LENGTH]
        word_byte_ids = torch.tensor(item["word_byte_ids"], dtype=torch.long)
        byte_word_counts = item["byte_word_counts"]

        # Pre-tokenized English target: already [MAX_SEQ_LEN]
        target_ids = torch.tensor(item["target_ids"], dtype=torch.long)
        target_mask = torch.tensor(item["target_mask"], dtype=torch.long)

        # Build word-level attention mask from byte_word_counts
        attention_mask = torch.zeros(MAX_AR_WORDS, dtype=torch.long)
        attention_mask[:byte_word_counts] = 1

        return {
            "input_ids": word_byte_ids,          # [MAX_AR_WORDS, MAX_WORD_LENGTH]
            "attention_mask": attention_mask,     # [MAX_AR_WORDS]
            "target_ids": target_ids,            # [MAX_SEQ_LEN]
            "target_mask": target_mask,          # [MAX_SEQ_LEN]
            "arabic_text": str(item["source"]) if item.get("source") else "",
            "english_text": str(item["target"]) if item.get("target") else ""
        }


@dataclass
class TextViTDataCollator:
    def __call__(self, features):
        return {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "target_ids": torch.stack([f["target_ids"] for f in features]),
            "target_mask": torch.stack([f["target_mask"] for f in features]),
            "arabic_texts": [f["arabic_text"] for f in features],
            "english_texts": [f["english_text"] for f in features]
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
    collator = TextViTDataCollator()
    dl = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=False, collate_fn=collator,
    )
    batch = next(iter(dl))

    dev = model._current_device()
    b_input_ids = batch['input_ids'].to(dev)
    b_attn_mask = batch['attention_mask'].to(dev)
    b_target_ids = batch['target_ids'].to(dev)
    b_target_mask = batch['target_mask'].to(dev)



    model.train()
    try:
        outputs = model(
            input_ids=b_input_ids, attention_mask=b_attn_mask,
            target_ids=b_target_ids, target_mask=b_target_mask,
        )
        print("MEMORY After FORWARD:")
        print(torch.cuda.memory_summary())
        print(f"  Loss: {outputs['loss'].item():.4f}")
        outputs['loss'].backward()
        print("  Backward: OK")
        model.zero_grad()
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
                input_ids=sample_batch['input_ids'].to(dev),
                attention_mask=sample_batch['attention_mask'].to(dev),
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
