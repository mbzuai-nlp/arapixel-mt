import re
import os
from pathlib import Path

from datasets import load_from_disk, Features, Sequence, Value, Image as HFImage
from transformers import AutoTokenizer

# ─── Constants (must match training) ───
MAX_SEQ_LEN = 64
MAX_AR_WORDS = 32
MAX_WORD_LENGTH = 32
PAD_TOKEN_ID = 256  # byte vocab: 0-255 real, 256 = pad

DECODER_CHECKPOINT = "HuggingFaceTB/SmolLM2-360M-Instruct"
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))


def process_word_bytes(word: str, max_len: int = MAX_WORD_LENGTH) -> list:
    """Encode a single word to UTF-8 bytes, truncate/pad to max_len."""
    ids = list(word.encode("utf-8"))
    ids = ids[:max_len]
    ids += [PAD_TOKEN_ID] * (max_len - len(ids))
    return ids


def tokenize_batch(examples, tokenizer):
    """
    Batched map function. For each example:
      - Split Arabic source into whitespace words → byte-encode each word
      - Tokenize English target with the decoder tokenizer (+ EOS)
    """
    all_word_byte_ids = []
    all_word_counts = []

    # ── Arabic byte tokenization ──
    for arabic_text in examples["source"]:
        arabic_text = str(arabic_text) if arabic_text else ""
        words = re.findall(r"\S+", arabic_text.strip())
        words = words[:MAX_AR_WORDS]

        if words:
            word_byte_ids = [process_word_bytes(w) for w in words]
        else:
            word_byte_ids = []

        # Pad to MAX_AR_WORDS
        while len(word_byte_ids) < MAX_AR_WORDS:
            word_byte_ids.append([PAD_TOKEN_ID] * MAX_WORD_LENGTH)

        all_word_byte_ids.append(word_byte_ids)
        all_word_counts.append(min(len(words), MAX_AR_WORDS))

    # ── English tokenization (batched for speed) ──
    english_texts = [
        (str(t) if t else "") + tokenizer.eos_token
        for t in examples["target"]
    ]
    target_encodings = tokenizer(
        english_texts,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
    )

    result = {
        "word_byte_ids": all_word_byte_ids,           # [MAX_AR_WORDS, MAX_WORD_LENGTH]
        "byte_word_counts": all_word_counts,           # int
        "target_ids": target_encodings["input_ids"],   # [MAX_SEQ_LEN]
        "target_mask": target_encodings["attention_mask"],  # [MAX_SEQ_LEN]
    }

    # ── Pass through existing columns ──
    for col in examples:
        if col not in result:
            result[col] = examples[col]

    return result


if __name__ == "__main__":
    dataset = load_from_disk(
        DATA_ROOT / "pixel_translation_MSA_filtered_rendered"
    )

    tokenizer = AutoTokenizer.from_pretrained(DECODER_CHECKPOINT)
    tokenizer.pad_token = tokenizer.eos_token

    # Define new features (keep existing rendered features + add tokenized ones)
    new_features = Features({
        # Existing rendered columns
        "word_images": Sequence(HFImage()),
        "patch_masks": Sequence(Sequence(Value("int8"))),
        "num_words": Value("int32"),
        "source": Value("string"),
        "target": Value("string"),
        # New tokenized columns
        "word_byte_ids": Sequence(Sequence(Value("int16"))),  # int16 to hold 256
        "byte_word_counts": Value("int32"),
        "target_ids": Sequence(Value("int32")),
        "target_mask": Sequence(Value("int8")),
    })

    tokenized = dataset.map(
        lambda examples: tokenize_batch(examples, tokenizer),
        batched=True,
        batch_size=7000,
        num_proc=64,
        features=new_features,
    )

    tokenized.save_to_disk(
        DATA_ROOT / "pixel_translation_MSA_filtered_rendered_tokenized",
        num_proc=64,
    )

    # Quick verification
    print("Columns:", tokenized["train"].column_names)
    sample = tokenized["train"][0]
    print(f"word_byte_ids shape: {len(sample['word_byte_ids'])}x{len(sample['word_byte_ids'][0])}")
    print(f"byte_word_counts: {sample['byte_word_counts']}")
    print(f"target_ids length: {len(sample['target_ids'])}")
    print(f"num_words (pixel): {sample['num_words']}")
    print(f"source: {sample['source'][:80]}")
    print(f"target: {sample['target'][:80]}")
