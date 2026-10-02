"""
finetuning/common.py

Central utilities for all finetuning and evaluation scripts.
Provides:
  - Model loaders (bytes, decoder, vocab, contextual Pixel)
  - Dynamic train collators for Byte and tokenizer models
  - Dynamic eval collators for Byte and tokenizer models
  - Dataset helpers (validate_processed_dataset, build_training_data, build_dataloader)
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

from BytesSmolLM.utils import FINETUNE_SOURCE_WORDS as BYTE_FINETUNE_SOURCE_WORDS
from DecoderOnly.utils import FINETUNE_SOURCE_TOKENS
from PixelSmolLM.rendering.config import (
    FINETUNE_SOURCE_WORDS as PIXEL_FINETUNE_SOURCE_WORDS,
)

# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def validate_processed_dataset(dataset, required_columns, model_type):
    missing = [c for c in required_columns if c not in dataset.column_names]
    if missing:
        raise ValueError(
            f"{model_type} expects a processed dataset with columns: "
            f"{', '.join(required_columns)}. Missing: {', '.join(missing)}. "
            "Run the appropriate prepare_data script first."
        )


def pad_1d_sequences(sequences, pad_value):
    tensors = [torch.tensor(seq, dtype=torch.long) for seq in sequences]
    return pad_sequence(tensors, batch_first=True, padding_value=pad_value)


# ──────────────────────────────────────────────────────────────────────────────
# Model loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_bytes_model(checkpoint_path):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    from BytesSmolLM.text_vit import TextViTConfig, TextViTArchitecture
    from BytesSmolLM.model import ConfigurableTextViTTranslationModel
    from BytesSmolLM.utils import TrainingMode

    decoder_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
    decoder_tokenizer.pad_token = decoder_tokenizer.eos_token
    decoder = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
    decoder.config.use_cache = False

    text_vit_config = TextViTConfig()
    text_vit_model  = TextViTArchitecture(config=text_vit_config)

    model = ConfigurableTextViTTranslationModel(
        text_vit_model=text_vit_model,
        decoder_model=decoder,
        tokenizer=decoder_tokenizer,
        training_mode=TrainingMode.BOTH,
    )

    print(f"Loading weights from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.set_training_mode(TrainingMode.BOTH)
    return model, decoder_tokenizer


def load_decoder_model(checkpoint_path):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    from DecoderOnly.model import DecoderOnlyTranslationModel

    decoder_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
    decoder_tokenizer.pad_token = decoder_tokenizer.eos_token
    decoder = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
    decoder.config.use_cache = False

    model = DecoderOnlyTranslationModel(decoder_model=decoder, tokenizer=decoder_tokenizer)

    print(f"Loading weights from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    for p in model.parameters():
        p.requires_grad = True
    return model, decoder_tokenizer


def load_vocab_model(checkpoint_path):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    from DecoderOnly.VocabPlus.model import VocabPlusTranslationModel

    default_tokenizer_dir = (
        REPO_ROOT / "DecoderOnly" / "VocabPlus" / "merged_tokenizer_32000"
    )
    MERGED_TOKENIZER_DIR = Path(os.environ.get("VOCAB_TOKENIZER_DIR", default_tokenizer_dir))
    decoded_tokenizer = AutoTokenizer.from_pretrained(MERGED_TOKENIZER_DIR)
    decoded_tokenizer.pad_token = decoded_tokenizer.eos_token

    meta_path = os.path.join(MERGED_TOKENIZER_DIR, "vocab_plus_meta.txt")
    with open(meta_path) as f:
        for line in f:
            if line.startswith("original_vocab_size:"):
                original_vocab_size = int(line.split(":")[1].strip())
                break
        else:
            raise RuntimeError(f"Could not find original_vocab_size in {meta_path}")

    decoder = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
    decoder.config.use_cache = False
    decoder.resize_token_embeddings(len(decoded_tokenizer))

    model = VocabPlusTranslationModel(
        decoder_model=decoder,
        tokenizer=decoded_tokenizer,
        original_vocab_size=original_vocab_size,
    )

    print(f"Loading weights from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.unfreeze_all()
    return model, decoded_tokenizer


def load_dynamic_pixel_model(checkpoint_path):
    from PixelSmolLM.factory import load_model

    return load_model(checkpoint_path, dynamic=True)


load_pixel_model = load_dynamic_pixel_model


# ──────────────────────────────────────────────────────────────────────────────
# Train collators
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class DynamicBytesTrainCollator:
    pad_token_id: int
    max_source_words: int = BYTE_FINETUNE_SOURCE_WORDS

    def __call__(self, features):
        from BytesSmolLM.utils import MAX_WORD_LENGTH, PAD_TOKEN_ID

        valid = [f for f in features if int(f.get("num_words", 0)) > 0 and str(f.get("target", "")).strip()]
        if not valid:
            return {
                "input_ids":     torch.empty(0, 0, MAX_WORD_LENGTH, dtype=torch.long),
                "attention_mask": torch.empty(0, 0, dtype=torch.long),
                "target_ids":    torch.empty(0, 0, dtype=torch.long),
                "target_mask":   torch.empty(0, 0, dtype=torch.long),
                "arabic_texts":  [],
                "english_texts": [],
            }

        batch_size = len(valid)
        max_words  = min(max(int(f["num_words"]) for f in valid), self.max_source_words)

        input_ids      = torch.full((batch_size, max_words, MAX_WORD_LENGTH), PAD_TOKEN_ID, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_words), dtype=torch.long)

        for i, f in enumerate(valid):
            n = min(len(f["input_ids"]), len(f["attention_mask"]), max_words)
            if n > 0:
                input_ids[i, :n]      = torch.tensor(f["input_ids"][:n],      dtype=torch.long)
                attention_mask[i, :n] = torch.tensor(f["attention_mask"][:n], dtype=torch.long)

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "target_ids":     pad_1d_sequences([f["target_ids"]   for f in valid], pad_value=self.pad_token_id),
            "target_mask":    pad_1d_sequences([f["target_mask"]  for f in valid], pad_value=0),
            "arabic_texts":   [str(f.get("source", "")) for f in valid],
            "english_texts":  [str(f.get("target", "")) for f in valid],
        }


@dataclass
class DynamicTokenTrainCollator:
    pad_token_id: int
    max_source_tokens: int = FINETUNE_SOURCE_TOKENS

    def __call__(self, features):
        valid = [f for f in features if len(f.get("source_ids", [])) > 0 and str(f.get("target", "")).strip()]
        if not valid:
            return {
                "source_ids":    torch.empty(0, 0, dtype=torch.long),
                "source_mask":   torch.empty(0, 0, dtype=torch.long),
                "target_ids":    torch.empty(0, 0, dtype=torch.long),
                "target_mask":   torch.empty(0, 0, dtype=torch.long),
                "arabic_texts":  [],
                "english_texts": [],
            }

        return {
            "source_ids":    pad_1d_sequences([f["source_ids"]  [:self.max_source_tokens] for f in valid], pad_value=self.pad_token_id),
            "source_mask":   pad_1d_sequences([f["source_mask"] [:self.max_source_tokens] for f in valid], pad_value=0),
            "target_ids":    pad_1d_sequences([f["target_ids"]  for f in valid], pad_value=self.pad_token_id),
            "target_mask":   pad_1d_sequences([f["target_mask"] for f in valid], pad_value=0),
            "arabic_texts":  [str(f.get("source", "")) for f in valid],
            "english_texts": [str(f.get("target", "")) for f in valid],
        }


# ──────────────────────────────────────────────────────────────────────────────
# Eval collators
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class DynamicBytesEvalCollator:
    fixed_source_length: Optional[int] = None

    def __call__(self, features):
        from BytesSmolLM.utils import MAX_WORD_LENGTH, PAD_TOKEN_ID

        valid = [f for f in features if int(f.get("num_words", 0)) > 0 and str(f.get("target", "")).strip()]
        if not valid:
            return {
                "input_ids":     torch.empty(0, 0, MAX_WORD_LENGTH, dtype=torch.long),
                "attention_mask": torch.empty(0, 0, dtype=torch.long),
                "sample_indices": [],
                "arabic_texts":  [],
                "english_texts": [],
            }

        batch_size = len(valid)
        max_words = (
            self.fixed_source_length
            if self.fixed_source_length is not None
            else max(int(f["num_words"]) for f in valid)
        )

        input_ids      = torch.full((batch_size, max_words, MAX_WORD_LENGTH), PAD_TOKEN_ID, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_words), dtype=torch.long)

        for i, f in enumerate(valid):
            n = min(len(f["input_ids"]), len(f["attention_mask"]), max_words)
            if n > 0:
                input_ids[i, :n]      = torch.tensor(f["input_ids"][:n],      dtype=torch.long)
                attention_mask[i, :n] = torch.tensor(f["attention_mask"][:n], dtype=torch.long)

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "sample_indices": [int(f["_eval_index"]) for f in valid],
            "arabic_texts":   [str(f.get("source", "")) for f in valid],
            "english_texts":  [str(f.get("target", "")) for f in valid],
        }


@dataclass
class DynamicTokenEvalCollator:
    pad_token_id: int
    fixed_source_length: Optional[int] = None

    def __call__(self, features):
        valid = [f for f in features if len(f.get("source_ids", [])) > 0 and str(f.get("target", "")).strip()]
        if not valid:
            return {
                "source_ids":    torch.empty(0, 0, dtype=torch.long),
                "source_mask":   torch.empty(0, 0, dtype=torch.long),
                "sample_indices": [],
                "arabic_texts":  [],
                "english_texts": [],
            }

        max_len = (
            self.fixed_source_length
            if self.fixed_source_length is not None
            else max(len(f["source_ids"]) for f in valid)
        )
        batch_size = len(valid)

        source_ids   = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        source_mask  = torch.zeros((batch_size, max_len), dtype=torch.long)

        for i, f in enumerate(valid):
            n = min(len(f["source_ids"]), len(f["source_mask"]), max_len)
            source_ids[i,  :n] = torch.tensor(f["source_ids"][:n],  dtype=torch.long)
            source_mask[i, :n] = torch.tensor(f["source_mask"][:n], dtype=torch.long)

        return {
            "source_ids":    source_ids,
            "source_mask":   source_mask,
            "sample_indices": [int(f["_eval_index"]) for f in valid],
            "arabic_texts":  [str(f.get("source", "")) for f in valid],
            "english_texts": [str(f.get("target", "")) for f in valid],
        }


# ──────────────────────────────────────────────────────────────────────────────
# build_training_data / build_dataloader — one place, used by both scripts
# ──────────────────────────────────────────────────────────────────────────────

def build_training_data(model_type, dataset, tokenizer):
    """Returns (train_dataset, collator) ready for the HF Trainer."""
    if model_type == "pixel":
        from PixelSmolLM.data import (
            DynamicPixelTrainCollator,
            PixelTranslationDataset,
        )

        validate_processed_dataset(
            dataset,
            ["word_images", "patch_masks", "num_words", "target_ids", "target_mask", "source", "target"],
            model_type,
        )
        return PixelTranslationDataset(
            dataset, include_targets=True
        ), DynamicPixelTrainCollator(
            pad_token_id=tokenizer.pad_token_id,
            max_source_words=PIXEL_FINETUNE_SOURCE_WORDS,
        )

    if model_type == "bytes":
        validate_processed_dataset(
            dataset,
            ["input_ids", "attention_mask", "num_words", "target_ids", "target_mask", "source", "target"],
            model_type,
        )
        return dataset, DynamicBytesTrainCollator(
            pad_token_id=tokenizer.pad_token_id,
            max_source_words=BYTE_FINETUNE_SOURCE_WORDS,
        )

    # decoder / vocab
    validate_processed_dataset(
        dataset,
        ["source_ids", "source_mask", "target_ids", "target_mask", "source", "target"],
        model_type,
    )
    return dataset, DynamicTokenTrainCollator(
        pad_token_id=tokenizer.pad_token_id,
        max_source_tokens=FINETUNE_SOURCE_TOKENS,
    )


def build_eval_dataloader(
    model_type,
    dataset,
    tokenizer,
    batch_size,
    fixed_source_length=None,
):
    """Return an evaluation DataLoader with dynamic or explicitly fixed padding."""
    if fixed_source_length is not None and fixed_source_length <= 0:
        raise ValueError("fixed_source_length must be a positive integer")

    if model_type == "pixel":
        from PixelSmolLM.data import build_eval_dataloader as build_pixel_eval_dataloader

        validate_processed_dataset(
            dataset,
            ["word_images", "patch_masks", "num_words", "source", "target"],
            model_type,
        )
        return build_pixel_eval_dataloader(
            model_type,
            dataset,
            tokenizer,
            batch_size,
            fixed_source_length=fixed_source_length,
        )

    if model_type == "bytes":
        validate_processed_dataset(
            dataset,
            ["input_ids", "attention_mask", "num_words", "source", "target"],
            model_type,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=DynamicBytesEvalCollator(
                fixed_source_length=fixed_source_length,
            ),
            num_workers=4,
            pin_memory=True,
        )

    # decoder / vocab
    validate_processed_dataset(
        dataset,
        ["source_ids", "source_mask", "source", "target"],
        model_type,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DynamicTokenEvalCollator(
            pad_token_id=tokenizer.pad_token_id,
            fixed_source_length=fixed_source_length,
        ),
        num_workers=4,
        pin_memory=True,
    )
