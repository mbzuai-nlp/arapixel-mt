"""Build/load the contextual 24-pixel-patch Pixel model."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pixel import PIXELConfig, PIXELForPreTraining
from PixelSmolLM.model import ConfigurablePixelTranslationModel
from PixelSmolLM.utils import TrainingMode

from .rendering.config import IMAGE_HEIGHT, IMAGE_WIDTH, PATCH_SIZE


BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"


class DynamicPixelTranslationModel(ConfigurablePixelTranslationModel):
    """Use batch-dynamic word padding during dialect fine-tuning/evaluation."""

    def _pad_pixel_embeddings(self, pixel_lists, decoder_hidden, device):
        if not pixel_lists:
            return (
                torch.empty(0, 0, decoder_hidden, device=device),
                torch.empty(0, 0, dtype=torch.long, device=device),
            )
        fixed_length = getattr(self, "fixed_source_length", None)
        maximum = (
            fixed_length
            if fixed_length is not None
            else max(pixel.shape[0] for pixel in pixel_lists)
        )
        padded, masks = [], []
        for pixel in pixel_lists:
            pixel = pixel[:maximum]
            length = pixel.shape[0]
            if length < maximum:
                pad = torch.zeros(
                    maximum - length,
                    decoder_hidden,
                    device=device,
                    dtype=pixel.dtype,
                )
                pixel = torch.cat((pixel, pad), dim=0)
            mask = torch.cat(
                (
                    torch.ones(length, device=device, dtype=torch.long),
                    torch.zeros(
                        maximum - length, device=device, dtype=torch.long
                    ),
                )
            )
            padded.append(pixel.unsqueeze(0))
            masks.append(mask.unsqueeze(0))
        return torch.cat(padded), torch.cat(masks)


def build_encoder():
    config = PIXELConfig(
        image_size=[IMAGE_HEIGHT, IMAGE_WIDTH],
        patch_size=PATCH_SIZE,
        hidden_size=1536,
        num_attention_heads=16,
        num_hidden_layers=4,
        intermediate_size=6144,
        hidden_act="gelu",
        dropout=0.0,
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
        layer_norm_eps=1e-5,
        mask_ratio=0,
    )
    return PIXELForPreTraining(config=config).vit


def build_model(mode: TrainingMode, dynamic: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    decoder = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    decoder.config.use_cache = False
    model_class = (
        DynamicPixelTranslationModel
        if dynamic
        else ConfigurablePixelTranslationModel
    )
    model = model_class(build_encoder(), decoder, tokenizer, mode)
    return model, tokenizer


def load_model(checkpoint: Path | str, dynamic: bool = True):
    model, tokenizer = build_model(TrainingMode.BOTH, dynamic=dynamic)
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    print(f"Loading contextual Pixel weights from {checkpoint}", flush=True)
    state = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(state)
    del state
    model.set_training_mode(TrainingMode.BOTH)
    return model, tokenizer
