# Contextual Pixel model

The Pixel model renders each source word as at most 16 patches, where each patch is a 24 x 24 image
of a character-bigram with one character overlapping.

The ViT and projection MLP produce one mean-pooled embedding per word for the
SmolLM2 decoder.

## Prepare MSA data

~~~bash
python -m PixelSmolLM.prepare \
  --raw-dataset "$RAW_MSA" \
  --pixel-prepared-dataset "$PIXEL_MSA"
~~~

The script defaults to the complete MSA set, 32 source words, and 64 target tokens. Percentage selects a
nested prefix for curriculum experiments. Use `finetuning.prepare` for finetuning data.

## Train

The training commands launch their distributed workers themselves; do
not prefix them with `torchrun`.

~~~bash
python -m PixelSmolLM.train_alignment \
  --pixel-prepared-dataset "$PIXEL_MSA" \
  --output-dir "$PIXEL_EXPERIMENT/alignment" \
  --run-name pixel-msa \
  --num-gpus 4 \
  --train-batch-size 64 \
  --effective-batch-size 512

python -m PixelSmolLM.train_full \
  --pixel-prepared-dataset "$PIXEL_MSA" \
  --alignment-checkpoint "$PIXEL_EXPERIMENT/alignment/final/pytorch_model.bin" \
  --output-dir "$PIXEL_EXPERIMENT" \
  --run-name pixel-msa \
  --num-gpus 4 \
  --train-batch-size 64 \
  --effective-batch-size 512
~~~

Alignment is one epoch. The full command creates `full_epoch1` and
`full_epoch2` as separate one-epoch runs.

To train several models sequentially, use the shared
[training runner](../training/README.md).

## Fine-tune and evaluate

Use the shared [fine-tuning](../finetuning/README.md) and
[evaluation](../evaluation/README.md) commands with `--model pixel` and
`--pixel-prepared-dataset`. Fine-tuning uses dynamic padding with a 64-word
training cap.

## Instantiate, load, and translate

~~~python
from pathlib import Path

import numpy as np
import torch

from evaluation.source_layout import configure_source_layout
from PixelSmolLM.factory import load_model
from PixelSmolLM.rendering import ContextualBigramRenderer, pack_word_patches
from PixelSmolLM.rendering.config import FINETUNE_SOURCE_WORDS

checkpoint = Path("/path/to/pytorch_model.bin")
model, tokenizer = load_model(checkpoint, dynamic=True)
_, fixed_source_length = configure_source_layout(
    "pixel", model, checkpoint, source_layout="auto"
)
model = model.to("cuda", dtype=torch.bfloat16).eval()

renderer = ContextualBigramRenderer()
_, _, words = renderer.render_text("مرحبا بالعالم")
source_limit = fixed_source_length or FINETUNE_SOURCE_WORDS
packed = [
    pack_word_patches(patches) for patches in words if patches
][:source_limit]
images = torch.stack([
    torch.from_numpy(np.stack((image, image, image), axis=0)).float() / 255.0
    for image, _ in packed
])
masks = torch.tensor([mask for _, mask in packed], dtype=torch.bool)

print(model.generate(
    word_images=images,
    patch_masks=masks,
    word_counts=[len(packed)],
    tokenizer=tokenizer,
    max_new_tokens=128,
    num_beams=4,
)[0])
~~~
