# Vocab+ decoder model

Vocab+ augments the SmolLM2 tokenizer with Arabic BPE entries. Its alignment
epoch freezes all original parameters and updates only the newly added
embedding/language-head rows. Two consecutive full-model epochs follow; the
second loads the first epoch's weights with a fresh optimizer and scheduler.

## Train the merged tokenizer

~~~bash
python -m DecoderOnly.VocabPlus.train_tokenizer \
  --raw-dataset "$RAW_MSA" \
  --output-dir "$VOCAB_TOKENIZER_DIR"
~~~

The output includes vocab_plus_meta.txt. Reuse this exact tokenizer for
training, fine-tuning, and evaluation.

## Train

The training commands launch their distributed workers themselves; do
not prefix them with `torchrun`.

~~~bash
python -m DecoderOnly.VocabPlus.train_alignment \
  --raw-dataset "$RAW_MSA" \
  --vocab-tokenizer "$VOCAB_TOKENIZER_DIR" \
  --output-dir "$VOCAB_EXPERIMENT/alignment" \
  --run-name vocab-msa \
  --num-gpus 4 \
  --train-batch-size 32 \
  --effective-batch-size 512

python -m DecoderOnly.VocabPlus.train_full \
  --raw-dataset "$RAW_MSA" \
  --vocab-tokenizer "$VOCAB_TOKENIZER_DIR" \
  --alignment-checkpoint "$VOCAB_EXPERIMENT/alignment/final/pytorch_model.bin" \
  --output-dir "$VOCAB_EXPERIMENT" \
  --run-name vocab-msa \
  --num-gpus 4 \
  --train-batch-size 16 \
  --effective-batch-size 512
~~~

Alignment is one epoch. The full command creates `full_epoch1` and
`full_epoch2` as separate one-epoch runs.

To train several models sequentially, use the shared
[training runner](../../training/README.md).

## Fine-tune and evaluate

Set `VOCAB_TOKENIZER_DIR` to the merged tokenizer, then use the shared
[fine-tuning](../../finetuning/README.md) and
[evaluation](../../evaluation/README.md) commands with `--model vocab` and
`--vocab-prepared-dataset`. Fine-tuning uses dynamic padding with a 256-token
training cap.

## Instantiate, load, and translate

~~~python
import os
from pathlib import Path

import torch

from DecoderOnly.utils import FINETUNE_SOURCE_TOKENS
from evaluation.source_layout import configure_source_layout
from finetuning.common import load_vocab_model

os.environ["VOCAB_TOKENIZER_DIR"] = "/path/to/merged_tokenizer_32000"
checkpoint = Path("/path/to/pytorch_model.bin")
model, tokenizer = load_vocab_model(checkpoint)
_, fixed_source_length = configure_source_layout(
    "vocab", model, checkpoint, source_layout="auto"
)
model = model.to("cuda", dtype=torch.bfloat16).eval()

source_limit = fixed_source_length or FINETUNE_SOURCE_TOKENS
encoded = tokenizer(
    ["مرحبا بالعالم"],
    return_tensors="pt",
    padding="max_length" if fixed_source_length is not None else True,
    truncation=True,
    max_length=source_limit,
    add_special_tokens=False,
)
print(model.generate(
    source_ids=encoded.input_ids,
    source_mask=encoded.attention_mask,
    tokenizer=tokenizer,
    max_new_tokens=128,
    num_beams=4,
)[0])
~~~
