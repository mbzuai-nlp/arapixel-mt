# Byte model

The Byte model provides a direct comparison to the Pixel model, using bytes instead of pixels. Arabic text is split on whitespace, and each word is represented using up to 32 UTF-8 bytes. These representations are then passed through Text ViT, which uses the same ViT layers and projection MLP as the Pixel model but takes text as input. The resulting representations are mean-pooled to produce one embedding per word, which is then passed to the SmolLM2 decoder.

## Prepare MSA data

~~~bash
python -m BytesSmolLM.prepare \
  --raw-dataset "$RAW_MSA" \
  --bytes-prepared-dataset "$BYTE_MSA"
~~~

Preparation uses the complete MSA set by default and creates
fixed 32-word, 32-byte-per-word inputs. A smaller percentage selects a nested
prefix for the curriculum experiment. Use `finetuning.prepare` for finetuning data.

## Train

The training commands launch their distributed workers themselves; do
not prefix them with `torchrun`.

~~~bash
python -m BytesSmolLM.train_alignment \
  --bytes-prepared-dataset "$BYTE_MSA" \
  --output-dir "$BYTE_EXPERIMENT/alignment" \
  --run-name bytes-msa \
  --num-gpus 4 \
  --train-batch-size 32 \
  --effective-batch-size 512

python -m BytesSmolLM.train_full \
  --bytes-prepared-dataset "$BYTE_MSA" \
  --alignment-checkpoint "$BYTE_EXPERIMENT/alignment/final/pytorch_model.bin" \
  --output-dir "$BYTE_EXPERIMENT" \
  --run-name bytes-msa \
  --num-gpus 4 \
  --train-batch-size 32 \
  --effective-batch-size 512
~~~

Alignment is one epoch. The full command creates `full_epoch1` and
`full_epoch2` as separate one-epoch runs.

To train several models sequentially, use the shared
[training runner](../training/README.md).

## Fine-tune and evaluate

Use the shared [fine-tuning](../finetuning/README.md) and
[evaluation](../evaluation/README.md) commands with `--model bytes` and
`--bytes-prepared-dataset`. Fine-tuning uses dynamic padding with a 64-word
training cap.

## Instantiate, load, and translate

~~~python
from pathlib import Path

import torch

from BytesSmolLM.utils import (
    FINETUNE_SOURCE_WORDS,
    MAX_WORD_LENGTH,
    PAD_TOKEN_ID,
    SimpleWordProcessor,
)
from evaluation.source_layout import configure_source_layout
from finetuning.common import load_bytes_model

checkpoint = Path("/path/to/pytorch_model.bin")
model, tokenizer = load_bytes_model(checkpoint)
_, fixed_source_length = configure_source_layout(
    "bytes", model, checkpoint, source_layout="auto"
)
model = model.to("cuda", dtype=torch.bfloat16).eval()

processor = SimpleWordProcessor(max_word_length=MAX_WORD_LENGTH)
source_limit = fixed_source_length or FINETUNE_SOURCE_WORDS
source = processor.process_text("مرحبا بالعالم")[:source_limit]
word_count = source.shape[0]

if fixed_source_length is None:
    input_ids = source.unsqueeze(0)
    attention_mask = torch.ones(1, word_count, dtype=torch.long)
else:
    input_ids = torch.full(
        (1, fixed_source_length, MAX_WORD_LENGTH),
        PAD_TOKEN_ID,
        dtype=torch.long,
    )
    input_ids[0, :word_count] = source
    attention_mask = torch.zeros(1, fixed_source_length, dtype=torch.long)
    attention_mask[0, :word_count] = 1

print(model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    tokenizer=tokenizer,
    max_new_tokens=128,
    num_beams=4,
)[0])
~~~
