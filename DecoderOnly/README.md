# Base decoder model

Base feeds SmolLM2 subword embeddings for the Arabic source directly into the
decoder that generates English. MSA training uses a fixed 256-token source
layout.

## Train

Base needs no alignment stage. The training command launches its
distributed workers itself; do not prefix it with `torchrun`.

~~~bash
python -m DecoderOnly.train_full \
  --raw-dataset "$RAW_MSA" \
  --output-dir "$BASE_EXPERIMENT" \
  --run-name base-msa \
  --num-gpus 4 \
  --train-batch-size 16 \
  --effective-batch-size 512
~~~

The command creates `full_epoch1` and `full_epoch2` as separate one-epoch
runs.

To train several models sequentially, use the shared
[training runner](../training/README.md).

## Fine-tune and evaluate

Use the shared [fine-tuning](../finetuning/README.md) and
[evaluation](../evaluation/README.md) commands with `--model decoder` and
`--decoder-prepared-dataset`. Fine-tuning uses dynamic padding with a 256-token
training cap.

## Instantiate, load, and translate

~~~python
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from DecoderOnly.model import DecoderOnlyTranslationModel
from DecoderOnly.utils import FINETUNE_SOURCE_TOKENS
from evaluation.source_layout import configure_source_layout

name = "HuggingFaceTB/SmolLM2-360M-Instruct"
checkpoint = Path("/path/to/pytorch_model.bin")
tokenizer = AutoTokenizer.from_pretrained(name)
tokenizer.pad_token = tokenizer.eos_token
backbone = AutoModelForCausalLM.from_pretrained(name)
model = DecoderOnlyTranslationModel(backbone, tokenizer)
model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
_, fixed_source_length = configure_source_layout(
    "decoder", model, checkpoint, source_layout="auto"
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
