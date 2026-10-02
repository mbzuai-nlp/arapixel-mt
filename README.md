# Pixel Translation

Arabic-to-English translation experiments comparing four SmolLM2-360M input
representations: Base subword tokens, Vocab+ Arabic-augmented subwords, bytes,
and pixels.


## Repository layout

- PixelSmolLM, BytesSmolLM, and DecoderOnly: model code, data preparation, training,
  and model-specific usage.
- training: Wrapper for model MSA initial pretraining.
- finetuning: dialect data preparation and fine-tuning.
- evaluation: distributed generation, scoring, and output validation.
- msa_curriculum: nested 5/10/25/50% MSA experiment wrappers.
- dataset_processing: normalized MSA/dialect construction and noisy-test
  generation.
- PixelSmolLM/pixel: the
  [PIXEL fork](https://github.com/Abdulrhman-shaheen/pixel-translation)
  of [xplip/pixel](https://github.com/xplip/pixel) used by the pixel translation
  model, included as a Git submodule.

Each model directory contains its own README.

## Installation

Clone the repository together with the PIXEL submodule:

~~~bash
git clone --recurse-submodules https://github.com/mbzuai-nlp/arapixel-mt.git
cd arapixel-mt
~~~

For an existing clone, initialize the submodule with:

~~~bash
git submodule update --init --recursive
~~~

Create a Python 3.9 environment and install PyTorch with CUDA 12.1, the
Cairo/Pango system libraries used by the renderer, and all project
dependencies:

~~~bash
conda create -n arapixel-mt python=3.9 -y
conda activate arapixel-mt

conda install -c pytorch -c nvidia \
  pytorch=2.5.1 torchvision=0.20.1 pytorch-cuda=12.1 -y

conda install -c conda-forge \
  cairo pango pycairo pygobject manimpango -y

python -m pip install --upgrade pip
python -m pip install -e ".[rendering,data,comet]"
~~~

The final command installs this repository in editable mode together with the
rendering, data-processing, and COMET dependencies declared in
`pyproject.toml`. Python package versions are pinned to the experiment
environment. PyTorch and the renderer's system libraries are installed first
so that the intended CUDA build and native Cairo/Pango libraries are used.

## Checkpoints

The final checkpoints for all four models are available in the
[PixelTranslationCheckpoints](https://huggingface.co/shaheen6/PixelTranslationCheckpoints)
repository. It contains MSA checkpoints trained on 5%, 10%, 25%, 50%, and the
full dataset, along with the corresponding fine-tuned checkpoints.

Download only the checkpoint you need. For example:

~~~bash
hf download shaheen6/PixelTranslationCheckpoints \
  --include "pixel/msa/full/*" \
  --local-dir checkpoints
~~~

The resulting checkpoint can be passed directly to the training and evaluation
commands:

~~~bash
--model-checkpoint checkpoints/pixel/msa/full/pytorch_model.bin
~~~

Fine-tuned checkpoints use `best_model.bin`, for example
`checkpoints/pixel/finetuned/full/msa_dialect_1x/best_model.bin`. The Base
directory corresponds to `--model decoder`. Vocab+ also needs the shared
tokenizer:

~~~bash
hf download shaheen6/PixelTranslationCheckpoints \
  --include "vocab/tokenizer/*" \
  --local-dir checkpoints

export VOCAB_TOKENIZER_DIR="$PWD/checkpoints/vocab/tokenizer"
~~~

## Raw data contract

MSA training expects a Hugging Face DatasetDict with train and validation
splits. Fine-tuning accepts a Dataset or DatasetDict. In both cases, rows must
contain source (Arabic) and target (English) strings.

Evaluation expects a DatasetDict. Every named split is scored and must also contain
the source and target columns.

## Prepare MSA inputs

Base and Vocab+ consume raw MSA text. Pixel and Byte use prepared data:

To prepare the data for Pixel and Byte, respectively, use the following commands:
~~~bash
python -m PixelSmolLM.prepare \
  --raw-dataset "$RAW_MSA" \
  --pixel-prepared-dataset "$PIXEL_MSA"

python -m BytesSmolLM.prepare \
  --raw-dataset "$RAW_MSA" \
  --bytes-prepared-dataset "$BYTE_MSA"
~~~

Both use 100% of the training set by default. Use `--percentage` to prepare a
nested prefix for a curriculum experiment.

Training the Vocab+ tokenizer:

~~~bash
python -m DecoderOnly.VocabPlus.train_tokenizer \
  --raw-dataset "$RAW_MSA" \
  --output-dir "$VOCAB_TOKENIZER_DIR"
~~~

## Reproduce MSA training

The reported protocol uses seed 42, learning rate 3e-4, cosine scheduling, 10%
warmup, four GPUs, and global batch 512. Base runs two consecutive full-model
epochs. Vocab+, Byte, and Pixel first run one alignment epoch and then the same
two full-model epochs. Epoch 2 loads epoch-1 weights but resets the optimizer
and scheduler.

~~~bash
python -m training.run \
  --models pixel bytes vocab decoder \
  --raw-dataset "$RAW_MSA" \
  --pixel-prepared-dataset "$PIXEL_MSA" \
  --bytes-prepared-dataset "$BYTE_MSA" \
  --vocab-tokenizer "$VOCAB_TOKENIZER_DIR" \
  --num-gpus 4 \
  --alignment-batch-sizes 64 32 32 \
  --full-batch-sizes 64 32 16 16 \
  --effective-batch-size 512 \
  --output-dir "$CHECKPOINT_ROOT/msa_full" \
  --run-name msa-full
~~~

The wrapper delegates to each model's train_alignment and train_full commands.
Those commands launch distributed workers themselves; do not prefix them with
torchrun.

Training reports to Weights & Biases by default. Authenticate once with
`wandb login`, or set `WANDB_MODE=disabled` for a run that should not be logged.

The batch-size lists always use [Pixel, Byte, Vocab+, Base] order; the three-value alignment list omits Base.

Default batch sizes and gradient accumulation:

| Model | Alignment microbatch / accumulation | Full microbatch / accumulation |
|---|---:|---:|
| Pixel | 64 / 2 | 64 / 2 |
| Byte | 32 / 4 | 32 / 4 |
| Vocab+ | 32 / 4 | 16 / 8 |
| Base | n/a | 16 / 8 |

## Fine-tune

Prepare each training mixture explicitly, then run fine-tuning:
`--model` is required and accepts `pixel`, `bytes`, `vocab`, or `decoder`.

The prepared-dataset argument must match the selected model.
Fine-tuning and evaluation each process one model per invocation.

First prepare the data for the desired model:
~~~bash
python -m finetuning.prepare \
  --model pixel \
  --raw-dataset /path/to/MSA_Dialect_Combined_1x \
  --pixel-prepared-dataset /path/to/prepared/pixel-1x
~~~
Then start finetuning with the model checkpoint from MSA training:
~~~bash
python -m finetuning.run \
  --model pixel \
  --model-checkpoint "$CHECKPOINT_ROOT/msa_full/pixel/full_epoch2/final/pytorch_model.bin" \
  --pixel-prepared-dataset /path/to/prepared/pixel-1x \
  --raw-dataset /path/to/dialects_dev \
  --experiment-dir /path/to/experiments/pixel-1x \
  --run-name pixel-da-msa-1x \
  --num-gpus 4 \
  --train-batch-size 16 \
  --effective-batch-size 128 \
  --eval-batch-size 32 \
  --evaluate-pretrained
~~~


Pixel uses microbatch 16 and accumulation 2 during fine-tuning; the other
models use microbatch 8 and accumulation 4. The runner derives accumulation
from the requested microbatch and GPU count to retain global batch 128.

## Evaluate

Evaluation accepts raw data and scores all DatasetDict splits:

~~~bash
python -m evaluation.run \
  --model pixel \
  --model-checkpoint /path/to/pytorch_model.bin \
  --num-gpus 4 \
  --batch-size 32 \
  --source-layout auto \
  --raw-dataset /path/to/raw_eval_dataset \
  --output-dir /path/to/evaluation/pixel
~~~

It writes results.csv, predictions.csv, validation.json, and a reusable
model-specific prepared-data cache. With `--source-layout auto`, evaluation
reads `training_complete.json` beside the checkpoint. An alignment/full MSA
checkpoint uses its fixed training length (32 words for Pixel/Byte or 256
tokens for Base/Vocab+); a fine-tuned checkpoint uses dynamic batch padding.

See the directory READMEs for the complete commands and output layouts.
