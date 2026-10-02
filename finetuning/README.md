# Dialect fine-tuning

Fine-tuning has two explicit steps so a prepared mixture can be reused:

1. finetuning.prepare converts raw source/target rows to one model-specific
  representation.
2. finetuning.run trains from that prepared dataset and evaluates on a raw
   dev/test DatasetDict.

## Raw and prepared data

The raw training input may be a Hugging Face Dataset or DatasetDict. Each row
must contain source (Arabic) and target (English) string columns.

| Model | `--model` value | Prepared-data argument |
|---|---|---|
| Pixel | `pixel` | `--pixel-prepared-dataset` |
| Byte | `bytes` | `--bytes-prepared-dataset` |
| Vocab+ | `vocab` | `--vocab-prepared-dataset` |
| Base | `decoder` | `--decoder-prepared-dataset` |

~~~bash
python -m finetuning.prepare \
  --model pixel \
  --raw-dataset /path/to/MSA_Dialect_Combined_1x \
  --pixel-prepared-dataset /path/to/prepared/pixel-1x
~~~

Pixel/Byte training preparation defaults to 64 source words. Base/Vocab+
defaults to 256 source tokens. Targets remain dynamic unless capped explicitly.

## Train and evaluate

~~~bash
python -m finetuning.run \
  --model pixel \
  --model-checkpoint /path/to/msa/full_epoch2/final/pytorch_model.bin \
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

The runner delegates to `finetuning.train` and `evaluation.run`. It does not
prepare the training mixture on the fly. Incoming MSA evaluation uses the fixed
MSA layout; post-fine-tuning evaluation uses dynamic padding.

Defaults are one epoch, learning rate 3e-5, cosine scheduling, 10% warmup, and
seed 42. Pixel uses microbatch 16; the other models use 8. Gradient
accumulation is derived from the microbatch and GPU count to retain global
batch 128. Evaluation batch size is per GPU. Fine-tuning reports to Weights &
Biases by default; set `WANDB_MODE=disabled` to disable logging.

~~~text
<experiment-dir>/prepared/<model>-prepared-dataset/
<experiment-dir>/checkpoints/<model>/best_model.bin
<experiment-dir>/results/<model>/pre/
<experiment-dir>/results/<model>/post/
~~~

Each result directory contains results.csv, predictions.csv, and
validation.json.
