# Nested MSA curriculum

The curriculum trains from scratch on nested prefixes of the already shuffled,
filtered 24,042,041-row MSA set. Larger percentages contain every example in
smaller percentages but never load their checkpoints.

| Percentage | Training rows |
|---:|---:|
| 5 | 1,202,102 |
| 10 | 2,404,204 |
| 25 | 6,010,510 |
| 50 | 12,021,020 |

## 1. Prepare Pixel and Byte prefixes

Base and Vocab+ consume raw text. Pixel and Byte use their model-specific preparers
through this thin wrapper:

~~~bash
python -m msa_curriculum.prepare \
  --model pixel --percentage 10 \
  --raw-dataset "$RAW_MSA" \
  --pixel-prepared-dataset "$PIXEL_PCT10"

python -m msa_curriculum.prepare \
  --model bytes --percentage 10 \
  --raw-dataset "$RAW_MSA" \
  --bytes-prepared-dataset "$BYTE_PCT10"
~~~

## 2. Train

~~~bash
python -m msa_curriculum.train \
  --models pixel bytes vocab decoder \
  --percentage 10 \
  --raw-dataset "$RAW_MSA" \
  --pixel-prepared-dataset "$PIXEL_PCT10" \
  --bytes-prepared-dataset "$BYTE_PCT10" \
  --vocab-tokenizer "$VOCAB_TOKENIZER_DIR" \
  --alignment-batch-sizes 64 32 32 \
  --full-batch-sizes 64 32 16 16 \
  --effective-batch-size 512 \
  --num-gpus 4 \
  --output-dir "$CHECKPOINT_ROOT/msa_curriculum/pct10" \
  --run-name msa-pct10
~~~

msa_curriculum.train is an alias of training.run, so curriculum and 100% MSA
training share the same alignment/full implementations.

## 3. Prepare 1:1 fine-tuning data

~~~bash
python -m finetuning.prepare \
  --model pixel \
  --raw-dataset /path/to/MSA_Dialect_Combined_1x \
  --pixel-prepared-dataset /path/to/prepared/pixel-1x
~~~

Repeat with the corresponding model-specific argument for the other models.

## 4. Evaluate, fine-tune, and evaluate again

~~~bash
python -m msa_curriculum.finetune_eval \
  --model pixel \
  --model-checkpoint "$CHECKPOINT_ROOT/msa_curriculum/pct10/pixel/full_epoch2/final/pytorch_model.bin" \
  --pixel-prepared-dataset /path/to/prepared/pixel-1x \
  --raw-dataset /path/to/dialects_dev \
  --experiment-dir /path/to/curriculum_work/pct10/pixel \
  --run-name pixel-pct10-1x \
  --num-gpus 4 \
  --train-batch-size 16 \
  --effective-batch-size 128 \
  --eval-batch-size 32 \
  --skip-comet
~~~

The wrapper evaluates the incoming MSA checkpoint with its fixed layout,
fine-tunes for one epoch, and evaluates the resulting checkpoint dynamically.
