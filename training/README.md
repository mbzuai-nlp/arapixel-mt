# Shared MSA training

The model directories provide model-specific training commands.
`training.run` is a convenience wrapper that runs those commands sequentially
for one or more model families. Each command uses `training.pretraining` for
one optimizer and scheduler stage.

## Training regime

- Base: two full-model epochs.
- Vocab+, Byte, Pixel: one alignment epoch, then two full-model epochs.

Each model `train_full` command runs both full epochs.

Training reports to Weights & Biases by default. Run `wandb login` before the
first logged experiment, or set `WANDB_MODE=disabled` to train without logging.

Available arguments:

- `--output-dir`
- `--run-name`
- `--percentage`
- `--num-gpus`
- `--alignment-batch-sizes` (Pixel, Byte, Vocab+)
- `--full-batch-sizes` (Pixel, Byte, Vocab+, Base)
- `--effective-batch-size`
- `--seed`
- `--learning-rate`
- `--warmup-ratio`
- `--save-steps`
- `--eval-steps`

Data arguments differ only by representation:

| Model | Training data argument | Alignment checkpoint |
|---|---|---|
| Pixel | pixel-prepared-dataset | required for full training |
| Byte | bytes-prepared-dataset | required for full training |
| Vocab+ | raw-dataset and vocab-tokenizer | required for full training |
| Base | raw-dataset | none |

## Run all models

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

The default percentage is 100. Percentages 5, 10, 25, and 50 select the same
nested-prefix logic used by msa_curriculum.

Models run sequentially in the order passed to `--models`. Each model receives
the requested number of GPUs, so this command uses four GPUs at a time rather
than sixteen GPUs concurrently. Gradient accumulation is derived from the
microbatch, GPU count, and effective batch size.

A completed stage is skipped only when both final weights and
training_complete.json exist; an interrupted stage resumes from its newest
valid Trainer checkpoint. Use dry-run to print every distributed command.

~~~text
<output-dir>/<model>/alignment/final/pytorch_model.bin
<output-dir>/<model>/full_epoch1/final/pytorch_model.bin
<output-dir>/<model>/full_epoch2/final/pytorch_model.bin
~~~
