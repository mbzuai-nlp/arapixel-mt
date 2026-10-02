# Evaluation

`evaluation.run` launches distributed evaluation from a normal Python command.
Its `evaluation.evaluate` worker prepares the selected representation,
generates one translation per row, scores every split, and validates the
outputs.

## Expected raw dataset format

The raw path must be a Hugging Face DatasetDict. Every named split is
evaluated, and every split must contain:

- source: Arabic input string;
- target: English reference string.

Split names are preserved in the eval_split column. Optional IDs and metadata
columns may be present and are ignored by generation.

## Run

~~~bash
python -m evaluation.run \
  --model pixel \
  --num-gpus 4 \
  --batch-size 32 \
  --model-checkpoint /path/to/pytorch_model.bin \
  --source-layout auto \
  --raw-dataset /path/to/raw_eval_dataset \
  --output-dir /path/to/evaluation/pixel
~~~

The command writes results.csv, predictions.csv, validation.json, and a derived
pixel-prepared-dataset cache below the output directory. To reuse a prepared
cache, pass the matching model-specific argument: pixel-prepared-dataset,
bytes-prepared-dataset, decoder-prepared-dataset, or vocab-prepared-dataset.

Because MSA pretraining and fine-tuning use fixed and dynamic padding,
respectively, evaluation must know the source layout to generate the position
IDs expected by the model.

`--source-layout auto` is the default. Evaluation looks for
`training_complete.json` in the same directory as the checkpoint:

- `"stage": "alignment"` or `"stage": "full"` means an MSA checkpoint, so
  evaluation uses the fixed MSA layout: 32 words for Pixel/Byte or 256 tokens
  for Base/Vocab+.
- `"training_phase": "finetuning"` means a fine-tuned checkpoint, so
  evaluation dynamically pads each batch to its longest source.

If the metadata file is missing, the fallback is the weights filename:
`pytorch_model.bin` is treated as MSA and `best_model.bin` as fine-tuned. For
renamed or externally produced checkpoints, pass
`--source-layout msa` or `--source-layout dynamic` explicitly.

Metrics are BLEU, spBLEU, chrF++, and COMET. Pass `--skip-comet` to compute only
BLEU, spBLEU, and chrF++. Generation defaults to batch 32 per GPU, four beams,
and 128 new tokens.
