# Dataset preparation

The pipeline builds normalized Arabic sources. All default input and output
paths are derived from `DATA_ROOT`, which defaults to `data` in the repository
root. Set it explicitly when the corpora live elsewhere:

```bash
export DATA_ROOT=/path/to/datasets
```

## MSA

The raw MSA processor expects aligned UTF-8 text files at:

```text
$DATA_ROOT/pixel_translation_MSA_raw/
├── opus/
│   ├── OpenSubtitles.ar-en.ar
│   └── OpenSubtitles.ar-en.en
├── qed/
│   ├── QED.ar-en.ar
│   └── QED.ar-en.en
├── ted/
│   ├── NeuLab-TedTalks.ar-en.ar
│   └── NeuLab-TedTalks.ar-en.en
└── unpc/
    ├── UNPC.ar-en.ar
    └── UNPC.ar-en.en
```

Run:

```bash
python -m dataset_processing.MSA.process_msa_raw_datasets \
  --workspace-output-path "$DATA_ROOT/pixel_translation_MSA_train"
```

This command performs language filtering, Arabic normalization, the 32-word
source/target filter, and merging. Its output is the training `Dataset`. Add
the validation split and save the final input as a Hugging Face `DatasetDict`
with `train` and `validation` splits. The reported experiments use the shuffled
`pixel_translation_MSA_filtered_32` DatasetDict, containing 24,042,041 training
rows. Pixel and Byte conversion from that DatasetDict is described in the root
README.

## Dialects

Dialect sources are read from `$DATA_ROOT/dialects/raw`, using this layout:

```text
$DATA_ROOT/dialects/raw/
├── Alexandria/{train,dev,test}.{ar,en}.tsv
├── MADAR/MADAR.<language-or-city>.{train,dev,test}.tsv
├── arabench/
│   ├── AraBench_dataset/
│   └── LDC.xml
└── FLORES/<language-variant>/
```

FLORES variants must first be downloaded as Hugging Face datasets under the
shown directory. The expected variants are listed in
`dataset_processing/dialects/1flores.py`.

Run the numbered scripts in order:

```bash
python -m dataset_processing.dialects.1alexandrie
python -m dataset_processing.dialects.1arabench_splits
python -m dataset_processing.dialects.1flores
python -m dataset_processing.dialects.1madar
python -m dataset_processing.dialects.2combine_dialects
python -m dataset_processing.dialects.4combine_MSA_dialects
```

The first four scripts normalize sources. `2combine_dialects` writes
`dialects_train`, `dialects_dev`, and `dialects_test` below
`$DATA_ROOT/dialects/processed`. The final script creates `Dialect_Combined`
and the `MSA_Dialect_Combined_{1,2,4}x` training mixtures. It reads the MSA
DatasetDict from `$DATA_ROOT/pixel_translation_MSA_filtered_32` by default;
set `MSA_DATASET_PATH` to override that location.

## Dotless and diacritized robustness sets

```bash
python -m dataset_processing.dialects.3add_noise --splits dev test --num-workers 4
```

For each clean split, the script deterministically samples at most 500 rows with seed 42 and adds `dotless_...` and `diacritics_...` variants while retaining the clean split. It uses a position-sensitive dotless mapping and CAMeL BERT disambiguators. If `calima-msa-s31.db` and `calima-egy-c044.db` are present beside the script they are used directly; otherwise the same named databases must be installed through CAMeL Tools (`camel_data`). Local database binaries are ignored and are not committed.

Every preparation command writes a new Hugging Face dataset directory. Keep raw and generated datasets outside Git.
