# Train the large-data baseline in Google Colab

[Open the notebook](https://colab.research.google.com/github/alisalmann7386-crypto/Amazon-ML-Challange-2026/blob/main/notebooks/Colab_Baseline.ipynb)

## What this version actually does

`src/scalable.py` uses a disk-backed **SQLite FTS5 token BM25 index** over the full S2/S3 catalog, unions top-20 name and top-20 address results, computes 12 lexical features, and trains **SGD logistic regression** incrementally. This is a separate CPU baseline from the original character TF-IDF `pipeline.py`. It is not MiniLM or a cross-encoder.

The default training experiment samples **20,000 S1 entities** reproducibly, approximately 60% training / 20% calibration / 20% untouched holdout, grouped by shared labeled targets within the sample. Increase `SAMPLE_SIZE` after inspecting candidate recall, runtime and RAM. All S2/S3 records remain searchable regardless of the sampled S1 set. No label-based positive injection is used. This is not a claim that all 2.2 million S1 records train the default model.

The classifier reads compressed feature shards in batches. Calibration/holdout pair scores are collected in memory; raising sample size substantially still increases RAM use. Country labels are unrestricted. Shared targets are grouped; unlabeled near-duplicates can still require stronger grouping.

## Prepare Google Drive

Create a folder `MyDrive/AmazonML2026/input/train` containing exactly one copy of each:

- `train_source1.tsv`
- `train_source2.tsv` or a ZIP containing it
- `train_source3.tsv` or a ZIP containing it
- `train_ground_truth.tsv`

TSVs may be nested in ZIP folders. Windows duplicate names such as `train_source3(1).tsv` are accepted and renamed when copied. Do not place both a ZIP and an extracted copy in the input folder. The notebook deliberately rejects ambiguous duplicates. Later put the three test files in `input/test`.

A ChatGPT upload is not automatically available to Colab. Put the original files or ZIPs in your Drive and update the notebook's `DRIVE_ROOT` if needed. Competition data is not pushed into the public repository.

## Run cells in order

1. Clone the repo and install pinned dependencies in an isolated Python environment.
2. Mount Drive, set paths, and run the synthetic smoke tests.
3. Copy/extract the provided data into Colab's local disk.
4. Build or restore the full-catalog training index. Completed indices are cached on Drive; incomplete index builds restart.
5. Train and evaluate. Completed feature shards and results are saved under a configuration-specific Drive run folder. Rerunning training reuses feature shards and restarts the estimator deterministically.
6. Inspect `metrics.json`, especially **holdout macro F0.5**, candidate recall and singleton accuracy. Calibration F0.5 is used for threshold selection and is not a holdout score.
7. Once test files are supplied, enable `RUN_TEST`, build the test index, predict both TSVs, validate and package.

No GPU is required. Choose a CPU runtime; a high-RAM runtime may help larger experiments. Index building and millions of query searches may take substantial time and disk space; no full-data runtime has been measured. Token retrieval can miss severe typos/transliterations; BM25 ranking with common terms can also be slow. Measure recall and profile before increasing the sample. A small k is not evidence of adequate recall.

## Command-line reproduction

Run from the cloned root (or the packaged code directory for prediction). Python 3.11+:

```bash
python -m pip install -r requirements.txt
python src/prepare_data.py --input /path/to/input/train --output dataset/train --split train
python src/scalable.py index --data dataset/train --split train --index artifacts/train.sqlite
python src/scalable.py train --index artifacts/train.sqlite --work artifacts/run --sample-size 20000 --k 20 --epochs 5 --seed 42
```

Inspect `artifacts/run/metrics.json`. The model is `artifacts/run/model.joblib`. `manifest.json` records input SHA256 hashes and the exact sampled split IDs. `holdout_pairs.tsv` contains candidate labels/scores for error analysis; missed true links are reflected in recall/metrics, though they have no scored candidate row. Model and reports are not committed by default.

When test data is available:

```bash
python src/prepare_data.py --input /path/to/input/test --output dataset/test --split test
python src/scalable.py index --data dataset/test --split test --index artifacts/test.sqlite
python src/scalable.py predict --index artifacts/test.sqlite --model artifacts/run/model.joblib --output output
python src/scalable.py validate --index artifacts/test.sqlite --output output
python src/package_submission.py --team YOUR_TEAM --index artifacts/test.sqlite --model artifacts/run/model.joblib --output output
```

The package includes the trained model at `code/business_entity_resolution/artifacts/model.joblib`. **Inside the extracted package**, regenerate predictions with:

```bash
python src/scalable.py index --data /path/to/dataset/test --split test --index artifacts/test.sqlite
python src/scalable.py predict --index artifacts/test.sqlite --model artifacts/model.joblib --output output
```

To retrain from scratch, use the training command above with the same sample size, k, epochs and seed saved in the model artifact. The default notebook uses the listed settings; record custom values in the methodology. Update team details and real metrics in `Documentation_template.md` before packaging. Only upload `matching_results.tsv` to the live leaderboard; retain both files for the final ZIP.

## Checkpoints and integrity

- Completed indices are reused only when SHA256 input fingerprints match. Choose a new path for different data.
- Feature caches reject a changed sample, seed, k, source hash or implementation version. Use a new work folder for changed settings.
- Predictions stream one S1 at a time and are staged under temporary names. Inference restarts after interruption; it does not resume mid-output.
- The disk-based validator checks coverage, duplicates, target identity and match-subset consistency without loading the entire dataset into Python.
- Only load trusted `joblib` artifacts.
- Full-data accuracy and Colab runtime behavior remain unverified until you run the notebook with all four training files. Local tests use synthetic fixtures.
