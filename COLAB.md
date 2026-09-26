# Run the project in Google Colab

[Open the notebook](https://colab.research.google.com/github/alisalmann7386-crypto/Amazon-ML-Challange-2026/blob/main/notebooks/Colab_Baseline.ipynb), select **Copy to Drive**, and run cells from top to bottom.

## 1. Drive folders

Create:

```text
MyDrive/AmazonML2026/input/train/
```

Upload exactly one copy of:

```text
train_source1.tsv
train_source2.tsv
train_source3.tsv
train_ground_truth.tsv
```

Plain TSVs or ZIPs are accepted. Do not keep both a ZIP and its extracted TSV in the input folder.

## 2. Configuration

Start with:

```python
SAMPLE_SIZE = 20000
EDA_MAX_ROWS = 10000       # Use 0 only when you want a full streaming EDA.
LIGHTGBM_DEVICE = "cpu"
RUN_TAG = "integrity_v2_k20"
RUN_TEST = False
```

The model does not require a GPU. TF-IDF and BM25 remain CPU operations. A GPU can help only LightGBM and does not solve excessive candidate generation.

## 3. What the notebook does

1. Mounts Drive and clones or safely fast-forwards the repository.
2. Installs pinned packages and runs the test suite.
3. Copies the TSV bytes to local Colab storage and writes a copy manifest.
4. Runs `data_integrity.py` before EDA. Stop here if it reports unknown label IDs.
5. Runs EDA on the verified data.
6. Normalizes Unicode text and creates optional transliterations.
7. Builds the full Source-2/Source-3 TF-IDF and BM25 indexes.
8. Runs the grouped retrieval pilot. K=20 is the default; K=50 is diagnostic only.
9. Generates bounded feature shards and trains LightGBM plus SGD.
10. Selects thresholds on calibration and evaluates the untouched grouped holdout.
11. Saves model files and checkpoints artifacts to Drive.

## 4. Understanding Source-3 warnings

Two recoverable source-row forms are handled without dropping the entity:

- three columns: the final country is treated as missing;
- more than four columns: extra tab fragments are joined into the address.

Each repair is recorded in `artifacts/data_integrity.json` and later in `catalog.json`. The original TSV is unchanged. Any other malformed row stops the run.

If the integrity report says ground truth references target IDs that do not exist, do not continue training. Check that Source 3 and ground truth came from the same download/version. Delete neither raw file; use the printed SHA-256 values to compare copies.

## 5. Resume behavior

Artifacts are stored under:

```text
MyDrive/AmazonML2026/artifacts/<run_key>/
```

The run key includes the configuration. Use a new `RUN_TAG` after changing input files or K. Completed compatible shards resume; changed hashes/configurations stop with an explicit error. A `.building.sqlite` file is incomplete and is never treated as the final catalog.

## 6. Test inference later

When the test files arrive, upload only:

```text
test_source1.tsv
test_source2.tsv
test_source3.tsv
```

Set `RUN_TEST=True`. The notebook builds fresh test indexes, loads the saved model and threshold, writes both TSV outputs, validates them and creates the ZIP. It must not expect or create `test_ground_truth.tsv`.

Before the hybrid submission, run the notebook's exact-baseline cell. It produces a no-ML `matching_results.tsv` using normalized business name plus country and validates the complete submission path.

## 7. Result labels

- Synthetic metrics: software smoke tests only.
- Sampled-real metrics: engineering checks, not competition performance.
- Full-target pilot metrics: blocking recall and compute estimates, not test F0.5.
- Test F0.5: available only from the official leaderboard because test labels are not supplied.
