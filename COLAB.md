# Hybrid baseline in Colab

[Open Colab_Baseline.ipynb](https://colab.research.google.com/github/alisalmann7386-crypto/Amazon-ML-Challange-2026/blob/main/notebooks/Colab_Baseline.ipynb)

## Drive inputs

Create `MyDrive/AmazonML2026/input/train/` containing exactly one copy of each training file: Source1, Source2, Source3 and ground truth. TSVs or ZIPs are accepted, including duplicate-download names like `train_source3(1).tsv`; don't include both an archive and its extracted copies. Later place the three test source files in `input/test/`.

The notebook stores completed work in `MyDrive/AmazonML2026/artifacts/<run_key>/`. Heavy operations run under `/content/er_hybrid/<run_key>/`. Change `SAMPLE_SIZE` and the optional LightGBM device in the configuration cell. The default CPU workflow runs on CPU or GPU runtimes; TF-IDF and BM25 stay on CPU.

## Training cells

1. Clone, or fast-forward an existing clean checkout. Local modifications cause a clear stop; they are not overwritten.
2. Install the pinned requirements in an isolated virtual environment, then run local smoke tests.
3. Mount Drive and set paths/config. Restore previously saved artifacts for the exact configuration.
4. Prepare train TSVs locally. Run streaming EDA (default notebook prefix sample; set `EDA_MAX_ROWS=0` for full statistics).
5. Normalize to a strict disk catalog, preserving raw Unicode and optional offline transliterations.
6. Build Unicode/transliterated TF-IDF shards and SQLite BM25 index. Log vocabulary sizes, nnz, memory and build times.
7. Compare all retrieval methods on training groups; generate descriptive positive/random/hard-negative EDA samples.
8. Generate cached pair-feature shards; train LightGBM and the SGD reference on identical splits; calibrate and report untouched holdout scores.
9. Save the model, threshold, feature names, configurations, importance, reports, input hashes and split IDs. Copy completed artifacts to Drive.

Default S1 sample = 20,000; full target catalog is indexed. For a first runtime feasibility check, reduce S1 `SAMPLE_SIZE` to 2,000. TF-IDF vocabulary is sampled separately (default 100,000 target rows). This still indexes every target row, but does not promise vocabulary coverage for rare scripts/tokens.

## Test cells

Keep `RUN_TEST=False` until test files arrive. Set it to `True` and rerun configuration/test cells when ready:

- Prepare the three test TSVs.
- Use saved model settings to build fresh test target indexes.
- Retrieve, score in batches, apply the saved threshold and write both required TSVs.
- Validate all IDs/coverage/subset rules.
- Complete methodology/team details, then create `output/submission.zip`.

There is no `test_ground_truth.tsv` and no locally computed true test F0.5. Upload only `matching_results.tsv` for leaderboard scoring.

## Resume and storage

A configuration-derived run key separates experiments. Catalog/TF-IDF/feature/prediction manifests also verify compatibility and input fingerprints. Reusing an index with changed input data produces an error; select a new `RUN_TAG` for changed datasets. The notebook restores the last completed Drive checkpoint; it does not silently update/reuse different data.

Completed feature and inference shards can be reused. LightGBM/SGD training restarts deterministically if interrupted; it does not resume individual boosting iterations. Incomplete catalog imports restart; completed TF-IDF shards resume. `finally` checkpointing runs for ordinary Python errors/manual interrupts, but cannot survive an abrupt runtime termination. Large checkpoint copies need time and Drive space.

Full-data CPU duration and peak memory remain unmeasured. The workflow bounds feature-generation memory, but LightGBM's training bins and sampled calibration scores need additional RAM. Never interpret synthetic smoke-test scores as challenge performance. See [README](README.md) and [verification](docs/VERIFICATION.md).

## Result labels

- `docs/verification/synthetic_*.json`: tiny software smoke tests only.
- `docs/verification/real_sample_*.json`: real records with a reduced, label-complete target catalog; not competition performance.
- Colab artifacts under `artifacts/<run key>/`: full-target results when the complete S2/S3 files are used.

The full-run notebook prepares four training files, runs EDA, builds TF-IDF and BM25, compares retrieval, trains/calibrates/evaluates, and checkpoints to Drive. Use `SAMPLE_SIZE=20000` for the documented baseline; the target catalog remains complete.
