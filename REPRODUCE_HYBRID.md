# Reproduce the hybrid business entity resolution model

Python 3.11+. Install pinned dependencies from this directory:

```bash
python -m pip install -r requirements.txt
```

## Regenerate both outputs from the packaged trained model

Use only the organizer-provided test catalog. No test labels are expected.

```bash
python src/inference.py --data /path/to/dataset/test --index artifacts/index_test --model-dir artifacts/final_model --output output
python src/validate_submission.py --index artifacts/index_test --output output
```

The inference command reads the saved normalization, transliteration, TF-IDF, BM25 and model settings from `model.joblib`, verifies `feature_names.json` and `threshold.json`, builds fresh indexes over **test S2/S3**, and scores every test S1. It never searches the training target catalog.

## Retrain from the provided train files

```bash
python src/io_utils.py --data /path/to/dataset/train --split train --index artifacts/index_train --config artifacts/final_model/model_config.json
python src/tfidf_retriever.py --index artifacts/index_train --config artifacts/final_model/model_config.json
python src/bm25_retriever.py --index artifacts/index_train --config artifacts/final_model/model_config.json
python src/train.py --index artifacts/index_train --config artifacts/final_model/model_config.json --work artifacts/retrain --final artifacts/retrained_model
```

The default is a deterministic sample of 20,000 S1 records split into approximately 60% train, 20% calibration and 20% holdout. The full supplied S2/S3 catalog is indexed. Exact selected S1 IDs and input SHA256 hashes are in `manifest.json`. LightGBM is the predetermined primary model; SGD is a reference on the same pairs/features/splits. The threshold is selected on calibration, not holdout. No ground-truth positives are injected into training candidates.

All texts are preserved as raw Unicode; normalization uses NFKC/casefold with combining marks retained. Optional AnyAscii transliteration creates additional fields without replacing originals. Character TF-IDF vocabulary is learned on a seeded bounded target sample (default 100,000), then **all targets** are transformed and searched in sparse shards. This vocabulary approximation and the shard disk I/O trade-off must be included in any performance report. A configured vocabulary sample of 0 fits all target texts and may require much more memory.

The package includes all source under `src/`, configurations, pinned dependencies and model artifacts. Input datasets and retrieval indices are supplied/rebuilt separately. Only trusted `joblib` files should be loaded. The model is original LightGBM-trained work; no external entity lookup or pretrained neural model is used.
