# What was actually run

## Automated checks

The local suite covers the original baselines and hybrid workflow, including empty target corpora, multilingual normalization, typo/cross-script retrieval, grouped splitting, global candidate caps, recoverable malformed-row preservation, model save/load, finite/range-checked thresholds, exact-baseline output, resumed inference and strict output validation. Run the command in the README to obtain the current test count for your checkout.

Local notebook code cells were parsed for Python syntax and modules imported. These checks do not mean the notebook was executed inside an actual Colab runtime.

A separate synthetic integration run built both retrieval backends, evaluated A/B/C/D retrieval modes, generated EDA pair samples/distributions, trained/calibrated both models, exported errors, saved/loaded the model, inferred and packaged outputs. The extracted ZIP then rebuilt fresh test indexes and reproduced both TSV files byte-for-byte.

## Recorded synthetic results — not competition results

Fixture: 12 S1 records, 24 S2/S3 targets; 7 training, 2 calibration, 3 holdout. One business variant is Devanagari. Smoke-test k=3 per channel, min_df=1, small matrix shards, LightGBM 20 rounds/min_child_samples=1. These differ from production defaults and exist solely for fast execution checks.

| Model | Holdout macro F0.5 | Micro precision | Micro recall | Singleton accuracy |
| --- | ---: | ---: | ---: | ---: |
| LightGBM | 0.571429 | 0.666667 | 1.0 | 0.0 |
| SGD | 1.0 | 1.0 | 1.0 | 1.0 |

LightGBM remains the requested primary model; no model choice was made using this holdout. The tiny fixture is inadequate for judging model quality.

All four retrieval comparisons had training-group candidate micro recall 1.0. Average candidate counts: TF-IDF 3.5714, BM25 3.0, hybrid 4.7143, hybrid+transliteration 4.7143. This does not establish that transliteration improves real-data recall. [Raw synthetic model report](verification/synthetic_metrics.json), [retrieval report](verification/synthetic_retrieval.json).

## Real-data run

The earlier EDA artifact reported S3=5,285,603, but the currently supplied Source-3 file was counted as 5,274,632 rows in both Drive and the prepared local copy. Therefore `verification/real_full_eda.json` is historical and must not be treated as the current file truth. Run `src/data_integrity.py` on the exact current four files before publishing row counts or training results.

A deterministic sampled-real run used 2,000 S1 rows and 206,896 S2/S3 targets, retaining all 6,896 labeled targets plus 100,000 hashed distractors per source. It used Unicode and transliterated TF-IDF/BM25 channels, 91 features, grouped 1,200/400/400 splits, calibration-only thresholds and untouched holdout evaluation. See [real results](REAL_RESULTS.md). These are historical sampled-catalog metrics from the older uncapped configuration, not full-catalog competition performance or the final v2 model.

## Not executed here

Full 10.3-million-target indexing/training, actual Colab execution, GPU execution, official test-file inference and leaderboard submission. The complete target catalog exceeds this Work runtime's 32 GB disk budget; use the prepared Colab notebook.

## Next experiment

First reconcile current file hashes and ground-truth IDs. Then run the full target catalog with the grouped training pilot: global K=20 first, K=50 only if needed, and targeted rescue if K=50 recall remains below the configured floor. Keep calibration and holdout fixed and untouched while choosing model thresholds.
