# What was actually run

## Automated checks

All 16 local tests pass. The suite covers the original baselines and hybrid workflow, including empty target corpora, multilingual normalization, typo/cross-script retrieval, grouped splitting, model save/load, finite/range-checked thresholds, resumed inference and strict output validation.

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

All four uploaded training files were read successfully. Data rows: S1 2,206,821; S2 5,034,616; S3 5,285,603; ground truth 2,206,821. Full EDA is saved in `verification/real_full_eda.json`.

A deterministic sampled-real run used 2,000 S1 rows and 206,896 S2/S3 targets, retaining all 6,896 labeled targets plus 100,000 hashed distractors per source. It used Unicode and transliterated TF-IDF/BM25 channels, 91 features, grouped 1,200/400/400 splits, calibration-only thresholds and untouched holdout evaluation. See [real results](REAL_RESULTS.md). These are sampled-catalog metrics, not full-catalog competition performance.

## Not executed here

Full 10.3-million-target indexing/training, actual Colab execution, GPU execution, official test-file inference and leaderboard submission. The complete target catalog exceeds this Work runtime's 32 GB disk budget; use the prepared Colab notebook.

## Next experiment

Use a manageable real S1 training sample with the full target catalog. Inspect candidate misses by country/script, compare A/B/C/D, and vary per-channel k and vocabulary sample size using training/calibration only. Keep the final holdout fixed and untouched while choosing retrieval settings. Prioritize candidate recall and singleton false merges before increasing model complexity.
