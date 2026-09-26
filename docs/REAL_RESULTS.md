# Real-data verification results

These measurements use real challenge records, but they are **sampled-catalog engineering results, not full-catalog competition performance**. The Work runtime has a 32 GB disk limit, while the full corpus contains 10,320,219 S2/S3 targets. The deterministic run used 2,000 S1 queries and a label-complete catalog of 206,896 real targets: every 6,896 labeled target for those queries plus 100,000 hashed distractors from each target source. Retrieval never receives ground-truth IDs.

## Exact full-data EDA

The EDA scan used every row in all four real files (not a prefix sample).

| File | Rows | Duplicate IDs | Missing names | Missing addresses | Missing countries | Non-ASCII |
|---|---:|---:|---:|---:|---:|---:|
| Source 1 | 2,206,821 | 0 | 0 | 0 | 0 | 0.0251% |
| Source 2 | 5,034,616 | 0 | 0 | 168,967 | 0 | 21.9822% |
| Source 3 | 5,285,603 | 0 | 0 | 175,916 | 0 | 18.7239% |

Ground truth contains 123,247 singletons (5.5848%), 119,157 one-match records and 1,964,417 multi-match records. There are 7,638,365 positive links: 3,693,619 to Source 2 and 3,944,746 to Source 3, averaging 3.4613 matches per S1. Country, length/token, script, numeric/postal, common-token and rare-token details are in `docs/verification/real_full_eda.json`.

## Retrieval (1,200 grouped training queries)

| Method | R@5 | R@10 | R@20 | R@30 | R@50 | Candidate micro recall | Avg candidates |
|---|---:|---:|---:|---:|---:|---:|---:|
| Character TF-IDF | 0.8891 | 0.9694 | 0.9783 | 0.9812 | 0.9841 | 0.9824 | 37.48 |
| BM25 | 0.8860 | 0.9743 | 0.9834 | 0.9863 | 0.9885 | 0.9875 | 37.50 |
| TF-IDF + BM25 | 0.9108 | 0.9747 | 0.9839 | 0.9865 | 0.9909 | 0.9916 | 59.98 |
| Hybrid + transliteration | 0.9117 | 0.9759 | 0.9853 | 0.9882 | 0.9921 | 0.9928 | 62.41 |

## Models (grouped 1,200 / 400 / 400 split)

| Model | Calibration threshold | Calibration macro F0.5 | Holdout macro F0.5 | Precision | Recall | Singleton accuracy |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM | 0.580 | 0.9793 | 0.9839 | 0.9879 | 0.9689 | 0.9630 |
| SGD reference | 0.775 | 0.9630 | 0.9720 | 0.9868 | 0.9393 | 0.9630 |

LightGBM was selected as the primary model before holdout evaluation. Thresholds were selected only on calibration. The holdout candidate micro recall was 0.9963, and LightGBM predicted 3.315 matches per S1 on average.

Machine-readable results are in `docs/verification/real_sample_retrieval.json`, `real_sample_training.json`, and `real_sample_manifest.json`. The fitted sampled-real artifacts are in `models/real_sample/`.
