# Experiment roadmap from Project Explanation

## Implemented baseline

Character TF-IDF union retrieval → 12 lexical/structured features → logistic classifier → OOF-selected macro F0.5 threshold.

## Planned upgrades (not implemented or benchmarked)

| Experiment | Reason | Accept only if |
| --- | --- | --- |
| k = 20 / 50 / 100 | Measure candidate recall ceiling | Recall benefit justifies runtime/memory |
| Rare-token / postal blocks | Recover links missed by character similarity | Recall improves without excessive candidate explosion |
| LightGBM / CatBoost | Nonlinear feature interactions | Untouched group holdout improves |
| MiniLM embeddings + ANN | Semantic retrieval over cached catalog vectors | Checkpoint license/size compliant; multilingual recall measured |
| Cross-encoder shortlist scoring | Jointly inspect name/address pairs | Better precision at tolerable latency |
| Hybrid lexical + neural retrieval | Complementary failure modes | Candidate recall improves by country/noise type |
| Threshold calibration | Reduce costly false merges | Singleton and non-singleton performance both audited |

## Why retrieval is cheaper than scoring every pair

With N S1 records and M target records, an all-pairs cross-encoder needs N×M joint transformer evaluations. A bi-encoder encodes M targets once and each query once; similarity uses cached vectors. Exact similarity still examines M vectors per query, but it is cheaper than rerunning a transformer M times. An ANN index can reduce vector-search work approximately, trading some recall for speed. Reranking K candidates then requires about N×K joint evaluations. Rebuild target embeddings when target text or the encoder changes.

This repository's TF-IDF baseline caches target vectors within a run; it does not implement MiniLM, ANN, persistent embedding caches, or cross-encoder training. A generic retrieval cross-encoder score should not be assumed to be a calibrated business-match probability. Fine-tune or calibrate on legitimate entity labels and repeat group validation.

## Experiment log

| ID | Retrieval | Matcher | Candidate recall | Holdout macro F0.5 | Runtime | Status |
| --- | --- | --- | --- | --- | --- | --- |
| B01 | TF-IDF union, k=20/field | Logistic | Not measured on official data | Not measured | Not measured | Ready to run |

Record source commit, dataset version, seed, thresholds, country breakdown, hardware and peak memory for every real run. No fabricated results or ranks.
