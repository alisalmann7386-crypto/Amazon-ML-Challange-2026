# Cost-aware blocking strategy

## Decision

Use the existing character TF-IDF, BM25 and transliteration channels together. Merge duplicate target IDs, rank with reciprocal-rank fusion, and keep a **global top 20** candidates per Source-1 entity by default.

Test global K=50 only when K=20 shows an important true-link recall gap. Do not run K=100 in the current Colab workflow.

## Why

With about 2.2 million Source-1 rows:

| Global K | Maximum full-run candidate pairs | Decision |
| ---: | ---: | --- |
| 20 | about 44 million | default |
| 50 | about 110 million | conditional ceiling |
| 100 | about 220 million | excluded |

These pairs still need feature generation and model scoring. At 40 float32 values, 110 million rows require about 17.7 GB for the numeric matrix alone, before IDs, DataFrames, indexes and temporary objects.

## Execution rule

1. Pass data integrity and corrected EDA.
2. Run the hybrid retriever on the grouped training pilot at K=20.
3. Record candidate micro recall, macro coverage, queries with a true candidate, candidate counts, runtime, peak RAM and disk use.
4. Run K=50 on the same queries only if K=20 loses meaningful positives.
5. Choose the smallest affordable K.
6. If K=50 recall is below about 93%, fix the blocker with a targeted rescue channel instead of increasing K again.

## Targeted rescue

Inspect the missed known positives and add only the channel needed for the failing slice:

- divergent names: emphasize address and numeric overlap;
- short or initial names: use locality, house number and rare address tokens;
- cross-script names: use the transliteration channel while preserving Unicode;
- missing addresses: rely on name and country with a strict budget;
- country/locality failures: allow a bounded fallback rather than an unrestricted global block.

Merge rescue candidates with the normal hybrid candidates, deduplicate them, rerank, and apply the same strict global candidate budget.

## Two different costs

The **retrieval cost** is the work needed to search TF-IDF shards and BM25 posting lists. The **ML cost** is the number of final unique pairs sent to feature generation and the classifier.

A top-20 output can still be expensive if a common token forces a very large posting scan. Therefore the pilot must report retrieval time and scanned work as well as the final pair count.

## Evaluation rules

- Never inject ground-truth positives into candidate lists.
- Evaluate all true links for multi-match Source-1 entities.
- Choose the model threshold on calibration only.
- Keep the grouped holdout untouched.
- Do not describe raw key-sharing coverage as Recall@20, model F0.5 or competition performance.
- Run an exact normalized-name-plus-country leaderboard baseline before expensive hybrid test inference.
