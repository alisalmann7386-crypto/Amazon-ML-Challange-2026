# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [To be supplied]  
**Team Members:** [To be supplied]  
**Submission Date:** [Set when submitting]

---

## 1. Executive Summary

The baseline combines sparse character TF-IDF and SQLite BM25 retrieval over normalized Unicode and optional offline transliterated fields. A LightGBM pair classifier uses lexical, vector, structured and retrieval-channel features; a threshold selected only on calibration produces zero/one/many matches per S1. SGD is a reference trained on identical pairs and splits. The repository includes a sampled-real engineering run; it is not a full-catalog competition score.

## 2. Methodology

### 2.1 Problem Analysis

Supplied data may contain abbreviations, script differences, typos, incomplete addresses and a small number of recoverable malformed TSV rows. Copy the exact row counts and SHA-256 hashes from the current run's `artifacts/data_integrity.json`. Do not reuse the historical Source-3 count in `docs/verification/real_full_eda.json`; it conflicts with the currently supplied file.

### 2.2 Solution Strategy

**Approach Type:** Hybrid Blocking + Classical Classifier  
**Core Innovation:** Complementary Unicode and transliterated candidate channels, auditable union metadata, and grouped calibration/holdout validation. No novel algorithm or accuracy advantage is claimed.

Strict TSV ingestion preserves string IDs and raw fields; a normalized disk catalog retains NFKC/casefold text and combining marks. Conservative legal-suffix/address maps are configurable. Suffix stripping affects only a secondary core name. `St → Street` is disabled by default because of Saint/Street ambiguity. AnyAscii creates an additional offline representation; it does not translate or look up business identities.

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** Unicode/transliterated names and addresses, character TF-IDF cosine and field-specific BM25 token queries.
- **Candidate pairs generated:** In the sampled-real run, hybrid+transliteration produced 62.41 candidates per training query on average. The configured default is top-20 per active channel, up to eight channels before deduplication.
- **How true matches are protected:** Union complementary channels and measure recall; no guarantee is claimed. Never inject ground-truth positives into classifier candidates. Report missing links, macro coverage, Recall@5/10/20/30/50, candidate-count quantiles and reduction ratio.

TF-IDF uses 3–5 `char_wb` n-grams, float32, sublinear TF and L2 normalization by default. A seeded target sample (up to 100,000 rows) fits the vocabulary; all supplied target rows are transformed into persisted CSR shards. Exact sparse top-N scans every shard without a dense all-pairs matrix. BM25 uses SQLite FTS5 disk indexes. Candidate metadata preserves native channel scores and ranks. Recall@K comparisons use reciprocal-rank fusion; classifier inference scores the entire configured union.

## 4. Matching Model

**Features used:** 91 fixed-order features spanning Unicode/transliterated names and addresses: exact/core equality; Levenshtein, Jaro and Jaro-Winkler; Jaccard/overlap/containment; token-set/sort; prefix and length ratios; character/word TF-IDF cosine; numeric intersections/conflicts; house/postal heuristics; rare-token overlap; country/script agreements; retrieval scores/flags and interaction indicators.

**Model type:** LightGBM binary classifier, CPU by default; optional GPU attempt falls back to CPU. SGD logistic regression with a train-only scaler is a separately calibrated reference. All retrieved positives and negatives are retained; top-5-channel nonmatches are counted as hard negatives. EDA random negatives are not inserted into training.

**Threshold selection:** Exact macro F0.5 on a calibration-only configurable threshold grid, including no-candidate records and singletons. Ties select the higher threshold.

Default S1 sample: 20,000, deterministically selected. Approximately 60/20/20 train/calibration/holdout. Full-label-graph connected components keep linked S1 entities together, even via unsampled intermediaries. All target texts are available for unsupervised indexing; calibration and holdout labels do not fit models. LightGBM is designated primary before evaluating holdout.

## 5. Results & Error Analysis

- **Full-catalog/leaderboard macro F0.5:** Not measured. No public/private leaderboard result is claimed.
- **Sampled-real engineering result:** 2,000 real S1 rows and a label-complete 206,896-target catalog, grouped 1,200/400/400. Hybrid+transliteration candidate micro recall was 0.9928 on training queries. LightGBM selected threshold 0.580 on calibration (macro F0.5 0.9793) and achieved holdout macro F0.5 0.9839, precision 0.9879, recall 0.9689 and singleton accuracy 0.9630. SGD holdout macro F0.5 was 0.9720. These are not full-catalog competition scores.
- **Synthetic verification only:** 12 S1 records, 24 targets; split 7/2/3. LightGBM holdout macro F0.5 = 0.5714285714, SGD = 1.0. LightGBM made two false-positive links and no false negatives in this tiny holdout. This fixture verifies execution and must not be presented as challenge accuracy.
- **Synthetic retrieval:** A/B/C/D training-group candidate micro recall was 1.0 for this easy fixture. Average candidates were about 3.57 / 3.00 / 4.71 / 4.71 at the smoke-test k=3. No transliteration benefit is inferred from this aggregate; a separate cross-script unit test verifies retrieval functionality.
- **Error exports:** False positives/negatives, retrieval misses, singleton false merges, numeric conflicts, high-name/low-address, low-name/high-address, cross-script failures and transliteration-only retrieval outcomes.

## 6. Conclusion

The repository implements and tests the requested multilingual classical baseline with independent calibration/holdout evaluation and reproducible submission outputs. The sampled-real experiment supports using the hybrid and transliteration union for the next full-target Colab run, while full-catalog retrieval recall, country/script transfer and resource use remain to be measured. Neural retrieval and cross-encoders are not part of this baseline.

## Appendix

### A. Code Artefacts

All source is under `src/`; full configurations and trained artifacts are packaged under `code/business_entity_resolution/`. The packaged README contains reproduction commands. Primary entry points: `eda.py`, `io_utils.py`, `tfidf_retriever.py`, `bm25_retriever.py`, `hybrid_retriever.py`, `train.py`, `inference.py`, `validate_submission.py`, and `package_submission.py`.

Run the README commands to generate `output/candidate_pairs.tsv`, `output/matching_results.tsv` and `output/submission.zip`. Test inference rebuilds indexes over test S2/S3 with saved settings. It never queries the train target catalog or expects test labels. Candidate output equals the actual classifier-scored union. The extracted hybrid package was verified to rebuild fresh synthetic test indices and reproduce both TSVs byte-for-byte.

### B. Additional Results

Machine-readable sampled-real retrieval/training reports and exact full-corpus EDA are in `docs/verification/`; portable sampled-real model artifacts are in `models/real_sample/`. Record the full-target Colab run key, input hashes, runtime, hardware, peak memory and leaderboard score before actual submission, and complete the team details above.

No external entity lookup, geocoding, translation API or business enrichment is used. Original code is MIT licensed; third-party libraries retain upstream licenses. AnyAscii is offline transliteration; the model contains no pretrained neural weights.
