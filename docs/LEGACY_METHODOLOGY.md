# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [To be supplied]  
**Team Members:** [To be supplied]  
**Submission Date:** [Set when submitting]

---

## 1. Executive Summary

The Colab baseline resolves each Source 1 business against Sources 2 and 3 using a disk-backed token BM25 candidate index and an incremental logistic pair classifier. The original character TF-IDF workflow remains available for small experiments. A separate grouped calibration split selects a precision-oriented macro F0.5 threshold, while preserving multi-match and no-match outcomes. Software has been tested on synthetic fixtures; official-data performance has not yet been measured.

---

## 2. Methodology

### 2.1 Problem Analysis

The supplied problem statement anticipates abbreviated names, inconsistent legal suffixes, partial addresses, transliteration and missing components. Training covers US and India; test also contains France, so country remains an open-set string. These are requirements from the brief, not findings from an EDA run on the real dataset. `src/analyze.py` will report field missingness, country distribution, repeated normalized names and ground-truth match cardinality when data is supplied.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier  
**Core Innovation:** A reproducible, auditable baseline combining independent name and address retrieval with group-aware threshold selection and exact candidate-set reporting; no novel algorithm is claimed.

Read strict UTF-8 TSV schemas, preserve string IDs, normalize text, retrieve candidates, compute 12 pair features, score with logistic regression, and threshold each pair independently. No forced top-1 match, one-to-one assignment or country whitelist is imposed. Normalization uses Unicode decomposition, accent removal, case folding, ampersand expansion, punctuation removal and whitespace normalization. Missing fields remain empty.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** SQLite FTS5 token BM25 search over normalized names and addresses in the complete combined S2/S3 catalog. Up to 16 unique tokens per field form an OR query.
- **Candidate pairs generated:** Not yet measured on official data. The training report records actual pair count and reduction ratio.
- **How you ensured true matches were not lost:** No guarantee is claimed. Union up to 20 BM25-ranked token matches per field (at most 40 unique targets per S1), then measure retrieved true links / all true links. Increase k or add complementary retrieval only after recall and cost analysis. Ground-truth positives are not injected into candidate sets.

The completed disk index is reused after input SHA256 verification. Retrieval uses a token inverted index, not ANN search; full-scale time and memory must be profiled. Candidate IDs are sorted deterministically. `candidate_pairs.tsv` is exactly the set the matching model scores.

---

## 4. Matching Model

**Features used:**
- Name features: sequence similarity, token Jaccard, nonempty exact equality.
- Address features: sequence similarity, token Jaccard, nonempty exact equality, numeric-token Jaccard and numeric conflict.
- Other: country agreement, country disagreement, name missingness and address missingness.

**Model type:** Incremental `SGDClassifier(loss="log_loss", alpha=1e-4, average=True, random_state=42)`, five epochs by default. All 12 features are already in [0,1]; this workflow does not fit a scaler. No pretrained neural model is used.  
**Threshold selection method:** Maximize macro F0.5 over a fixed threshold grid using probabilities on a dedicated grouped calibration split; break score ties in favor of the higher threshold.

By default, deterministically sample 20,000 S1 entities and split approximately 60/20/20 into training, calibration and holdout groups. All candidates are retrieved against the complete S2/S3 catalog. S1 entities sharing a labeled target within the sample remain in the same group. Train only on the training split, choose the threshold only on calibration, and report the untouched holdout separately. Entities with no candidates remain in the macro metric. Feature shards are cached with a manifest containing input hashes and split IDs. Full-data training accuracy is not claimed.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** Not measured on official data. No leaderboard result is claimed.
- **Common false positives (wrong merges):** To measure on real validation data. Planned slices: chains with similar names at different locations, address-number conflicts and singleton false merges.
- **Common false negatives (missed matches):** To measure on real validation data. Planned slices: candidate retrieval misses, transliteration, short names, missing addresses and unseen-country conventions.

Synthetic automated tests cover metric/singleton behavior, Unicode and list validation, empty retrieval, shared-target grouping, and an end-to-end synthetic run with invalid-output rejection. A packaged synthetic run regenerated both output files byte-for-byte. These checks establish software behavior, not challenge accuracy.

---

## 6. Conclusion

The repository provides an executable offline baseline and an auditable submission workflow for zero/one/many entity matching. Candidate recall, singleton precision and independent group holdout quality are the next measurements once official data is available. MiniLM, ANN, boosting and cross-encoder stages remain planned experiments, not implemented or benchmarked components.

---

## Appendix

### A. Code Artefacts

The final ZIP contains source under `code/business_entity_resolution/src/`, pinned `requirements.txt` and a reproduction `README.md`. `core.py` owns I/O, retrieval, features, metric and output checks; `pipeline.py` provides the original small-data TF-IDF path; `scalable.py` provides disk indexing, batched training, holdout evaluation and streaming prediction; `prepare_data.py` copies TSVs from folders/ZIPs; `analyze.py` reports data quality; `make_demo.py` generates synthetic fixtures; `package_submission.py` builds the required archive layout.

From the packaged code directory:

```bash
python -m pip install -r requirements.txt
python src/scalable.py index --data /path/to/dataset/train --split train --index artifacts/train.sqlite
python src/scalable.py train --index artifacts/train.sqlite --work artifacts/run --sample-size 20000 --k 20 --epochs 5 --seed 42
python src/scalable.py index --data /path/to/dataset/test --split test --index artifacts/test.sqlite
python src/scalable.py predict --index artifacts/test.sqlite --model artifacts/run/model.joblib --output output
```

The dataset is supplied separately by the organizers. The baseline performs no external entity lookup, geocoding or enrichment. Original code/model are MIT licensed; third-party dependencies retain their licenses. Any optional pretrained model requires a separate MIT/Apache-2.0 and parameter-limit check.

### B. Additional Results

Official data results, runtime, peak memory, hardware, source commit, dataset version and leaderboard scores: not yet recorded. Fill these in from the selected real-data run before submitting. Current reproduction defaults are k=20 per field, sample-size=20000, epochs=5 and seed=42; document any changed configuration.

---

**Note:** Filled using the user-supplied organizer template. The original blank template and problem README are preserved under `student_resource/`.
