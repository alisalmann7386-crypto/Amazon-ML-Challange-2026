# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [To be supplied]  
**Team Members:** [To be supplied]  
**Submission Date:** [Set when submitting]

---

## 1. Executive Summary

Our baseline resolves each Source 1 business against Sources 2 and 3 using a union of character TF-IDF candidates and a trained pair classifier. Grouped out-of-fold predictions select a precision-oriented macro F0.5 threshold, while preserving multi-match and no-match outcomes. Software has been tested on synthetic fixtures; official-data performance has not yet been measured.

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

- **Blocking keys used:** Separate name and address character TF-IDF indices, with 2–5 character n-grams and at most 200,000 features per field, fitted on the combined S2/S3 catalog.
- **Candidate pairs generated:** Not yet measured on official data. The training report records actual pair count and reduction ratio.
- **How you ensured true matches were not lost:** No guarantee is claimed. Union up to 20 positive-cosine neighbors per field (at most 40 unique targets per S1), then measure retrieved true links / all true links. Increase k or add complementary retrieval only after recall and cost analysis. Ground-truth positives are not injected into candidate sets.

Target vectors are reused across all queries within a run. Retrieval uses exact sparse dot products, not ANN search; full-scale time and memory must be profiled. Candidate IDs are sorted deterministically. `candidate_pairs.tsv` is exactly the set the matching model scores.

---

## 4. Matching Model

**Features used:**
- Name features: sequence similarity, token Jaccard, nonempty exact equality.
- Address features: sequence similarity, token Jaccard, nonempty exact equality, numeric-token Jaccard and numeric conflict.
- Other: country agreement, country disagreement, name missingness and address missingness.

**Model type:** StandardScaler followed by L2 logistic regression (`C=1`, `max_iter=1000`, seed 42). No pretrained neural model is used.  
**Threshold selection method:** Maximize macro F0.5 over a fixed threshold grid using grouped OOF probabilities; break score ties in favor of the higher threshold.

All pairs from one S1 remain in the same fold; S1 records sharing labeled targets are connected into a common group. Each fold fits a fresh scaler and classifier. Unsupervised target-catalog text is available to retrieval across folds, but validation S1 labels are excluded from that fold's classifier training. Entities with no candidates remain in the macro metric. The selected OOF score is a tuning statistic; reserve an untouched group holdout for a final performance estimate.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** Not measured on official data. No leaderboard result is claimed.
- **Common false positives (wrong merges):** To measure on real validation data. Planned slices: chains with similar names at different locations, address-number conflicts and singleton false merges.
- **Common false negatives (missed matches):** To measure on real validation data. Planned slices: candidate retrieval misses, transliteration, short names, missing addresses and unseen-country conventions.

Five automated tests passed locally: metric/singleton behavior, Unicode and list validation, empty retrieval, shared-target grouping, and an end-to-end synthetic run with invalid-output rejection. A packaged synthetic run regenerated both output files byte-for-byte. These checks establish software behavior, not challenge accuracy.

---

## 6. Conclusion

The repository provides an executable offline baseline and an auditable submission workflow for zero/one/many entity matching. Candidate recall, singleton precision and independent group holdout quality are the next measurements once official data is available. MiniLM, ANN, boosting and cross-encoder stages remain planned experiments, not implemented or benchmarked components.

---

## Appendix

### A. Code Artefacts

The final ZIP contains source under `code/business_entity_resolution/src/`, pinned `requirements.txt` and a reproduction `README.md`. `core.py` owns I/O, retrieval, features, metric and output checks; `pipeline.py` provides training, prediction, validation and evaluation; `analyze.py` reports data quality; `make_demo.py` generates synthetic fixtures; `package_submission.py` builds the required archive layout.

From the packaged code directory:

```bash
python -m pip install -r requirements.txt
python src/pipeline.py train --data /path/to/dataset/train --k 20 --folds 5
python src/pipeline.py predict --data /path/to/dataset/test --output output
python src/pipeline.py validate --data /path/to/dataset/test --output output
```

The dataset is supplied separately by the organizers. The baseline performs no external entity lookup, geocoding or enrichment. Original code/model are MIT licensed; third-party dependencies retain their licenses. Any optional pretrained model requires a separate MIT/Apache-2.0 and parameter-limit check.

### B. Additional Results

Official data results, runtime, peak memory, hardware, source commit, dataset version and leaderboard scores: not yet recorded. Fill these in from the selected real-data run before submitting. Current reproduction defaults are k=20 per field and 5 folds; document any changed configuration.

---

**Note:** Filled using the user-supplied organizer template. The original blank template and problem README are preserved under `student_resource/`.
