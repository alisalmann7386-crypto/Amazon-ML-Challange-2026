# Incremental repository audit

Starting commit: `7d3f4b6244b9b946655daa9a58be3805f62e5e54`.

| Component | Finding | Action |
| --- | --- | --- |
| `core.py` | Strict schemas, ID lists, macro F0.5 and singleton behavior work. Original NFKD normalization removes Indic combining marks; original retrieval is in-memory. | Preserve legacy API/tests; use new `normalize.py` in hybrid path. Enable strict CSV quoting checks. |
| `scalable.py` | Disk catalog, fingerprints, feature caching, sampled training and independent holdout work. BM25-only; per-candidate target lookup; no transliteration. | Preserve CPU reference; new catalog and hybrid modules bulk-fetch targets and add channel metadata. |
| `pipeline.py` | Useful small-data TF-IDF/OOF reference; cannot serve million-row training unchanged. | Preserve and label legacy; reuse metric conventions. |
| `analyze.py` | Basic in-memory EDA; missing script/length/pair-distribution analysis. | Preserve; add streaming `eda.py` with disk-backed exact uniqueness and token counts. |
| `prepare_data.py` | Canonical TSV/ZIP handling and duplicate-copy protection work. | Add SHA-256, byte and physical-line manifests so copying cannot silently reduce rows. |
| `package_submission.py` | Existing package shape and legacy validation work. | Add hybrid mode, full model config artifacts, and fresh-index reproduction instructions. |
| notebooks | Legacy BM25-only notebook works structurally, but doesn't perform requested hybrid training. | Keep legacy guide; replace main notebook flow and add EDA notebook. |
| dependencies/docs | No LightGBM, RapidFuzz, sparse top-N or offline transliteration. | Pin and test dependencies; preserve old guides under `docs/LEGACY_*`. |

The user's removal of a sentence from `COLAB.md` was preserved before moving its prior contents to the legacy guide. No source files or working baselines were deleted.

New primary workflow: verified copy → disk-backed integrity audit → NFKC/mark-preserving normalization with recorded recoverable repairs → optional AnyAscii fields → sharded character TF-IDF + SQLite BM25 → reciprocal-rank merge → global K=20 cap → fixed-order features → LightGBM and SGD reference → calibration-only threshold → untouched holdout → artifacts → fresh test indexes → exact baseline and streamed/resumable hybrid inference → strict validation → package.
