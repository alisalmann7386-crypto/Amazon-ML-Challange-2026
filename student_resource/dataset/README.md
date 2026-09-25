# Dataset contract

Supply the original files locally; raw competition data is not committed.

| Folder | Files |
| --- | --- |
| `train/` | `train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, `train_ground_truth.tsv` |
| `test/` | `test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv` |

Source column order: `entity_id`, `business_name`, `business_address`, `country`.
Ground-truth column order: `source1_entity_id`, `matched_entity_ids`.
All delimiters are tabs. Match IDs within a cell are comma-separated with no spaces; blank means no match. Prefixes are `S1-`, `S2-`, `S3-`. IDs are always strings.

US and India appear in training; France additionally appears in test according to the provided brief. The implementation accepts arbitrary country strings and never drops a country.

Inspect missingness, country counts and match cardinality:

```bash
python src/analyze.py --data student_resource/dataset/train
```

No real data is bundled. `python src/make_demo.py` generates invented fixtures under `demo_data/` for software testing only.
