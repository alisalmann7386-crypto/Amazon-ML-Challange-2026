# Real-data result status

## Current status: rerun required

Do not use the previously committed full-data EDA as the current dataset truth. It reported **5,285,603 Source-3 rows**, while the currently supplied Source-3 file was independently counted as **5,274,632 data rows** in Drive and in the prepared local copy. The copy itself lost zero rows; the two numbers describe different file contents or an earlier incorrect scan.

Before publishing new results, run:

```bash
python src/data_integrity.py --data dataset/train --split train --output artifacts/data_integrity.json
```

The command records hashes, logical rows, physical lines, recoverable repairs, duplicate IDs and missing ground-truth IDs. A completed report, not a partial `.building.sqlite` index, is the source of truth.

## Historical sampled-real engineering run

The repository contains an earlier deterministic run using:

- 2,000 real Source-1 records;
- a reduced, label-complete catalog of 206,896 targets;
- 6,896 labeled targets plus 100,000 hashed distractors per target source.

Retrieval did not receive true IDs. These values only show that the software path worked on real records; they are **not full-catalog or competition performance**.

| Historical model | Calibration threshold | Calibration macro F0.5 | Holdout macro F0.5 |
| --- | ---: | ---: | ---: |
| LightGBM | 0.580 | 0.9793 | 0.9839 |
| SGD reference | 0.775 | 0.9630 | 0.9720 |

Those artifacts use the older uncapped candidate-union configuration. They are not the final `hybrid-v2` K=20 model and must not be reused for a final submission.

## Results required from the new run

After integrity passes, save and report:

- exact row counts and current file hashes;
- global K=20 and K=50 candidate recall and cost;
- selected K and the reason;
- LightGBM and SGD calibration scores;
- calibration-selected thresholds;
- untouched grouped-holdout macro F0.5, precision, recall and singleton accuracy;
- exact model and metric artifact paths.

Machine-generated run outputs under `artifacts/` remain outside Git unless they are small, intentionally selected, and clearly labelled with their data scope.
