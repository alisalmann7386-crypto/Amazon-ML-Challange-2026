# Submission logbook

No official submission has been created or uploaded.

After validating real predictions, run:

```bash
python src/package_submission.py --team YOUR_TEAM --test-dir student_resource/dataset/test
```

The ZIP contains `output/`, `code/business_entity_resolution/src/`, a reproduction README, pinned requirements and the working methodology document. The methodology uses the supplied organizer template; complete team details and measured results before final submission. Generated ZIPs remain local and are ignored by Git.

| Run | Source commit | Data version | Configuration | Holdout F0.5 | Public score | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| — | — | — | — | — | — | No real-data run yet |

## Primary hybrid workflow

Use `python src/package_submission.py --team YOUR_TEAM --hybrid-index artifacts/index_test --model-dir artifacts/final_model --output output` to create `output/submission.zip` after validation. See the root README for training/inference. The older commands above remain for the legacy CPU baseline.
