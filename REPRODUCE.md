# Reproduce the business entity resolution baseline

Use Python 3.11+. From this directory:

```bash
python -m pip install -r requirements.txt
python src/pipeline.py train --data /path/to/dataset/train --model artifacts/model.joblib --k 20 --folds 5
python src/pipeline.py predict --data /path/to/dataset/test --model artifacts/model.joblib --output output
python src/pipeline.py validate --data /path/to/dataset/test --output output
```

Training expects `train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, and `train_ground_truth.tsv`. Test expects `test_source1.tsv`, `test_source2.tsv`, and `test_source3.tsv`. Data is supplied separately by the organizers. No external business data is downloaded.

Sources have exactly `entity_id`, `business_name`, `business_address`, `country` columns, tab-separated. Ground truth has `source1_entity_id`, `matched_entity_ids`. Read IDs as strings, including leading zeros. Empty match lists represent singletons.

The default model is character TF-IDF candidate retrieval plus a 12-feature logistic pair classifier. Threshold selection uses grouped OOF macro F0.5. Both output TSVs contain every S1 once. The candidate list is precisely the set passed to the matching model. Exact sparse retrieval may need substantial time and memory at competition scale; neural and ANN extensions are not included.

A model is retrained from the supplied data; the ZIP does not need to carry a model checkpoint. To reproduce a different run, record its `k`, folds, dependency versions and source commit in the methodology. Current packaging assumes the documented default command configuration.
