"""CLI: train using grouped OOF, predict, evaluate, and validate."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from core import (Retriever, features, macro_f05, read_sources, truth_for,
                  validate, write_lists)

def groups_for(queries, truth):
    # Connected S1 entities sharing a labeled target stay in the same fold.
    parent = {q['entity_id']: q['entity_id'] for q in queries}
    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    owner = {}
    for key, ids in truth.items():
        for target in ids:
            if target in owner:
                parent[root(key)] = root(owner[target])
            owner[target] = key
    return np.array([root(q['entity_id']) for q in queries])

def pair_data(queries, targets, k):
    retriever = Retriever(targets, k)
    rows, pairs = [], []
    for qi, q in enumerate(queries):
        for ti in retriever.candidates(q):
            rows.append(features(q, targets[ti]))
            pairs.append((qi, ti))
    return np.asarray(rows, dtype=float).reshape(-1, 12), pairs

def model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000,
                                                              random_state=42))

def predictions(queries, targets, pairs, scores, threshold):
    out = {q['entity_id']: set() for q in queries}
    for (qi, ti), score in zip(pairs, scores):
        if score >= threshold:
            out[queries[qi]['entity_id']].add(targets[ti]['entity_id'])
    return out

def train(args):
    queries, targets = read_sources(args.data, 'train')
    truth = truth_for(Path(args.data) / 'train_ground_truth.tsv', queries, targets)
    X, pairs = pair_data(queries, targets, args.k)
    y = np.array([int(targets[ti]['entity_id'] in truth[queries[qi]['entity_id']]) for qi, ti in pairs])
    if len(set(y)) != 2:
        raise ValueError('Retrieved training pairs need positive and negative examples')
    groups = groups_for(queries, truth)
    nfolds = min(args.folds, len(set(groups)))
    if nfolds < 2:
        raise ValueError('Need at least two independent S1 groups')
    pair_q = np.array([qi for qi, ti in pairs])
    oof = np.full(len(pairs), np.nan)
    for train_q, valid_q in GroupKFold(n_splits=nfolds).split(queries, groups=groups):
        train_mask, valid_mask = np.isin(pair_q, train_q), np.isin(pair_q, valid_q)
        if len(set(y[train_mask])) != 2:
            raise ValueError('A fold has one label class; use more training data or fewer folds')
        fitted = model().fit(X[train_mask], y[train_mask])
        if valid_mask.any():
            oof[valid_mask] = fitted.predict_proba(X[valid_mask])[:, 1]
    if not np.isfinite(oof).all():
        raise ValueError('Incomplete OOF scores')
    # Threshold selection is itself model selection; report on a separate holdout too.
    thresholds = np.append(np.linspace(0, 1, 201), 1.000001)
    evaluations = [(macro_f05(truth, predictions(queries, targets, pairs, oof, t)), float(t)) for t in thresholds]
    best_score, threshold = max(evaluations)
    candidates = predictions(queries, targets, pairs, np.ones(len(pairs)), 0)
    positives = sum(map(len, truth.values()))
    found = sum(len(truth[key] & candidates[key]) for key in truth)
    report = {'oof_threshold_selection_macro_f05': best_score, 'threshold': threshold,
              'candidate_micro_recall': found / positives if positives else None,
              'candidate_reduction_ratio': 1-len(pairs)/(len(queries)*len(targets)) if targets else None,
              's1_records': len(queries), 'target_records': len(targets), 'pairs': len(pairs),
              'folds': nfolds, 'k_per_field': args.k,
              'note': 'OOF threshold-selection score is not an unbiased final holdout estimate.'}
    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'estimator': model().fit(X, y), 'threshold': threshold, 'k': args.k,
                 'report': report}, args.model)
    Path(args.model).with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))

def predict(args):
    queries, targets = read_sources(args.data, 'test')
    artifact = joblib.load(args.model)  # Load only model files you trust.
    X, pairs = pair_data(queries, targets, artifact['k'])
    scores = artifact['estimator'].predict_proba(X)[:, 1] if pairs else np.array([])
    matches = predictions(queries, targets, pairs, scores, artifact['threshold'])
    candidates = predictions(queries, targets, pairs, np.ones(len(pairs)), 0)
    write_lists(Path(args.output)/'matching_results.tsv', queries, matches, 'matched_entity_ids')
    write_lists(Path(args.output)/'candidate_pairs.tsv', queries, candidates, 'candidate_entity_ids')
    validate(args.output, queries, targets)
    print(f'PASS: {len(queries)} S1 rows, {len(pairs)} candidate pairs')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    p = subs.add_parser('train')
    p.add_argument('--data', required=True)
    p.add_argument('--model', default='artifacts/model.joblib')
    p.add_argument('--k', type=int, default=20)
    p.add_argument('--folds', type=int, default=5)
    p.set_defaults(run=train)
    p = subs.add_parser('predict')
    p.add_argument('--data', required=True)
    p.add_argument('--model', default='artifacts/model.joblib')
    p.add_argument('--output', default='output')
    p.set_defaults(run=predict)
    p = subs.add_parser('validate')
    p.add_argument('--data', required=True)
    p.add_argument('--output', default='output')
    p.set_defaults(run=lambda a: (validate(a.output, *read_sources(a.data, 'test')), print('PASS')))
    p = subs.add_parser('evaluate')
    p.add_argument('--data', required=True, help='Train-schema held-out directory including truth')
    p.add_argument('--output', required=True)
    def evaluate(a):
        q, t = read_sources(a.data, 'train')
        truth = truth_for(Path(a.data)/'train_ground_truth.tsv', q, t)
        print(json.dumps({'macro_f05': macro_f05(truth, validate(a.output, q, t))}))
    p.set_defaults(run=evaluate)
    args = parser.parse_args()
    args.run(args)

if __name__ == '__main__':
    main()
