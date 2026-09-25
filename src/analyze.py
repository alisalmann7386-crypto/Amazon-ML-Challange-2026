"""Produce a data-quality report using only supplied TSV records."""
import argparse
import json
from collections import Counter
from pathlib import Path
from core import normalize, read_sources, truth_for

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--split', choices=['train', 'test'], default='train')
    p.add_argument('--output', default='artifacts/data_report.json')
    a = p.parse_args()
    queries, targets = read_sources(a.data, a.split)
    result = {}
    for label, rows in [('source1', queries), ('source2_and_3', targets)]:
        counts = Counter(normalize(r['business_name']) for r in rows if normalize(r['business_name']))
        result[label] = {'rows': len(rows), 'countries': dict(Counter(r['country'] for r in rows)),
            'missing': {f: sum(not r[f].strip() for r in rows) for f in ['business_name', 'business_address', 'country']},
            'repeated_normalized_names': sum(n > 1 for n in counts.values())}
    if a.split == 'train':
        truth = truth_for(Path(a.data)/'train_ground_truth.tsv', queries, targets)
        result['match_count_distribution'] = dict(Counter(map(len, truth.values())))
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
