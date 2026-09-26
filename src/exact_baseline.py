"""Create a fast no-ML baseline using normalized business name + country."""
import argparse
import csv
import json
import os
import time
from pathlib import Path

from io_utils import connect, dump
from validate_submission import validate


def run(index, output):
    started = time.perf_counter()
    db = connect(index)
    # The expression index keeps each query from scanning the full target table.
    db.execute("CREATE INDEX IF NOT EXISTS targets_exact_name_country ON targets(name,json_extract(data,'$.country_norm'))")
    db.commit()
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    result_tmp = output / 'matching_results.tsv.tmp'
    candidate_tmp = output / 'candidate_pairs.tsv.tmp'
    queries = pairs = matched_queries = 0
    with result_tmp.open('w', encoding='utf-8', newline='') as result_file, candidate_tmp.open('w', encoding='utf-8', newline='') as candidate_file:
        results = csv.writer(result_file, delimiter='\t', lineterminator='\n')
        candidates = csv.writer(candidate_file, delimiter='\t', lineterminator='\n')
        results.writerow(['source1_entity_id', 'matched_entity_ids'])
        candidates.writerow(['source1_entity_id', 'candidate_entity_ids'])
        for row in db.execute('SELECT entity_id,data,name FROM queries ORDER BY entity_id'):
            query = json.loads(row['data']); country = query['country_norm']; name = row['name']
            if not name:
                ids = []
            elif country:
                ids = [target[0] for target in db.execute("SELECT entity_id FROM targets WHERE name=? AND json_extract(data,'$.country_norm')=? ORDER BY entity_id", (name, country))]
            else:
                ids = [target[0] for target in db.execute('SELECT entity_id FROM targets WHERE name=? ORDER BY entity_id', (name,))]
            joined = ','.join(ids); results.writerow([row['entity_id'], joined]); candidates.writerow([row['entity_id'], joined])
            queries += 1; pairs += len(ids); matched_queries += bool(ids)
    db.close(); os.replace(result_tmp, output / 'matching_results.tsv'); os.replace(candidate_tmp, output / 'candidate_pairs.tsv')
    validate(index, output)
    report = {'method': 'exact normalized business name + country; name-only fallback when country is missing', 'model': None,
              'queries': queries, 'candidate_pairs': pairs, 'queries_with_candidates': matched_queries,
              'elapsed_seconds': time.perf_counter() - started,
              'note': 'This is a submission-pipeline and leaderboard baseline, not a locally measured test F0.5 score.'}
    dump(output / 'exact_baseline.json', report); print(json.dumps(report, indent=2)); return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', required=True)
    parser.add_argument('--output', default='output/exact_baseline')
    args = parser.parse_args(); run(args.index, args.output)
