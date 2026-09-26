"""Preflight the supplied train/test TSVs before EDA or index construction.

The scan is streaming and disk-backed. It proves row counts, records the exact
input hashes, reports recoverable TSV repairs, rejects duplicate IDs, and checks
that every training label references an available S1 and S2/S3 entity.
"""
import argparse
import json
import os
import sqlite3
from pathlib import Path

from core import FIELDS, id_list
from scalable import fingerprint, log, stream


def physical_lines(path):
    count = 0
    last = b''
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            count += block.count(b'\n')
            last = block[-1:]
    return count + bool(last and last != b'\n')


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(temporary, path)


def audit(data, split, output):
    data, output = Path(data), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scan_db = output.with_suffix(output.suffix + '.scan.sqlite')
    if scan_db.exists():
        scan_db.unlink()
    db = sqlite3.connect(scan_db)
    db.execute('PRAGMA journal_mode=OFF'); db.execute('PRAGMA synchronous=OFF'); db.execute('PRAGMA temp_store=FILE')
    db.execute('CREATE TABLE ids(entity_id TEXT PRIMARY KEY, source INTEGER NOT NULL)')
    counts, duplicates, repairs, lines = {}, {}, {}, {}
    try:
        for source in (1, 2, 3):
            name = f'{split}_source{source}.tsv'
            path = data / name
            source_repairs = []
            count = duplicate_count = 0; pending=[]
            for row in stream(path, FIELDS, source_repairs):
                entity_id = row['entity_id']
                if not entity_id.startswith(f'S{source}-') or entity_id.strip() != entity_id or ',' in entity_id:
                    raise ValueError(f'{path}: invalid entity ID {entity_id!r}')
                pending.append((entity_id,source));count += 1
                if len(pending)==50000:
                    before=db.total_changes;db.executemany('INSERT OR IGNORE INTO ids VALUES (?,?)',pending);duplicate_count+=len(pending)-(db.total_changes-before);pending=[]
                if count % 500000 == 0:db.commit();log(f'Integrity scan S{source}: {count:,} rows')
            if pending:
                before=db.total_changes;db.executemany('INSERT OR IGNORE INTO ids VALUES (?,?)',pending);duplicate_count+=len(pending)-(db.total_changes-before)
            db.commit()
            counts[name] = count
            duplicates[name] = duplicate_count
            repairs[name] = source_repairs
            lines[name] = physical_lines(path)

        truth = {'rows': 0, 'total_positive_links': 0, 'unknown_source1': 0, 'missing_source1_labels': 0,
                 'unknown_targets': 0, 'unknown_source1_examples': [], 'unknown_target_examples': []}
        if split == 'train':
            path = data / 'train_ground_truth.tsv'
            db.execute('CREATE TABLE truth_qids(qid TEXT PRIMARY KEY)')
            db.execute('CREATE TABLE truth_targets(tid TEXT PRIMARY KEY)')
            qids=[];target_ids=[]
            for row in stream(path, ['source1_entity_id', 'matched_entity_ids']):
                truth['rows'] += 1
                qid = row['source1_entity_id']
                qids.append((qid,));matches=id_list(row['matched_entity_ids']);truth['total_positive_links']+=len(matches);target_ids.extend((target_id,) for target_id in matches)
                if len(qids)>=50000:
                    db.executemany('INSERT OR IGNORE INTO truth_qids VALUES (?)',qids);qids=[]
                if len(target_ids)>=100000:
                    db.executemany('INSERT OR IGNORE INTO truth_targets VALUES (?)',target_ids);target_ids=[]
                if truth['rows'] % 250000 == 0:db.commit();log(f'Integrity scan ground truth: {truth["rows"]:,} rows')
            if qids:db.executemany('INSERT OR IGNORE INTO truth_qids VALUES (?)',qids)
            if target_ids:db.executemany('INSERT OR IGNORE INTO truth_targets VALUES (?)',target_ids)
            db.commit()
            truth['unknown_source1']=db.execute('SELECT count(*) FROM truth_qids LEFT JOIN ids ON qid=entity_id AND source=1 WHERE entity_id IS NULL').fetchone()[0]
            truth['missing_source1_labels']=db.execute('SELECT count(*) FROM ids LEFT JOIN truth_qids ON entity_id=qid WHERE source=1 AND qid IS NULL').fetchone()[0]
            truth['unknown_targets']=db.execute('SELECT count(*) FROM truth_targets LEFT JOIN ids ON tid=entity_id AND source IN (2,3) WHERE entity_id IS NULL').fetchone()[0]
            truth['unknown_source1_examples']=[row[0] for row in db.execute('SELECT qid FROM truth_qids LEFT JOIN ids ON qid=entity_id AND source=1 WHERE entity_id IS NULL ORDER BY qid LIMIT 25')]
            truth['unknown_target_examples']=[row[0] for row in db.execute('SELECT tid FROM truth_targets LEFT JOIN ids ON tid=entity_id AND source IN (2,3) WHERE entity_id IS NULL ORDER BY tid LIMIT 25')]
            lines['train_ground_truth.tsv'] = physical_lines(path)
            counts['train_ground_truth.tsv'] = truth['rows']

        failures = sum(duplicates.values()) + truth['unknown_source1'] + truth['missing_source1_labels'] + truth['unknown_targets']
        result = {
            'status': 'fail' if failures else ('pass_with_repairs' if any(repairs.values()) else 'pass'),
            'split': split,
            'files': fingerprint(data, split),
            'logical_rows': counts,
            'physical_lines_including_header': lines,
            'duplicate_ids': duplicates,
            'repairs': repairs,
            'ground_truth': truth,
            'note': 'Recoverable rows are preserved in memory during ingestion; original TSV bytes are not modified.'
        }
        _write(output, result)
        if failures:
            raise ValueError(
                f'Integrity check failed: duplicate IDs={sum(duplicates.values())}, '
                f'unknown S1 labels={truth["unknown_source1"]}, missing S1 labels={truth["missing_source1_labels"]}, '
                f'unknown target IDs={truth["unknown_targets"]}. '
                f'See {output}.'
            )
        log(f'Integrity check {result["status"]}: {output}')
        return result
    finally:
        db.close()
        if scan_db.exists():
            scan_db.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--split', choices=['train', 'test'], required=True)
    parser.add_argument('--output', default='artifacts/data_integrity.json')
    args = parser.parse_args()
    audit(args.data, args.split, args.output)
