"""Disk-backed strict validation against the normalized target catalog."""
import argparse
from itertools import zip_longest
from pathlib import Path
from core import id_list
from io_utils import connect,batches
from scalable import stream

def validate(index,output):
    db=connect(index);db.execute('CREATE TEMP TABLE seen(id TEXT PRIMARY KEY)');count=0
    m=stream(Path(output)/'matching_results.tsv',['source1_entity_id','matched_entity_ids']);c=stream(Path(output)/'candidate_pairs.tsv',['source1_entity_id','candidate_entity_ids'])
    for mr,cr in zip_longest(m,c):
        if mr is None or cr is None or mr['source1_entity_id']!=cr['source1_entity_id']:raise ValueError('Outputs must have identical S1 order')
        qid=mr['source1_entity_id'];db.execute('INSERT INTO seen VALUES (?)',(qid,))
        if not db.execute('SELECT 1 FROM queries WHERE entity_id=?',(qid,)).fetchone():raise ValueError('Unknown S1')
        mids,cids=set(id_list(mr['matched_entity_ids'])),set(id_list(cr['candidate_entity_ids']))
        if not mids<=cids:raise ValueError('Matches must be subset of candidates')
        for part in batches(cids,500):
            n=db.execute(f'SELECT count(*) FROM targets WHERE entity_id IN ({",".join("?" for _ in part)})',part).fetchone()[0]
            if n!=len(part):raise ValueError('Unknown target IDs')
        count+=1
    if count!=db.execute('SELECT count(*) FROM queries').fetchone()[0]:raise ValueError('Missing S1 rows')
    db.close();print(f'PASS: {count} S1 rows');return count

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--output',default='output');a=p.parse_args();validate(a.index,a.output)
