"""Streaming normalized catalog with strict validation and source fingerprints."""
import argparse
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from core import FIELDS, id_list
from scalable import stream, fingerprint, log
from normalize import normalize_record

VERSION='hybrid-v1'

def config(path='configs/baseline.json'):
    cfg=json.loads(Path(path).read_text())
    if cfg['version']!=VERSION:raise ValueError('Unsupported configuration version')
    return cfg

def dump(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8');os.replace(tmp,path)

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def connect(index):
    p=Path(index)
    db=sqlite3.connect(str(p/'catalog.sqlite' if p.is_dir() else p));db.row_factory=sqlite3.Row
    db.execute('PRAGMA cache_size=-65536');db.execute('PRAGMA temp_store=FILE')
    return db

def batches(iterator,n):
    from itertools import islice
    it=iter(iterator)
    while batch:=list(islice(it,n)):yield batch

def get_rows(db,table,rids):
    out={}
    for part in batches(sorted(set(map(int,rids))),500):
        for row in db.execute(f'SELECT rowid,data FROM {table} WHERE rowid IN ({",".join("?" for _ in part)})',part):
            out[row['rowid']]=json.loads(row['data'])
    return out

def build(data,split,index,cfg):
    out=Path(index);out.mkdir(parents=True,exist_ok=True)
    meta={'version':VERSION,'split':split,'files':fingerprint(data,split),'normalization':cfg['normalization'],'transliteration':cfg['transliteration']}
    manifest=out/'catalog.json'
    if manifest.exists():
        if json.loads(manifest.read_text())!=meta:raise ValueError('Catalog fingerprint/config mismatch; use a new index folder')
        return
    temp=out/'catalog.building.sqlite'
    if temp.exists():temp.unlink()
    db=connect(temp)
    for table in ('targets','queries'):
        db.execute(f'CREATE TABLE {table}(entity_id TEXT UNIQUE NOT NULL,data TEXT,name TEXT,address TEXT,tname TEXT,taddress TEXT,matches TEXT)')
    db.execute('CREATE TABLE links(qid TEXT,tid TEXT,PRIMARY KEY(qid,tid))')
    try:
        for s in (2,3,1):
            table='queries' if s==1 else 'targets';count=0
            for batch in batches(stream(Path(data)/f'{split}_source{s}.tsv',FIELDS),5000):
                values=[]
                for r in batch:
                    eid=r['entity_id']
                    if not eid.startswith(f'S{s}-') or eid.strip()!=eid or ',' in eid:raise ValueError('Invalid source ID')
                    r=normalize_record(r,cfg['normalization'],cfg['transliteration'])
                    values.append((eid,json.dumps(r,ensure_ascii=False),r['business_name_norm'],r['business_address_norm'],r['business_name_transliterated'],r['business_address_transliterated'],None))
                db.executemany(f'INSERT INTO {table} VALUES (?,?,?,?,?,?,?)',values);db.commit();count+=len(values)
                if count%100000==0:log(f'Normalized S{s}: {count:,}')
        if split=='train':
            db.execute('CREATE TEMP TABLE truth_ids(id TEXT PRIMARY KEY)')
            for batch in batches(stream(Path(data)/'train_ground_truth.tsv',['source1_entity_id','matched_entity_ids']),5000):
                for r in batch:
                    qid=r['source1_entity_id'];ids=id_list(r['matched_entity_ids'])
                    db.execute('INSERT INTO truth_ids VALUES (?)',(qid,))
                    if db.execute('UPDATE queries SET matches=? WHERE entity_id=?',(json.dumps(ids),qid)).rowcount!=1:raise ValueError('Unknown S1 truth ID')
                    db.executemany('INSERT INTO links VALUES (?,?)',[(qid,t) for t in ids])
                db.commit()
            if db.execute('SELECT 1 FROM queries WHERE matches IS NULL LIMIT 1').fetchone():raise ValueError('Missing S1 labels')
            if db.execute('SELECT 1 FROM links LEFT JOIN targets ON tid=entity_id WHERE entity_id IS NULL LIMIT 1').fetchone():raise ValueError('Unknown target label')
            db.execute('CREATE INDEX links_target ON links(tid,qid)')
        db.commit()
    finally:db.close()
    os.replace(temp,out/'catalog.sqlite');dump(manifest,meta);log('Normalized catalog complete')

def sample_queries(db,n,seed):
    import heapq
    heap=[]
    for r in db.execute('SELECT rowid,entity_id FROM queries'):
        key=int.from_bytes(hashlib.blake2b(f'{seed}:{r[1]}'.encode(),digest_size=8).digest(),'big')
        item=(-key,r[0])
        if len(heap)<n:heapq.heappush(heap,item)
        elif item>heap[0]:heapq.heapreplace(heap,item)
    rows=get_rows(db,'queries',[rid for _,rid in heap]);result=[]
    for _,rid in sorted(heap,reverse=True):
        q=rows[rid];q['matches']=json.loads(db.execute('SELECT matches FROM queries WHERE rowid=?',(rid,)).fetchone()[0]);result.append(q)
    return result

def linked_groups(db,queries):
    # Traverse full ground-truth graph, including unsampled intermediary S1s.
    parent={q['entity_id']:q['entity_id'] for q in queries};owner={};seen_global=set()
    for q in queries:
        start=q['entity_id']
        if start in owner:parent[start]=owner[start];continue
        todo={start};members=set()
        while todo:
            members.update(todo);tids=set()
            for part in batches(todo,400):
                tids.update(r[0] for r in db.execute(f'SELECT tid FROM links WHERE qid IN ({",".join("?" for _ in part)})',part))
            neighbors=set()
            for part in batches(tids,400):
                neighbors.update(r[0] for r in db.execute(f'SELECT qid FROM links WHERE tid IN ({",".join("?" for _ in part)})',part))
            todo=neighbors-members
        for m in members:
            if m in parent:owner[m]=start;parent[m]=start
    return [parent[q['entity_id']] for q in queries]

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--split',choices=['train','test'],required=True);p.add_argument('--index',required=True);p.add_argument('--config',default='configs/baseline.json');a=p.parse_args();build(a.data,a.split,a.index,config(a.config))
