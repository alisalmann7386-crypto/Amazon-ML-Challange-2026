"""Disk-backed CPU baseline: SQLite FTS5 candidates and incremental logistic training.

No neural weights, GPU, or external business lookup required. Query sampling limits
training cost; retrieval always searches the complete supplied S2/S3 catalog.
"""
import argparse
import csv
import hashlib
import heapq
import json
import os
import sqlite3
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GroupShuffleSplit

from core import FIELDS, features, id_list, normalize, macro_f05
from pipeline import groups_for

VERSION = 1

def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)

def stream(path, fields):
    with open(path, encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        if reader.fieldnames != fields:
            raise ValueError(f'{path}: expected {fields}, got {reader.fieldnames}')
        for n, row in enumerate(reader, 2):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f'{path}:{n}: malformed TSV')
            yield row

def fingerprint(directory, split):
    names = [f'{split}_source{i}.tsv' for i in (1,2,3)]
    if split == 'train':
        names.append('train_ground_truth.tsv')
    result = {}
    for name in names:
        p = Path(directory)/name
        h = hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda: f.read(8*1024*1024), b''):
                h.update(block)
        result[name] = {'sha256': h.hexdigest(), 'bytes': p.stat().st_size}
    return result

def connect(path):
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    con.execute('PRAGMA cache_size=-65536')
    con.execute('PRAGMA temp_store=FILE')
    return con

def build(args):
    final = Path(args.index)
    final.parent.mkdir(parents=True, exist_ok=True)
    log('Hashing input files for cache identity')
    files = fingerprint(args.data, args.split)
    if final.exists():
        with connect(final) as db:
            meta = json.loads(db.execute('SELECT value FROM metadata').fetchone()[0])
        if meta == {'version': VERSION, 'split': args.split, 'files': files}:
            log('Reusing completed index with matching input hashes'); return
        raise ValueError('Index belongs to different inputs/version. Choose a new --index path.')
    temp = final.with_suffix(final.suffix+'.building')
    if temp.exists():
        temp.unlink()  # Only an incomplete build; completed indices are never overwritten.
    db = connect(temp)
    try:
        db.executescript('''
            CREATE TABLE targets(entity_id TEXT PRIMARY KEY, business_name TEXT,
              business_address TEXT, country TEXT, nname TEXT, naddress TEXT);
            CREATE TABLE queries(entity_id TEXT PRIMARY KEY, business_name TEXT,
              business_address TEXT, country TEXT, matches TEXT);
            CREATE TABLE metadata(value TEXT);
        ''')
        for source in (2,3,1):
            n = 0
            batch = []
            for row in stream(Path(args.data)/f'{args.split}_source{source}.tsv', FIELDS):
                eid = row['entity_id']
                if not eid.startswith(f'S{source}-') or ',' in eid or eid.strip() != eid:
                    raise ValueError(f'Invalid entity ID: {eid}')
                values = tuple(row[x] for x in FIELDS)
                batch.append(values+(None,) if source == 1 else values+(normalize(row['business_name']),normalize(row['business_address'])))
                n += 1
                if len(batch) == 10000:
                    insert(db, source, batch); batch=[]
                    if n % 100000 == 0:
                        log(f'Source {source}: {n:,} records imported')
            insert(db, source, batch)
            log(f'Source {source}: finished {n:,} records')
        if args.split == 'train':
            db.execute('CREATE TEMP TABLE seen_truth(id TEXT PRIMARY KEY)')
            for n,row in enumerate(stream(Path(args.data)/'train_ground_truth.tsv', ['source1_entity_id','matched_entity_ids']),1):
                qid=row['source1_entity_id']
                db.execute('INSERT INTO seen_truth VALUES (?)',(qid,))
                matches=id_list(row['matched_entity_ids'])
                for mid in matches:
                    if not db.execute('SELECT 1 FROM targets WHERE entity_id=?',(mid,)).fetchone():
                        raise ValueError(f'Ground truth references absent target: {mid}')
                changed=db.execute('UPDATE queries SET matches=? WHERE entity_id=?',(json.dumps(matches),qid)).rowcount
                if changed != 1:
                    raise ValueError(f'Unknown S1 in ground truth: {qid}')
                if n % 100000 == 0:
                    db.commit();log(f'Ground truth: {n:,} rows checked')
            if db.execute('SELECT 1 FROM queries WHERE matches IS NULL LIMIT 1').fetchone():
                raise ValueError('Missing ground truth for an S1 entity')
        log('Building full-catalog token search index; this can take substantial time')
        db.execute("CREATE VIRTUAL TABLE search USING fts5(nname,naddress,content='targets',content_rowid='rowid',tokenize='unicode61 remove_diacritics 2')")
        db.execute("INSERT INTO search(search) VALUES('rebuild')")
        db.execute('INSERT INTO metadata VALUES (?)',(json.dumps({'version':VERSION,'split':args.split,'files':files}),))
        db.commit()
    finally:
        db.close()
    os.replace(temp,final)
    log(f'Complete index: {final} ({final.stat().st_size/1e9:.2f} GB)')

def insert(db, source, batch):
    sql='INSERT INTO queries VALUES (?,?,?,?,?)' if source==1 else 'INSERT INTO targets VALUES (?,?,?,?,?,?)'
    db.executemany(sql,batch);db.commit()

def retrieve(db, query, k):
    if k < 1:
        raise ValueError('k must be positive')
    ids=set()
    for field,normalized in [('business_name','nname'),('business_address','naddress')]:
        # Token OR broadens recall. Quote every term; raw user text is never FTS syntax.
        tokens=list(dict.fromkeys(normalize(query[field]).split()))[:16]
        if not tokens:
            continue
        expression=normalized+' : ('+' OR '.join('"'+t.replace('"','""')+'"' for t in tokens)+')'
        for row in db.execute('SELECT rowid FROM search WHERE search MATCH ? ORDER BY rank LIMIT ?', (expression,k)):
            ids.add(row[0])
    result=[]
    for rid in ids:
        result.append(dict(db.execute('SELECT * FROM targets WHERE rowid=?',(rid,)).fetchone()))
    return sorted(result,key=lambda row: row['entity_id'])

def sample_queries(db, count, seed):
    if count < 10:
        raise ValueError('--sample-size must be at least 10')
    heap=[]
    for row in db.execute('SELECT entity_id FROM queries'):
        eid=row[0]
        score=int.from_bytes(hashlib.blake2b(f'{seed}:{eid}'.encode(),digest_size=8).digest(),'big')
        item=(-score,eid)
        if len(heap)<count: heapq.heappush(heap,item)
        elif item>heap[0]: heapq.heapreplace(heap,item)
    return [dict(db.execute('SELECT * FROM queries WHERE entity_id=?',(eid,)).fetchone()) for _,eid in sorted(heap,reverse=True)]

def cached_pairs(db, queries, folder, k, batch_queries=250):
    folder.mkdir(parents=True,exist_ok=True)
    for start in range(0,len(queries),batch_queries):
        path=folder/f'{start:09d}.npz'
        if path.exists():
            continue
        X=[];y=[];qi=[];target_ids=[]
        for i in range(start,min(start+batch_queries,len(queries))):
            q=queries[i];truth=set(json.loads(q['matches']))
            for target in retrieve(db,q,k):
                X.append(features(q,target));y.append(int(target['entity_id'] in truth));qi.append(i);target_ids.append(target['entity_id'])
        tmp=path.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,X=np.asarray(X,dtype=np.float32).reshape(-1,12),y=np.array(y,dtype=np.int8),qi=np.array(qi,dtype=np.int32),target_ids=np.asarray(target_ids,dtype=str))
        os.replace(tmp,path)
        log(f'{folder.name}: candidates/features {min(start+batch_queries,len(queries)):,}/{len(queries):,}')
    return sorted(folder.glob('*.npz'))

def score_files(estimator, files, queries):
    scores=[];qindices=[];labels=[];targets=[]
    for path in files:
        with np.load(path,allow_pickle=False) as shard:
            if len(shard['y']):
                scores.append(estimator.predict_proba(shard['X'])[:,1]);qindices.append(shard['qi']);labels.append(shard['y']);targets.extend(shard['target_ids'].tolist())
    return (np.concatenate(scores) if scores else np.array([]),np.concatenate(qindices) if qindices else np.array([],dtype=int),np.concatenate(labels) if labels else np.array([],dtype=int),targets)

def metrics(truth_counts, qi, y, scores, threshold):
    pred=np.bincount(qi[scores>=threshold],minlength=len(truth_counts))
    tp=np.bincount(qi[(scores>=threshold)&(y==1)],minlength=len(truth_counts))
    result=np.zeros(len(truth_counts))
    singleton=truth_counts==0
    result[singleton]=(pred[singleton]==0)
    normal=~singleton
    result[normal]=1.25*tp[normal]/(.25*truth_counts[normal]+pred[normal])
    return {'macro_f05':float(result.mean()),'singleton_accuracy':float(result[singleton].mean()) if singleton.any() else None,
            'candidate_micro_recall':float(y.sum()/truth_counts.sum()) if truth_counts.sum() else None,
            'candidate_pairs':len(y),'s1_count':len(truth_counts)}

def train(args):
    if args.epochs < 1 or args.k < 1:
        raise ValueError('epochs and k must be positive')
    work=Path(args.work);work.mkdir(parents=True,exist_ok=True)
    db=connect(args.index)
    meta=json.loads(db.execute('SELECT value FROM metadata').fetchone()[0])
    if meta['split']!='train': raise ValueError('Training needs a train index with labels')
    queries=sample_queries(db,args.sample_size,args.seed)
    truth={q['entity_id']:set(json.loads(q['matches'])) for q in queries}
    groups=groups_for(queries,truth)
    splitter=GroupShuffleSplit(n_splits=1,test_size=.4,random_state=args.seed)
    train_i,other_i=next(splitter.split(queries,groups=groups))
    cal_i,hold_i=next(GroupShuffleSplit(n_splits=1,test_size=.5,random_state=args.seed+1).split(other_i,groups=groups[other_i]))
    partitions={'train':[queries[i] for i in train_i],'calibration':[queries[other_i[i]] for i in cal_i],'holdout':[queries[other_i[i]] for i in hold_i]}
    manifest={'version':VERSION,'index':meta,'k':args.k,'seed':args.seed,'sample_size':args.sample_size,
              'ids':{name:[q['entity_id'] for q in qs] for name,qs in partitions.items()}}
    manifest_path=work/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:
        raise ValueError('Work directory is for another run. Use a new --work directory.')
    manifest_path.write_text(json.dumps(manifest),encoding='utf-8')
    log('Sample split: '+str({name:len(qs) for name,qs in partitions.items()}))
    files={name:cached_pairs(db,qs,work/name,args.k) for name,qs in partitions.items()}
    db.close()
    estimator=SGDClassifier(loss='log_loss',alpha=1e-4,random_state=args.seed,average=True)
    positives=negatives=0
    rng=np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        for index in rng.permutation(len(files['train'])):
            with np.load(files['train'][index],allow_pickle=False) as shard:
                X,y=shard['X'],shard['y']
                if not len(y): continue
                if epoch==0: positives+=int(y.sum());negatives+=int(len(y)-y.sum())
                order=rng.permutation(len(y))
                estimator.partial_fit(X[order],y[order],classes=np.array([0,1]))
        log(f'Training epoch {epoch+1}/{args.epochs} complete')
    if not positives or not negatives:
        raise ValueError('Retrieved training sample must contain both label classes')
    cal_scores,cal_qi,cal_y,_=score_files(estimator,files['calibration'],partitions['calibration'])
    cal_counts=np.array([len(json.loads(q['matches'])) for q in partitions['calibration']])
    _,threshold=max((metrics(cal_counts,cal_qi,cal_y,cal_scores,t)['macro_f05'],float(t)) for t in np.append(np.linspace(0,1,201),1.000001))
    hold_scores,hold_qi,hold_y,hold_targets=score_files(estimator,files['holdout'],partitions['holdout'])
    hold_counts=np.array([len(json.loads(q['matches'])) for q in partitions['holdout']])
    report={'retrieval':'SQLite FTS5 token BM25 union, not TF-IDF or neural', 'sample_size_actual':len(queries),'k_per_field':args.k,'epochs':args.epochs,'seed':args.seed,'threshold':threshold,
            'training_positive_pairs':positives,'training_negative_pairs':negatives,
            'calibration':metrics(cal_counts,cal_qi,cal_y,cal_scores,threshold),
            'holdout':metrics(hold_counts,hold_qi,hold_y,hold_scores,threshold),
            'scope':'Sampled S1 queries; full supplied S2/S3 catalog. Holdout not used for threshold selection.'}
    artifact={'version':VERSION,'estimator':estimator,'threshold':threshold,'k':args.k,'report':report,'input_fingerprints':meta,'seed':args.seed,'epochs':args.epochs,'sample_size':args.sample_size}
    joblib.dump(artifact,work/'model.joblib')
    (work/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (work/'holdout_pairs.tsv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(['source1_entity_id','target_entity_id','label','score','predicted'])
        for qi,tid,y,score in zip(hold_qi,hold_targets,hold_y,hold_scores):
            w.writerow([partitions['holdout'][qi]['entity_id'],tid,int(y),float(score),int(score>=threshold)])
    log(json.dumps(report,indent=2));log(f'Saved {work}/model.joblib')

def predict(args):
    artifact=joblib.load(args.model)  # Only load trusted artifacts.
    if artifact.get('version')!=VERSION: raise ValueError('Model version mismatch')
    db=connect(args.index)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    # Stage outputs; a failed run never leaves partial files under final names.
    with (out/'matching_results.tsv.tmp').open('w',encoding='utf-8',newline='') as m, (out/'candidate_pairs.tsv.tmp').open('w',encoding='utf-8',newline='') as c:
        mw,cw=csv.writer(m,delimiter='\t',lineterminator='\n'),csv.writer(c,delimiter='\t',lineterminator='\n')
        mw.writerow(['source1_entity_id','matched_entity_ids']);cw.writerow(['source1_entity_id','candidate_entity_ids'])
        for n,row in enumerate(db.execute('SELECT * FROM queries ORDER BY entity_id'),1):
            q=dict(row);targets=retrieve(db,q,artifact['k'])
            scores=artifact['estimator'].predict_proba(np.asarray([features(q,t) for t in targets],dtype=np.float32))[:,1] if targets else []
            candidates=[t['entity_id'] for t in targets]
            matches=[t['entity_id'] for t,score in zip(targets,scores) if score>=artifact['threshold']]
            mw.writerow([q['entity_id'],','.join(matches)]);cw.writerow([q['entity_id'],','.join(candidates)])
            if n%10000==0: log(f'Predicted {n:,} S1 entities')
    db.close()
    for name in ['matching_results.tsv','candidate_pairs.tsv']:
        os.replace(out/(name+'.tmp'),out/name)
    validate_index(args.index,out)

def validate_index(index,output):
    db=connect(index)
    db.execute('CREATE TEMP TABLE seen(id TEXT PRIMARY KEY)')
    matches=stream(Path(output)/'matching_results.tsv',['source1_entity_id','matched_entity_ids'])
    candidates=stream(Path(output)/'candidate_pairs.tsv',['source1_entity_id','candidate_entity_ids'])
    from itertools import zip_longest
    count=0
    for m,c in zip_longest(matches,candidates):
        if m is None or c is None or m['source1_entity_id']!=c['source1_entity_id']:
            raise ValueError('Outputs must have identical S1 rows in identical order')
        qid=m['source1_entity_id'];db.execute('INSERT INTO seen VALUES (?)',(qid,))
        if not db.execute('SELECT 1 FROM queries WHERE entity_id=?',(qid,)).fetchone():
            raise ValueError('Unknown S1 output ID')
        mids,cids=set(id_list(m['matched_entity_ids'])),set(id_list(c['candidate_entity_ids']))
        if not mids<=cids: raise ValueError('Matches outside candidate list')
        for cid in cids:
            if not db.execute('SELECT 1 FROM targets WHERE entity_id=?',(cid,)).fetchone():
                raise ValueError('Unknown target output ID')
        count+=1
    if count!=db.execute('SELECT count(*) FROM queries').fetchone()[0]:
        raise ValueError('Missing S1 rows')
    db.close();log(f'PASS: {count:,} output rows checked against index')

def main():
    p=argparse.ArgumentParser(description=__doc__);subs=p.add_subparsers(dest='command',required=True)
    s=subs.add_parser('index');s.add_argument('--data',required=True);s.add_argument('--split',choices=['train','test'],required=True);s.add_argument('--index',required=True);s.set_defaults(run=build)
    s=subs.add_parser('train');s.add_argument('--index',required=True);s.add_argument('--work',default='artifacts/scalable');s.add_argument('--sample-size',type=int,default=20000);s.add_argument('--k',type=int,default=20);s.add_argument('--seed',type=int,default=42);s.add_argument('--epochs',type=int,default=5);s.set_defaults(run=train)
    s=subs.add_parser('predict');s.add_argument('--index',required=True);s.add_argument('--model',required=True);s.add_argument('--output',default='output');s.set_defaults(run=predict)
    s=subs.add_parser('validate');s.add_argument('--index',required=True);s.add_argument('--output',default='output');s.set_defaults(run=lambda a:validate_index(a.index,a.output))
    a=p.parse_args();a.run(a)

if __name__=='__main__': main()
