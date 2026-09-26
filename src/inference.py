"""Fresh test-catalog indexes, saved configs, resumable output shards; no test labels."""
import argparse
import csv
import json
import os
from pathlib import Path
import joblib
from io_utils import build,connect,dump,digest,batches
from tfidf_retriever import CharTfidfRetriever
from bm25_retriever import BM25Retriever
from hybrid_retriever import HybridRetriever
from features import FeatureBuilder,FEATURE_NAMES
from train import probabilities,FORMAT
from validate_submission import validate
from metrics import validate_threshold


def run(data,index,model_dir,output):
    model_dir=Path(model_dir);artifact=joblib.load(model_dir/'model.joblib')
    if artifact['format']!=FORMAT or artifact['feature_names']!=FEATURE_NAMES:raise ValueError('Feature schema/version mismatch')
    if json.loads((model_dir/'feature_names.json').read_text())!=FEATURE_NAMES:raise ValueError('Feature-order mismatch')
    threshold=validate_threshold(json.loads((model_dir/'threshold.json').read_text())['threshold'])
    artifact_threshold=validate_threshold(artifact['threshold'])
    if threshold!=artifact_threshold:raise ValueError('Threshold file differs from model artifact')
    cfg=artifact['config'];build(data,'test',index,cfg)
    CharTfidfRetriever(index,cfg).fit();bm=BM25Retriever(index,cfg).fit();bm.close()
    r=HybridRetriever(index,cfg);features=FeatureBuilder(r);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    folder=out/'inference_shards';folder.mkdir(exist_ok=True)
    import hashlib
    h=hashlib.sha256((model_dir/'model.joblib').read_bytes()).hexdigest()
    identity={'model_sha256':h,'test_catalog':json.loads((Path(index)/'catalog.json').read_text()),'threshold':threshold}
    marker=folder/'manifest.json'
    if marker.exists() and json.loads(marker.read_text())!=identity:raise ValueError('Prediction cache mismatch; choose a new output folder')
    dump(marker,identity);paths=[]
    for batch in batches(r.db.execute('SELECT rowid,data FROM queries ORDER BY rowid'),cfg['query_batch_size']):
        path=folder/f'{batch[0][0]:012d}.json';paths.append(path)
        if path.exists():continue
        qs=[json.loads(row['data']) for row in batch];cs=r.query_batch(qs);X=features.batch(qs,cs);scores=probabilities(artifact['model'],X);offset=0;rows=[]
        for q,items in zip(qs,cs):
            probabilities_=scores[offset:offset+len(items)];offset+=len(items)
            rows.append({'qid':q['entity_id'],'candidates':[c['target']['entity_id'] for c in items],'matches':[c['target']['entity_id'] for c,s in zip(items,probabilities_) if s>=threshold]})
        dump(path,rows)
    for file,key,column in [('matching_results.tsv','matches','matched_entity_ids'),('candidate_pairs.tsv','candidates','candidate_entity_ids')]:
        temp=out/(file+'.tmp')
        with temp.open('w',encoding='utf-8',newline='') as f:
            w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['source1_entity_id',column])
            for path in paths:
                for row in json.loads(path.read_text()):w.writerow([row['qid'],','.join(sorted(row[key]))])
        os.replace(temp,out/file)
    r.close();validate(index,out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--index',default='artifacts/index_test');p.add_argument('--model-dir',default='artifacts/final_model');p.add_argument('--output',default='output');a=p.parse_args();run(a.data,a.index,a.model_dir,a.output)
