"""Hybrid features -> LightGBM + SGD reference -> calibration -> untouched holdout."""
import argparse
import csv
import json
import os
import platform
import time
from pathlib import Path
import joblib
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from io_utils import config,dump,digest,batches
from splitting import split
from hybrid_retriever import HybridRetriever
from features import FeatureBuilder,FEATURE_NAMES
from metrics import threshold_search,breakdown
from retrieval_evaluation import compare
from error_analysis import export
from scalable import log

FORMAT='hybrid-v1'

def probabilities(model,X):
    if len(X)==0:return np.array([])
    if isinstance(model,lgb.Booster):return model.predict(X,num_threads=2)
    return model.predict_proba(X)[:,1]

def make_shards(retriever,builder,queries,folder,cfg):
    folder.mkdir(parents=True,exist_ok=True);offset=0;total=0;start=time.perf_counter()
    for qs in batches(queries,cfg['feature_shard_queries']):
        path=folder/f'{offset:09d}.npz';audit=folder/f'{offset:09d}.json'
        if path.exists() and audit.exists():offset+=len(qs);continue
        Xs=[];ys=[];qis=[];tids=[];records=[];hard=normal=positives=0
        for sub in batches(qs,cfg['query_batch_size']):
            cs=retriever.query_batch(sub);Xs.append(builder.batch(sub,cs))
            for q,items in zip(sub,cs):
                i=offset+len(records);actual=set(q['matches'])
                records.append(items)
                for c in items:
                    tid=c['target']['entity_id'];y=int(tid in actual);ys.append(y);qis.append(i);tids.append(tid)
                    if y:positives+=1
                    elif min(c['ranks'].values())<=5:hard+=1
                    else:normal+=1
        X=np.concatenate(Xs) if Xs else np.zeros((0,len(FEATURE_NAMES)),np.float32)
        tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,X=X,y=np.asarray(ys,np.int8),qi=np.asarray(qis,np.int32),tids=np.asarray(tids,dtype=str),counts=np.array([positives,hard,normal]));os.replace(tmp,path)
        dump(audit,records);offset+=len(qs);total+=len(ys);log(f'{folder.name} features: {offset}/{len(queries)} S1')
    paths=sorted(p for p in folder.glob('*.npz') if p.stem.isdigit())
    log(f'New feature pairs/sec: {total/max(time.perf_counter()-start,1e-9):.1f}');return paths

def matrix(paths,folder):
    lengths=[]
    for p in paths:
        with np.load(p,allow_pickle=False) as d:lengths.append(len(d['y']))
    n=sum(lengths)
    if not n:raise ValueError('No retrieved pairs for classifier training')
    X=np.lib.format.open_memmap(folder/'train_X.npy',mode='w+',dtype=np.float32,shape=(n,len(FEATURE_NAMES)))
    y=np.lib.format.open_memmap(folder/'train_y.npy',mode='w+',dtype=np.int8,shape=(n,));pos=0
    for p,size in zip(paths,lengths):
        with np.load(p,allow_pickle=False) as d:X[pos:pos+size]=d['X'];y[pos:pos+size]=d['y'];pos+=size
    X.flush();y.flush();return X,y

def scored(model,paths,scaler=None):
    qi=[];y=[];scores=[];tids=[]
    for p in paths:
        with np.load(p,allow_pickle=False) as d:
            X=d['X'];scores.append(probabilities(model,scaler.transform(X) if scaler is not None and len(X) else X));qi.append(d['qi']);y.append(d['y']);tids.extend(d['tids'].tolist())
    return np.concatenate(qi),np.concatenate(y),np.concatenate(scores),tids

def decisions(queries,qi,tids,scores,t):
    pred={q['entity_id']:set() for q in queries};cand={k:set() for k in pred}
    for i,tid,score in zip(qi,tids,scores):
        key=queries[i]['entity_id'];cand[key].add(tid)
        if score>=t:pred[key].add(tid)
    return pred,cand

def run(index,cfg,work,final,retrieval_output,error_output,skip_retrieval=False):
    threshold_cfg=cfg['thresholds']
    if threshold_cfg['steps']<2 or not 0<=threshold_cfg['start']<=threshold_cfg['stop']<=1:
        raise ValueError('Threshold grid requires steps >= 2 and 0 <= start <= stop <= 1')
    work,final=Path(work),Path(final);work.mkdir(parents=True,exist_ok=True);final.mkdir(parents=True,exist_ok=True)
    parts=split(index,cfg)
    manifest={'format':FORMAT,'catalog':json.loads((Path(index)/'catalog.json').read_text()),'config':cfg,'feature_names':FEATURE_NAMES,'split_ids':{k:[q['entity_id'] for q in v] for k,v in parts.items()}}
    mp=work/'manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:raise ValueError('Incompatible cache: use a new work directory')
    dump(mp,manifest);dump(work/'splits.json',parts)
    if not skip_retrieval:compare(index,cfg,retrieval_output,parts['train'])
    r=HybridRetriever(index,cfg);builder=FeatureBuilder(r)
    shards={name:make_shards(r,builder,qs,work/name,cfg) for name,qs in parts.items()}
    X,y=matrix(shards['train'],work)
    if len(np.unique(y))!=2:raise ValueError('Training pairs need both classes')
    counts=np.zeros(3,dtype=np.int64)
    for p in shards['train']:
        with np.load(p,allow_pickle=False) as d:counts+=d['counts']
    log(f'Training matrix: {X.shape}, float32 bytes={X.nbytes}; positives/hard/normal={counts.tolist()}')
    try:
        import torch
        cuda=bool(torch.cuda.is_available());gpu=torch.cuda.get_device_name(0) if cuda else None
    except ImportError:cuda='not probed (torch optional)';gpu=None
    log(f'CUDA available: {cuda}; GPU name: {gpu}; TF-IDF backend: CPU sparse; BM25 backend: CPU')
    settings=dict(cfg['model']);rounds=settings.pop('n_estimators');jobs=settings.pop('n_jobs',2)
    params={**settings,'objective':'binary','verbosity':-1,'seed':cfg['seed'],'num_threads':jobs,'deterministic':True,'force_col_wise':True}
    training=lgb.Dataset(X,label=y,feature_name=FEATURE_NAMES,free_raw_data=False)
    try:model=lgb.train(params,training,num_boost_round=rounds)
    except lgb.basic.LightGBMError:
        if params.get('device_type','cpu')=='cpu':raise
        log('Requested GPU backend unavailable; retrying LightGBM on CPU');params['device_type']='cpu';model=lgb.train(params,training,num_boost_round=rounds)
    log(f'LightGBM backend: {params.get("device_type","cpu")}')
    scaler=StandardScaler()
    for p in shards['train']:
        with np.load(p,allow_pickle=False) as d:
            if len(d['y']):scaler.partial_fit(d['X'])
    reference=SGDClassifier(loss='log_loss',alpha=1e-4,average=True,random_state=cfg['seed']);rng=np.random.default_rng(cfg['seed'])
    for epoch in range(cfg['reference_epochs']):
        for pidx in rng.permutation(len(shards['train'])):
            with np.load(shards['train'][pidx],allow_pickle=False) as d:
                if len(d['y']):
                    order=rng.permutation(len(d['y']));reference.partial_fit(scaler.transform(d['X'][order]),d['y'][order],classes=np.array([0,1]))
    grid=cfg['thresholds'];grid=np.append(np.linspace(grid['start'],grid['stop'],grid['steps']),1.000001)
    reports={};primary_pred=None
    for name,est,scale in [('lightgbm',model,None),('sgd',reference,scaler)]:
        qi,yy,ss,tids=scored(est,shards['calibration'],scale)
        threshold,trace=threshold_search(parts['calibration'],qi,yy,ss,grid)
        pred,cand=decisions(parts['calibration'],qi,tids,ss,threshold);cal=breakdown(parts['calibration'],pred,cand)
        qi,yy,ss,tids=scored(est,shards['holdout'],scale);pred,cand=decisions(parts['holdout'],qi,tids,ss,threshold)
        reports[name]={'threshold':threshold,'calibration':cal,'holdout':breakdown(parts['holdout'],pred,cand)}
        dump(work/(name+'_threshold_grid.json'),trace)
        if name=='lightgbm':primary_pred=pred
    artifact={'format':FORMAT,'model':model,'threshold':reports['lightgbm']['threshold'],'config':cfg,'feature_names':FEATURE_NAMES,'catalog_training_fingerprint':digest(manifest['catalog'])}
    joblib.dump(artifact,final/'model.joblib');joblib.dump({'model':reference,'scaler':scaler,'threshold':reports['sgd']['threshold'],'feature_names':FEATURE_NAMES},final/'reference.joblib')
    dump(final/'threshold.json',{'threshold':artifact['threshold'],'selected_on':'calibration','tie_break':'higher threshold'})
    dump(final/'feature_names.json',FEATURE_NAMES);dump(final/'model_config.json',cfg);dump(final/'normalization_config.json',cfg['normalization']);dump(final/'transliteration_config.json',cfg['transliteration']);dump(final/'manifest.json',manifest)
    report={'primary_model':'lightgbm (fixed before holdout)','training_counts':{'positive':int(counts[0]),'hard_negative':int(counts[1]),'normal_negative':int(counts[2]),'positive_negative_ratio':float(counts[0]/max(1,counts[1]+counts[2]))},'splits':{k:len(v) for k,v in parts.items()},'models':reports,'feature_names':FEATURE_NAMES,'scope':cfg.get('evaluation_scope','Sampled S1; complete supplied target catalog.')+' All retrieved pairs retained. Holdout never used for fitting or threshold selection.','timing':r.timing}
    dump(work/'metrics.json',report);dump(final/'train_metrics.json',report)
    with (final/'feature_importance.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['feature','gain','split']);w.writerows(zip(FEATURE_NAMES,model.feature_importance('gain'),model.feature_importance('split')))
    def holdout_candidates():
        for p in shards['holdout']:
            yield from json.loads(p.with_suffix('.json').read_text())
    export(r,parts['holdout'],holdout_candidates(),primary_pred,error_output);r.close()
    print(json.dumps({k:v['holdout']['all'] for k,v in reports.items()},indent=2));return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--config',default='configs/baseline.json');p.add_argument('--work',default='artifacts/run');p.add_argument('--final',default='artifacts/final_model');p.add_argument('--retrieval-output',default='artifacts/retrieval_comparison');p.add_argument('--error-output',default='artifacts/error_analysis');p.add_argument('--skip-retrieval-eval',action='store_true');a=p.parse_args();run(a.index,config(a.config),a.work,a.final,a.retrieval_output,a.error_output,a.skip_retrieval_eval)
