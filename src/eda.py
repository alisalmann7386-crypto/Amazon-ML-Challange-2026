"""Streaming EDA with disk-backed uniqueness/token counts; optional pair samples."""
import argparse
import csv
import json
import random
import sqlite3
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
import numpy as np
from core import FIELDS,id_list
from scalable import stream
from normalize import normalize_record
from transliterate import scripts
from similarity import numbers,postal
from io_utils import config,dump,batches,get_rows

def distribution(counter):
    n=sum(counter.values())
    if not n:return {'count':0}
    values=sorted(counter);points={}
    for percentile in [.5,.95]:
        rank=max(1,int(np.ceil(n*percentile)));running=0
        for value in values:
            running+=counter[value]
            if running>=rank:points[str(percentile)]=value;break
    return {'count':n,'min':values[0],'max':values[-1],'mean':sum(k*v for k,v in counter.items())/n,'median':points['0.5'],'p95':points['0.95']}

def source_report(path,cfg,limit):
    with tempfile.TemporaryDirectory() as temp:
        db=sqlite3.connect(str(Path(temp)/'eda.sqlite'));db.execute('CREATE TABLE unique_values(kind TEXT,value TEXT,PRIMARY KEY(kind,value))');db.execute('CREATE TABLE tokens(value TEXT PRIMARY KEY,n INTEGER)')
        countries=Counter();script_counts=Counter();hists={k:Counter() for k in ['name_length','address_length','name_tokens','address_tokens']};flags=Counter();n=duplicates=0
        for batch in batches(stream(path,FIELDS),1000):
            if limit:batch=batch[:max(0,limit-n)]
            if not batch:break
            tokens=Counter()
            for raw in batch:
                n+=1;r=normalize_record(raw,cfg['normalization'],cfg['transliteration'])
                for key,val in [('id',r['entity_id']),('name',r['business_name_raw']),('normalized_name',r['business_name_norm']),('address',r['business_address_raw'])]:
                    changed=db.execute('INSERT OR IGNORE INTO unique_values VALUES (?,?)',(key,val)).rowcount
                    if key=='id' and not changed:duplicates+=1
                for f in ['business_name','business_address','country']:flags['missing_'+f]+=not raw[f].strip()
                countries[raw['country']]+=1;joined=raw['business_name']+' '+raw['business_address'];script=scripts(joined);script_counts['+'.join(script) or 'NONE']+=1
                flags['non_ascii']+=any(ord(c)>127 for c in joined);flags['accented_latin']+=any('LATIN' in unicodedata.name(c,'') and any(unicodedata.combining(d) for d in unicodedata.normalize('NFD',c)) for c in joined)
                flags['indic']+=any(s in script for s in ['DEVANAGARI','BENGALI','GUJARATI','GURMUKHI','TAMIL','TELUGU','KANNADA','MALAYALAM']);flags['devanagari']+='DEVANAGARI' in script;flags['mixed_script']+=len(script)>1
                flags['numeric_token']+=bool(numbers(joined));flags['likely_postal']+=bool(postal(joined))
                for field,short in [('business_name','name'),('business_address','address')]:
                    hists[short+'_length'][len(raw[field])]+=1;hists[short+'_tokens'][len(r[field+'_norm'].split())]+=1;tokens.update(r[field+'_norm'].split())
            db.executemany('INSERT INTO tokens VALUES (?,?) ON CONFLICT(value) DO UPDATE SET n=n+excluded.n',tokens.items());db.commit()
            if limit and n>=limit:break
        report={'rows':n,'scope':f'first {limit} rows maximum' if limit else 'all rows','duplicate_ids':duplicates,'countries':dict(countries),'scripts':dict(script_counts),'counts':dict(flags),'percentages':{k:100*v/n if n else None for k,v in flags.items()},'distributions':{k:distribution(v) for k,v in hists.items()},'unique':{k:db.execute('SELECT count(*) FROM unique_values WHERE kind=? AND value<>""',(k,)).fetchone()[0] for k in ['name','normalized_name','address']},'common_tokens':db.execute('SELECT value,n FROM tokens ORDER BY n DESC,value LIMIT 30').fetchall(),'rare_tokens':db.execute('SELECT value,n FROM tokens ORDER BY n,value LIMIT 30').fetchall()}
        db.close();return report

def source_report_memory(path,cfg,limit):
    countries=Counter();script_counts=Counter();tokens=Counter();hists={k:Counter() for k in ['name_length','address_length','name_tokens','address_tokens']};flags=Counter()
    ids=set();names=set();normalized_names=set();addresses=set();n=duplicates=0
    for batch in batches(stream(path,FIELDS),5000):
        if limit:batch=batch[:max(0,limit-n)]
        if not batch:break
        for raw in batch:
            n+=1;r=normalize_record(raw,cfg['normalization'],cfg['transliteration']);eid=r['entity_id']
            if eid in ids:duplicates+=1
            else:ids.add(eid)
            names.add(r['business_name_raw']);normalized_names.add(r['business_name_norm']);addresses.add(r['business_address_raw'])
            for f in ['business_name','business_address','country']:flags['missing_'+f]+=not raw[f].strip()
            countries[raw['country']]+=1;joined=raw['business_name']+' '+raw['business_address'];script=scripts(joined);script_counts['+'.join(script) or 'NONE']+=1
            flags['non_ascii']+=any(ord(c)>127 for c in joined);flags['accented_latin']+=any('LATIN' in unicodedata.name(c,'') and any(unicodedata.combining(d) for d in unicodedata.normalize('NFD',c)) for c in joined)
            flags['indic']+=any(s in script for s in ['DEVANAGARI','BENGALI','GUJARATI','GURMUKHI','TAMIL','TELUGU','KANNADA','MALAYALAM']);flags['devanagari']+='DEVANAGARI' in script;flags['mixed_script']+=len(script)>1
            flags['numeric_token']+=bool(numbers(joined));flags['likely_postal']+=bool(postal(joined))
            for field,short in [('business_name','name'),('business_address','address')]:hists[short+'_length'][len(raw[field])]+=1;hists[short+'_tokens'][len(r[field+'_norm'].split())]+=1;tokens.update(r[field+'_norm'].split())
        if n%500000==0:print(f'EDA {Path(path).name}: {n:,}',flush=True)
        if limit and n>=limit:break
    return {'rows':n,'scope':f'first {limit} rows maximum' if limit else 'all rows','duplicate_ids':duplicates,'countries':dict(countries),'scripts':dict(script_counts),'counts':dict(flags),'percentages':{k:100*v/n if n else None for k,v in flags.items()},'distributions':{k:distribution(v) for k,v in hists.items()},'unique':{'name':len(names-{""}),'normalized_name':len(normalized_names-{""}),'address':len(addresses-{""})},'common_tokens':sorted(tokens.items(),key=lambda x:(-x[1],x[0]))[:30],'rare_tokens':sorted(tokens.items(),key=lambda x:(x[1],x[0]))[:30]}

def truth_report(path,limit):
    counts=Counter();n=links=s2=s3=0
    for row in stream(path,['source1_entity_id','matched_entity_ids']):
        ids=id_list(row['matched_entity_ids']);n+=1;counts[len(ids)]+=1;links+=len(ids);s2+=sum(t.startswith('S2-') for t in ids);s3+=sum(t.startswith('S3-') for t in ids)
        if limit and n>=limit:break
    return {'rows':n,'scope':f'first {limit} rows maximum' if limit else 'all rows','singleton_percentage':100*counts[0]/n if n else None,'one_match':counts[1],'two_matches':counts[2],'three_plus':sum(v for k,v in counts.items() if k>=3),'total_positive_links':links,'source2_links':s2,'source3_links':s3,'average_matches_per_s1':links/n if n else None,'match_count_distribution':dict(counts)}

def pair_report(index,cfg,out):
    from hybrid_retriever import HybridRetriever
    from splitting import split
    from features import FeatureBuilder,FEATURE_NAMES
    queries=split(index,cfg)['train'][:100];r=HybridRetriever(index,cfg);builder=FeatureBuilder(r);rng=random.Random(cfg['seed']);saved=[];values={'true_positive':[],'random_negative':[],'hard_negative':[]}
    total=r.db.execute('SELECT count(*) FROM targets').fetchone()[0]
    for qs in batches(queries,cfg['query_batch_size']):
        for q,cs in zip(qs,r.query_batch(qs)):
            truth=set(q['matches']);positive=[]
            for part in batches(truth,400):
                for row in r.db.execute(f'SELECT rowid,data FROM targets WHERE entity_id IN ({",".join("?" for _ in part)})',part):positive.append({'rowid':row[0],'target':json.loads(row[1]),'scores':{},'ranks':{}})
            negative=sorted([c for c in cs if c['target']['entity_id'] not in truth],key=lambda c:min(c['ranks'].values()))[:2]
            randoms=[]
            if total:
                for _ in range(20):
                    rid=rng.randint(1,total);t=get_rows(r.db,'targets',[rid])[rid]
                    if t['entity_id'] not in truth:randoms.append({'rowid':rid,'target':t,'scores':{},'ranks':{}});break
            for kind,candidates in [('true_positive',positive[:2]),('hard_negative',negative),('random_negative',randoms)]:
                X=builder.batch([q],[candidates]);values[kind].extend(X.tolist())
                for c,features in zip(candidates,X):saved.append([kind,q['entity_id'],c['target']['entity_id'],*features.tolist()])
    with (out/'pair_samples.tsv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(['sample_type','source1_entity_id','candidate_entity_id',*FEATURE_NAMES]);w.writerows(saved)
    summary={}
    for kind,rows in values.items():
        X=np.asarray(rows);summary[kind]={'pairs':len(rows),'features':{name:{'mean':float(X[:,i].mean()),'p50':float(np.median(X[:,i])),'p95':float(np.quantile(X[:,i],.95))} for i,name in enumerate(FEATURE_NAMES)} if len(rows) else {}}
    dump(out/'similarity_distributions.json',{'scope':'descriptive train-group sample; known positives used only for EDA, never candidate injection','groups':summary});r.close()

def run(data,cfg,output,limit=0,index=None,memory_mode=False):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);report={}
    for s in (1,2,3):
        path=Path(data)/f'train_source{s}.tsv';report[path.name]=(source_report_memory(path,cfg,limit) if memory_mode else source_report(path,cfg,limit)) if path.exists() else {'status':'not available; not read'}
    path=Path(data)/'train_ground_truth.tsv';report[path.name]=truth_report(path,limit) if path.exists() else {'status':'not available; not read'}
    dump(out/'report.json',report)
    if index:pair_report(index,cfg,out)
    print(f'EDA report: {out}/report.json');return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--config',default='configs/baseline.json');p.add_argument('--output',default='artifacts/eda');p.add_argument('--max-rows',type=int,default=0);p.add_argument('--index');p.add_argument('--pairs-only',action='store_true');p.add_argument('--memory-mode',action='store_true');a=p.parse_args()
    if a.pairs_only:
        if not a.index: p.error('--pairs-only requires --index')
        Path(a.output).mkdir(parents=True,exist_ok=True);pair_report(a.index,config(a.config),Path(a.output))
    else:run(a.data,config(a.config),a.output,a.max_rows,a.index,a.memory_mode)
