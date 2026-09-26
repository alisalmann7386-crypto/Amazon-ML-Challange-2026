"""Label-aware evaluation only; retrieval itself never sees labels."""
import argparse
import json
from pathlib import Path
import numpy as np
from io_utils import config,batches,dump
from hybrid_retriever import HybridRetriever,ranked,CHANNELS
from metrics import slices
from splitting import split

MODES={'A_tfidf':['tfidf_name','tfidf_address'],'B_bm25':['bm25_name','bm25_address'],'C_hybrid':['tfidf_name','tfidf_address','bm25_name','bm25_address'],'D_hybrid_transliteration':CHANNELS}

def compare(index,cfg,output,queries=None):
    queries=split(index,cfg)['train'] if queries is None else queries
    r=HybridRetriever(index,cfg);stats={m:{} for m in MODES};attributions={'only_tfidf':0,'only_bm25':0,'only_transliteration':0,'multiple_families':0,'missed_all':0}
    for qs in batches(queries,cfg['query_batch_size']):
        # Use >=50 per channel for Recall@50; union metrics use configured channel caps.
        for q,cs in zip(qs,r.query_batch(qs,max(50,cfg['tfidf']['top_k'],cfg['bm25']['top_k']))):
            actual=set(q['matches']);base=[]
            for c in cs:
                channels={ch:rank for ch,rank in c['ranks'].items() if rank<=cfg['tfidf' if ch.startswith('tfidf') else 'bm25']['top_k']}
                if channels:base.append(dict(c,ranks=channels,scores={ch:c['scores'][ch] for ch in channels}))
            for mode,channels in MODES.items():
                ids=ranked(base,channels);ranking=ranked(cs,channels);found=len(actual&set(ids))
                for key in slices(q):
                    s=stats[mode].setdefault(key,{'true':0,'found':0,'non_singletons':0,'coverage':0.,'counts':[],'at':{k:0 for k in [5,10,20,30,50]}})
                    s['true']+=len(actual);s['found']+=found;s['counts'].append(len(ids))
                    if actual:s['coverage']+=found/len(actual);s['non_singletons']+=1
                    for k in s['at']:s['at'][k]+=len(actual&set(ranking[:k]))
            for tid in actual:
                c=next((c for c in base if c['target']['entity_id']==tid),None)
                if c is None:attributions['missed_all']+=1;continue
                families=set('transliteration' if ch.endswith(('tname','taddress')) else 'tfidf' if ch.startswith('tfidf') else 'bm25' for ch in c['ranks'])
                key='only_'+next(iter(families)) if len(families)==1 else 'multiple_families';attributions[key]+=1
    total=r.db.execute('SELECT count(*) FROM targets').fetchone()[0];report={}
    for mode,groups in stats.items():
        report[mode]={}
        for key,s in groups.items():
            counts=np.asarray(s['counts']);report[mode][key]={'candidate_micro_recall':s['found']/s['true'] if s['true'] else None,'macro_candidate_coverage_non_singletons':s['coverage']/s['non_singletons'] if s['non_singletons'] else None,'s1_count':len(counts),'average_candidates':float(counts.mean()),'median_candidates':float(np.median(counts)),'p95_candidates':float(np.quantile(counts,.95)),'reduction_ratio':1-float(counts.mean())/total if total else None,**{f'recall_at_{k}':found/s['true'] if s['true'] else None for k,found in s['at'].items()}}
    timing=r.timing.copy();r.close()
    for backend in ['tfidf','bm25']:timing[backend+'_queries_per_second']=timing['queries']/max(timing[backend+'_seconds'],1e-9)
    evaluation_scope=cfg.get('evaluation_scope','Sampled S1 queries; complete supplied target catalog.')
    output=Path(output);dump(output/'metrics.json',{'scope':evaluation_scope+' Training groups only; Recall@K uses reciprocal-rank fusion of up to max(50,channel k) neighbors/channel; union counts use configured channel k.','methods':report,'attribution':attributions,'timing':timing})
    dump(output/'transliteration.json',{'without':report['C_hybrid'],'with':report['D_hybrid_transliteration'],'attribution':attributions})
    print(json.dumps({m:g.get('all') for m,g in report.items()},indent=2));print(json.dumps(timing,indent=2));return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--config',default='configs/baseline.json');p.add_argument('--output',default='artifacts/retrieval_comparison');a=p.parse_args();compare(a.index,config(a.config),a.output)
if __name__=='__main__':main()
