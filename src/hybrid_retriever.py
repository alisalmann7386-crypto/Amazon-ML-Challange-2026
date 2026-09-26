"""Merge channels with reciprocal-rank fusion and enforce the final candidate cap."""
import time
from io_utils import connect,get_rows
from tfidf_retriever import CharTfidfRetriever
from bm25_retriever import BM25Retriever

CHANNELS=['tfidf_'+f for f in ['name','address','tname','taddress']]+['bm25_'+f for f in ['name','address','tname','taddress']]

class HybridRetriever:
    def __init__(self,index,cfg):
        self.index=index;self.cfg=cfg;self.db=connect(index)
        self.tf=CharTfidfRetriever(index,cfg).load();self.bm=BM25Retriever(index,cfg).fit()
        self.timing={'tfidf_seconds':0.,'bm25_seconds':0.,'queries':0,'uncapped_candidates':0,'final_candidates':0}
    def query_batch(self,queries,top_k=None,use_final_cap=True,final_k=None):
        start=time.perf_counter();tf=self.tf.query_batch(queries,top_k);self.timing['tfidf_seconds']+=time.perf_counter()-start
        start=time.perf_counter();bm=self.bm.query_batch(queries,top_k);self.timing['bm25_seconds']+=time.perf_counter()-start
        self.timing['queries']+=len(queries);merged=[];all_ids=set()
        for left,right in zip(tf,bm):
            candidates={}
            for channel,items in {**left,**right}.items():
                for rank,(rid,score) in enumerate(items,1):
                    c=candidates.setdefault(rid,{'rowid':rid,'scores':{},'ranks':{}})
                    c['scores'][channel]=score;c['ranks'][channel]=rank;all_ids.add(rid)
            merged.append(candidates)
        rows=get_rows(self.db,'targets',all_ids)
        if use_final_cap:
            final_k=self.cfg.get('blocking',{}).get('final_k',20) if final_k is None else final_k
            if not isinstance(final_k,int) or final_k<1:raise ValueError('blocking.final_k must be a positive integer')
        output=[]
        for candidates in merged:
            ranked_candidates=[]
            for rid,candidate in candidates.items():
                candidate=dict(candidate,target=rows[rid])
                candidate['rrf_score']=sum(1/(60+rank) for rank in candidate['ranks'].values())
                ranked_candidates.append(candidate)
            ranked_candidates.sort(key=lambda candidate:(-candidate['rrf_score'],candidate['target']['entity_id']))
            self.timing['uncapped_candidates']+=len(ranked_candidates)
            if use_final_cap:ranked_candidates=ranked_candidates[:final_k]
            self.timing['final_candidates']+=len(ranked_candidates);output.append(ranked_candidates)
        return output
    def query(self,q,top_k=None,use_final_cap=True,final_k=None):return self.query_batch([q],top_k,use_final_cap,final_k)[0]
    def close(self):self.db.close();self.bm.close()

def ranked(candidates,channels):
    scored=[]
    for c in candidates:
        values=[1/(60+r) for ch,r in c['ranks'].items() if ch in channels]
        if values:scored.append((sum(values),c['target']['entity_id']))
    return [tid for _,tid in sorted(scored,key=lambda pair:(-pair[0],pair[1]))]

if __name__=='__main__':
    from retrieval_evaluation import main
    main()
