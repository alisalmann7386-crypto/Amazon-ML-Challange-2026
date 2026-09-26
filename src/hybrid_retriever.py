"""Union all channels; reciprocal-rank fusion is used only for Recall@K reporting."""
import time
from io_utils import connect,get_rows
from tfidf_retriever import CharTfidfRetriever
from bm25_retriever import BM25Retriever

CHANNELS=['tfidf_'+f for f in ['name','address','tname','taddress']]+['bm25_'+f for f in ['name','address','tname','taddress']]

class HybridRetriever:
    def __init__(self,index,cfg):
        self.index=index;self.cfg=cfg;self.db=connect(index)
        self.tf=CharTfidfRetriever(index,cfg).load();self.bm=BM25Retriever(index,cfg).fit()
        self.timing={'tfidf_seconds':0.,'bm25_seconds':0.,'queries':0}
    def query_batch(self,queries,top_k=None):
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
        return [[dict(c,target=rows[rid]) for rid,c in sorted(cs.items(),key=lambda pair:rows[pair[0]]['entity_id'])] for cs in merged]
    def query(self,q,top_k=None):return self.query_batch([q],top_k)[0]
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
