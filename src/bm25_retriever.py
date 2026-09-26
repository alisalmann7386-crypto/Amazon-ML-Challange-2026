"""SQLite is the backend; BM25 is the ranking function. No target N+1 fetches."""
import argparse
import json
import time
from pathlib import Path
from io_utils import connect,config,dump
from tfidf_retriever import FIELDS
from scalable import log

class BM25Retriever:
    def __init__(self,index,cfg):self.index=Path(index);self.cfg=cfg;self.db=connect(index)
    def fit(self):
        start=time.perf_counter()
        self.db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS hybrid_search USING fts5(name,address,tname,taddress,content='targets',content_rowid='rowid',tokenize='unicode61 remove_diacritics 0')")
        marker=self.index/'bm25.json';identity={'catalog':json.loads((self.index/'catalog.json').read_text()),'bm25':self.cfg['bm25']}
        if marker.exists():
            if json.loads(marker.read_text())!=identity:raise ValueError('BM25 config mismatch')
            return self
        self.db.execute("INSERT INTO hybrid_search(hybrid_search) VALUES('rebuild')");self.db.commit();dump(marker,identity)
        dump(self.index/'bm25_stats.json',{'build_seconds':time.perf_counter()-start});log(f'BM25 build seconds: {time.perf_counter()-start:.2f}');return self
    def query_batch(self,queries,top_k=None):
        k=top_k or self.cfg['bm25']['top_k'];out=[]
        if k<1:raise ValueError('top_k must be positive')
        channels=['name','address']+(['tname','taddress'] if self.cfg['transliteration']['enabled'] else [])
        for q in queries:
            result={}
            for field in channels:
                terms=list(dict.fromkeys(q[FIELDS[field]].split()))[:self.cfg['bm25']['max_tokens']]
                expression=field+' : ('+' OR '.join('"'+t.replace('"','""')+'"' for t in terms)+')'
                result['bm25_'+field]=[(int(r[0]),float(r[1])) for r in self.db.execute('SELECT rowid,rank FROM hybrid_search WHERE hybrid_search MATCH ? ORDER BY rank,rowid LIMIT ?',(expression,k))] if terms else []
            out.append(result)
        return out
    def query(self,q,top_k=None):return self.query_batch([q],top_k)[0]
    def close(self):self.db.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--config',default='configs/baseline.json');a=p.parse_args();b=BM25Retriever(a.index,config(a.config)).fit();b.close()
