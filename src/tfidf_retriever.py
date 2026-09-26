"""Persisted float32 CSR shards and exact sparse top-N over every target shard."""
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from io_utils import connect,batches,config,dump,digest
from scalable import log

FIELDS={'name':'business_name_norm','address':'business_address_norm','tname':'business_name_transliterated','taddress':'business_address_transliterated'}

class CharTfidfRetriever:
    def __init__(self,index,cfg):
        self.index=Path(index);self.cfg=cfg;self.folder=self.index/'tfidf';self.models={};self.shards={}
    def fit(self):
        self.folder.mkdir(exist_ok=True)
        identity={'catalog':json.loads((self.index/'catalog.json').read_text()),'tfidf':self.cfg['tfidf'],'seed':self.cfg['seed']}
        marker=self.folder/'manifest.json'
        if marker.exists():
            if json.loads(marker.read_text())!=identity:raise ValueError('TF-IDF config mismatch; use new index')
            return self.load()
        db=connect(self.index);n=db.execute('SELECT count(*) FROM targets').fetchone()[0]
        cfg=self.cfg['tfidf'];limit=cfg['vocabulary_sample'];count=min(n,limit) if limit else n
        ids=np.sort(np.random.default_rng(self.cfg['seed']).choice(n,count,replace=False)+1)
        channels=['name','address']+(['tname','taddress'] if self.cfg['transliteration']['enabled'] else [])
        progress=self.folder/'progress.json'
        if progress.exists() and json.loads(progress.read_text())!=identity:raise ValueError('Partial TF-IDF config mismatch')
        dump(progress,identity)
        report={}
        for field in channels:
            start=time.perf_counter();directory=self.folder/field;directory.mkdir(exist_ok=True)
            model_file=directory/'vectorizer.joblib'
            corpus=[]
            if not model_file.exists() or not (directory/'word_vectorizer.joblib').exists():
                for part in batches(ids.tolist(),500):
                    corpus.extend(r[0] for r in db.execute(f'SELECT {field} FROM targets WHERE rowid IN ({",".join("?" for _ in part)}) ORDER BY rowid',part))
                options={k:cfg[k] for k in ['analyzer','min_df','max_df','max_features']}
                options['ngram_range']=tuple(cfg['ngram_range'])
                vec=TfidfVectorizer(**options,lowercase=False,dtype=np.float32,norm='l2',sublinear_tf=True)
                try:vec.fit(corpus)
                except ValueError as e:
                    if any(t in str(e) for t in ['empty vocabulary','no terms remain','After pruning','max_df corresponds']):vec=None
                    else:raise
                tmp=directory/'vectorizer.tmp';joblib.dump(vec,tmp);tmp.replace(model_file)
                word=TfidfVectorizer(lowercase=False,token_pattern=r'(?u)\b\w+\b',dtype=np.float32,max_features=cfg['max_features'])
                try:word.fit(corpus)
                except ValueError:word=None
                word_file=directory/'word_vectorizer.joblib';word_tmp=directory/'word_vectorizer.tmp'
                joblib.dump(word,word_tmp);word_tmp.replace(word_file)
            vec=joblib.load(model_file);self.models[field]=vec
            total_nnz=total_bytes=0;shards=[]
            for batch in batches(db.execute(f'SELECT rowid,{field} FROM targets ORDER BY rowid'),cfg['batch_size']):
                offset=int(batch[0][0]);path=directory/f'{offset:012d}.npz'
                if not path.exists():
                    matrix=vec.transform([r[1] for r in batch]).tocsr() if vec else sparse.csr_matrix((len(batch),0),dtype=np.float32)
                    temp=path.with_suffix('.tmp.npz');sparse.save_npz(temp,matrix);temp.replace(path)
                else:matrix=sparse.load_npz(path)
                shards.append({'file':path.name,'offset':offset,'rows':len(batch)})
                total_nnz+=matrix.nnz;total_bytes+=matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes
            dump(directory/'shards.json',shards);self.shards[field]=shards
            report[field]={'vocabulary_size':len(vec.vocabulary_) if vec else 0,'shape':[n,len(vec.vocabulary_) if vec else 0], 'nnz':total_nnz,'csr_bytes':total_bytes,'build_seconds':time.perf_counter()-start,'vocabulary_fit_rows':count}
            log(f'TF-IDF {field}: {report[field]}')
        db.close();dump(self.folder/'stats.json',report);self.save(identity);return self
    def save(self,identity=None):
        if identity is not None:dump(self.folder/'manifest.json',identity)
        return self
    def load(self):
        meta=json.loads((self.folder/'manifest.json').read_text())
        if meta['tfidf']!=self.cfg['tfidf'] or meta['catalog']!=json.loads((self.index/'catalog.json').read_text()):raise ValueError('Stale TF-IDF cache')
        for field in FIELDS:
            directory=self.folder/field
            if (directory/'shards.json').exists():
                self.models[field]=joblib.load(directory/'vectorizer.joblib');self.shards[field]=json.loads((directory/'shards.json').read_text())
        return self
    def query_batch(self,queries,top_k=None):
        k=top_k or self.cfg['tfidf']['top_k'];out=[{} for _ in queries]
        if k<1:raise ValueError('top_k must be positive')
        for field,vec in self.models.items():
            if vec is None:continue
            Q=vec.transform([q[FIELDS[field]] for q in queries]).tocsr();best=[[] for _ in queries]
            for shard in self.shards[field]:
                T=sparse.load_npz(self.folder/field/shard['file']).tocsr()
                sims=sp_matmul_topn(Q,T.T.tocsr(),top_n=min(k,T.shape[0]),threshold=0.0,sort=True,n_threads=self.cfg['tfidf']['threads'])
                for i in range(len(queries)):
                    a,b=sims.indptr[i:i+2]
                    best[i].extend((int(j)+shard['offset'],float(s)) for j,s in zip(sims.indices[a:b],sims.data[a:b]) if s>0)
                    best[i]=sorted(best[i],key=lambda p:(-p[1],p[0]))[:k]
            for i,items in enumerate(best):out[i]['tfidf_'+field]=items
        return out
    def query(self,query,top_k=None):return self.query_batch([query],top_k)[0]

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--config',default='configs/baseline.json');a=p.parse_args();CharTfidfRetriever(a.index,config(a.config)).fit()
