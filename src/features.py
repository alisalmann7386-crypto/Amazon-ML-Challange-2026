"""Fixed feature order and pairwise vector scores, independent of retrieval channel."""
from pathlib import Path
import joblib
import numpy as np
import similarity as s
from hybrid_retriever import CHANNELS
from tfidf_retriever import FIELDS

TEXT_STATS=['exact','levenshtein','jaro','jaro_winkler','jaccard','overlap','containment','token_set','token_sort','prefix','length_ratio','missing']
FEATURE_NAMES=[f'{field}_{metric}' for field in ['name','tname','address','taddress'] for metric in TEXT_STATS]+['core_exact']+[f'{field}_{kind}_cosine' for field in FIELDS for kind in ['char','word']]+['rare_name_token_overlap','numeric_jaccard','number_intersection','numeric_conflict','house_equal','postal_equal','postal_conflict','postal_missing','country_equal','country_different','country_missing','script_match','script_mismatch','high_name_high_address','high_name_low_address','low_name_high_address']+[f'{c}_{kind}' for c in CHANNELS for kind in ['score','present']]+['retrieval_channel_count','multiple_channels']

def text_stats(a,b):
    return [s.exact(a,b),s.levenshtein(a,b),s.jaro(a,b),s.jaro_winkler(a,b),s.jaccard(a.split(),b.split()),s.overlap(a.split(),b.split()),s.containment(a.split(),b.split()),s.token_set(a,b),s.token_sort(a,b),s.prefix(a,b),s.length_ratio(a,b),float(not a or not b)]

class FeatureBuilder:
    def __init__(self,retriever):
        self.char=retriever.tf.models;self.word={}
        for field in FIELDS:
            path=Path(retriever.index)/'tfidf'/field/'word_vectorizer.joblib'
            self.word[field]=joblib.load(path) if path.exists() else None
        vec=self.word.get('name');self.rare=set()
        if vec is not None:
            cutoff=float(np.quantile(vec.idf_,.75))
            self.rare={t for t,i in vec.vocabulary_.items() if vec.idf_[i]>=cutoff}
    def batch(self,queries,candidate_lists):
        pairs=[(q,c) for q,cs in zip(queries,candidate_lists) for c in cs]
        if not pairs:return np.zeros((0,len(FEATURE_NAMES)),dtype=np.float32)
        vector_scores={}
        for field,key in FIELDS.items():
            a=[q[key] for q,c in pairs];b=[c['target'][key] for q,c in pairs]
            vector_scores[field]=[s.cosine_pairs(self.char.get(field),a,b),s.cosine_pairs(self.word.get(field),a,b)]
        output=[]
        for i,(q,c) in enumerate(pairs):
            t=c['target'];row=[]
            for field in ['name','tname','address','taddress']:row.extend(text_stats(q[FIELDS[field]],t[FIELDS[field]]))
            row.append(s.exact(q['business_name_core'],t['business_name_core']))
            for field in FIELDS:row.extend(float(values[i]) for values in vector_scores[field])
            a,b=q['business_address_norm'],t['business_address_norm'];na,nb=s.numbers(a),s.numbers(b);pa,pb=s.postal(a),s.postal(b)
            ca,cb=q['country_norm'],t['country_norm'];qa,ta=q['business_name_norm'],t['business_name_norm']
            name=s.jaro_winkler(qa,ta);addr=s.token_set(a,b)
            row.extend([len(set(qa.split())&set(ta.split())&self.rare),s.jaccard(na,nb),len(na&nb),float(bool(na and nb) and not na&nb),s.exact(s.house(a),s.house(b)),float(bool(pa&pb)),float(bool(pa and pb) and not pa&pb),float(not pa or not pb),s.exact(ca,cb),float(bool(ca and cb) and ca!=cb),float(not ca or not cb),float(q['script']==t['script'] and q['script']!='NONE'),float(q['script']!=t['script'] and 'NONE' not in (q['script'],t['script'])),float(name>=.85 and addr>=.8),float(name>=.85 and addr<.4),float(name<.4 and addr>=.8)])
            for channel in CHANNELS:row.extend([c['scores'].get(channel,0.),float(channel in c['scores'])])
            row.extend([len(c['scores']),float(len(c['scores'])>1)])
            if len(row)!=len(FEATURE_NAMES):raise RuntimeError('Feature order mismatch')
            output.append(row)
        return np.asarray(output,dtype=np.float32)
