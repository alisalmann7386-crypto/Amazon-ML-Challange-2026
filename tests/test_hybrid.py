import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from normalize import clean,normalize_record
from transliterate import transliterate,scripts
from similarity import levenshtein,jaro_winkler,jaccard,token_set,token_sort,cosine_pairs
from core import table,FIELDS
from make_demo import generate
from io_utils import config,build,connect,linked_groups,dump
from tfidf_retriever import CharTfidfRetriever
from bm25_retriever import BM25Retriever
from hybrid_retriever import HybridRetriever
from splitting import split
from features import FeatureBuilder,FEATURE_NAMES
from train import run
from inference import run as infer
from validate_submission import validate
from retrieval_evaluation import compare
from eda import run as eda
from metrics import validate_threshold,threshold_search
from data_integrity import audit
from exact_baseline import run as exact_baseline

class TestHybrid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name);generate(cls.root/'data')
        cls.cfg=config();cls.cfg['sample_size']=12;cls.cfg['tfidf'].update(min_df=1,batch_size=5,vocabulary_sample=100,top_k=3);cls.cfg['bm25']['top_k']=3;cls.cfg['query_batch_size']=4;cls.cfg['feature_shard_queries']=5;cls.cfg['model'].update(n_estimators=20,min_child_samples=1,num_leaves=7)
        for split_ in ['train','test']:
            for source in [1,2,3]:
                p=cls.root/'data'/split_/f'{split_}_source{source}.tsv'
                with p.open() as f:rows=list(csv.DictReader(f,delimiter='\t'))
                rows[1]['business_name']='श्री बालाजी ट्रेडर्स' if source==2 else 'Shree Balaji Traders'
                with p.open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=FIELDS,delimiter='\t');w.writeheader();w.writerows(rows)
        cls.index=cls.root/'index';build(cls.root/'data/train','train',cls.index,cls.cfg)
        CharTfidfRetriever(cls.index,cls.cfg).fit();bm=BM25Retriever(cls.index,cls.cfg).fit();bm.close()
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def test_unicode_and_transliteration(self):
        self.assertEqual(clean('ＡＢＣ Straße & Co.'),'abc strasse and co')
        q=normalize_record({'entity_id':'S1-x','business_name':'श्री बालाजी ट्रेडर्स','business_address':'12 Rd','country':'India'})
        self.assertEqual(q['business_name_norm'],'श्री बालाजी ट्रेडर्स')
        self.assertIn('balaji',q['business_name_transliterated']);self.assertEqual(q['business_address_norm'],'12 road');self.assertIn('DEVANAGARI',scripts(q['business_name_raw']))
        a=normalize_record({'entity_id':'x','business_name':'ACME Pvt Ltd','business_address':'St Mary Rd','country':''})
        self.assertEqual(a['business_name_norm'],'acme private limited');self.assertEqual(a['business_name_core'],'acme');self.assertEqual(a['business_address_norm'],'st mary road')
        self.assertGreater(levenshtein('shree balaji traders',q['business_name_transliterated']),.5)
    def test_similarity(self):
        self.assertEqual(levenshtein('',''),0);self.assertEqual(jaro_winkler('abc','abc'),1)
        self.assertAlmostEqual(jaccard(['a','b'],['b','c']),1/3)
        self.assertEqual(token_set('red blue','blue red'),1);self.assertEqual(token_sort('red blue','blue red'),1)
    def test_retrieval_and_features(self):
        r=HybridRetriever(self.index,self.cfg)
        q=normalize_record({'entity_id':'S1-new','business_name':'Shree Balaji Traders','business_address':'','country':'India'},self.cfg['normalization'],self.cfg['transliteration'])
        cs=r.query(q);ids={c['target']['entity_id'] for c in cs};self.assertIn('S2-001',ids)
        hindi=next(c for c in cs if c['target']['entity_id']=='S2-001');self.assertTrue(any(ch.endswith('tname') for ch in hindi['ranks']))
        self.assertEqual([c['target']['entity_id'] for c in cs],[c['target']['entity_id'] for c in r.query(q)])
        vec=r.tf.models['name'];self.assertAlmostEqual(cosine_pairs(vec,['shree balaji traders'],['shree balaji traders'])[0],1,places=5)
        typo=dict(q,business_name_norm='shree balji tradres');self.assertTrue(r.tf.query(typo)['tfidf_name'])
        X=FeatureBuilder(r).batch([q],[cs]);self.assertEqual(X.shape,(len(cs),len(FEATURE_NAMES)));self.assertTrue(np.isfinite(X).all());r.close()

    def test_global_candidate_cap(self):
        r=HybridRetriever(self.index,self.cfg)
        q=normalize_record({'entity_id':'S1-cap','business_name':'Example Company','business_address':'10 Main Road','country':'US'},self.cfg['normalization'],self.cfg['transliteration'])
        self.assertLessEqual(len(r.query(q,final_k=2)),2)
        self.assertGreaterEqual(len(r.query(q,use_final_cap=False)),len(r.query(q,final_k=2)))
        r.close()
    def test_tsv_rejections(self):
        p=self.root/'bad.tsv';p.write_text('entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-x\ta\tb\tUS\textra\n')
        with self.assertRaises(ValueError):table(p,FIELDS)
        p.write_text('entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-x\ta\tb\tUS\nS1-x\tc\td\tUS\n')
        with self.assertRaises(ValueError):table(p,FIELDS)
    def test_train_infer_save_load_and_metrics(self):
        report=run(self.index,self.cfg,self.root/'run',self.root/'final',self.root/'retrieval',self.root/'errors')
        self.assertEqual(sum(report['splits'].values()),12)
        self.assertTrue((self.root/'retrieval/transliteration.json').exists())
        self.assertTrue((self.root/'final/threshold.json').exists())
        eda(self.root/'data/train',self.cfg,self.root/'eda',index=self.index)
        infer(self.root/'data/test',self.root/'testindex',self.root/'final',self.root/'output')
        first=(self.root/'output/matching_results.tsv').read_bytes()
        infer(self.root/'data/test',self.root/'testindex',self.root/'final',self.root/'output')
        self.assertEqual(first,(self.root/'output/matching_results.tsv').read_bytes());self.assertEqual(validate(self.root/'testindex',self.root/'output'),12)
        fp=self.root/'final/feature_names.json';fp.write_text('[]')
        with self.assertRaises(ValueError):infer(self.root/'data/test',self.root/'testindex',self.root/'final',self.root/'output')
        fp.write_text(json.dumps(FEATURE_NAMES))
        (self.root/'final/threshold.json').write_text(json.dumps({'threshold':-1}))
        with self.assertRaises(ValueError):infer(self.root/'data/test',self.root/'testindex',self.root/'final',self.root/'output')
    def test_full_graph_groups(self):
        import sqlite3
        db=sqlite3.connect(':memory:');db.execute('CREATE TABLE links(qid TEXT,tid TEXT)');db.executemany('INSERT INTO links VALUES (?,?)',[('a','x'),('b','x'),('b','y'),('c','y')])
        self.assertEqual(len(set(linked_groups(db,[{'entity_id':'a'},{'entity_id':'c'}]))),1);db.close()

    def test_empty_target_corpus(self):
        root=self.root/'empty';data=root/'data';data.mkdir(parents=True)
        for source,rows in [(1,[['S1-only','Example','1 Road','US']]),(2,[]),(3,[])]:
            with (data/f'train_source{source}.tsv').open('w',newline='',encoding='utf-8') as f:
                w=csv.writer(f,delimiter='\t');w.writerow(FIELDS);w.writerows(rows)
        with (data/'train_ground_truth.tsv').open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f,delimiter='\t');w.writerow(['source1_entity_id','matched_entity_ids']);w.writerow(['S1-only',''])
        cfg=dict(self.cfg);cfg['tfidf']=dict(self.cfg['tfidf']);cfg['tfidf']['min_df']=1
        index=root/'index';build(data,'train',index,cfg)
        tf=CharTfidfRetriever(index,cfg).fit()
        query=normalize_record({'entity_id':'q','business_name':'Example','business_address':'1 Road','country':'US'},cfg['normalization'],cfg['transliteration'])
        self.assertEqual(tf.query(query),{})
        bm=BM25Retriever(index,cfg).fit();retriever=HybridRetriever(index,cfg)
        self.assertEqual(retriever.query(query),[])
        retriever.close();bm.close()

    def test_threshold_validation(self):
        self.assertEqual(validate_threshold(.5),.5)
        for value in (-.01,1.1,float('nan'),float('inf'),'bad'):
            with self.assertRaises(ValueError):validate_threshold(value)
        with self.assertRaises(ValueError):threshold_search([],np.array([],dtype=int),np.array([]),np.array([]),[-.1,.5])

    def test_integrity_repairs_preserve_source3_rows(self):
        root=self.root/'repair_case';generate(root/'data')
        path=root/'data/train/train_source3.tsv';rows=path.read_text(encoding='utf-8').splitlines()
        first=rows[1].split('\t');second=rows[2].split('\t')
        rows[1]='\t'.join(first[:3])
        rows[2]='\t'.join([second[0],second[1],second[2],'extra address fragment',second[3]])
        path.write_text('\n'.join(rows)+'\n',encoding='utf-8')
        report=audit(root/'data/train','train',root/'integrity.json')
        self.assertEqual(report['logical_rows']['train_source3.tsv'],12)
        self.assertEqual(len(report['repairs']['train_source3.tsv']),2)
        index=root/'index';build(root/'data/train','train',index,self.cfg)
        with connect(index) as db:self.assertEqual(db.execute('SELECT count(*) FROM targets WHERE entity_id LIKE "S3-%"').fetchone()[0],12)

    def test_exact_baseline_outputs_validate(self):
        output=self.root/'exact_baseline';report=exact_baseline(self.index,output)
        self.assertEqual(report['queries'],12)
        self.assertEqual(validate(self.index,output),12)

if __name__=='__main__':unittest.main()
