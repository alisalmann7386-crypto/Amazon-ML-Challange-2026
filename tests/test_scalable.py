import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace as Args
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from make_demo import generate
from scalable import build,connect,retrieve,train,predict,metrics
from prepare_data import prepare
from core import read_lists

class TestScalable(unittest.TestCase):
    def test_metric_includes_unretrieved_and_singletons(self):
        result=metrics(np.array([2,0,1]),np.array([0,0,1]),np.array([1,0,0]),np.array([.9,.2,.1]),.5)
        self.assertAlmostEqual(result['macro_f05'],(1.25/1.5+1)/3)
        self.assertAlmostEqual(result['candidate_micro_recall'],1/3)

    def test_zip_flatten_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);generate(root/'demo')
            with zipfile.ZipFile(root/'inputs.zip','w') as z:
                for p in (root/'demo/train').glob('*.tsv'):z.write(p,'nested/'+p.name.replace('source3','source3(1)'))
            prepare(root/'inputs.zip',root/'ready','train')
            self.assertTrue((root/'ready/train_source3.tsv').exists())
            with self.assertRaises(ValueError):prepare(root,root/'again','train')

    def test_index_train_predict_and_cache(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);generate(root/'data')
            for split in ['train','test']:
                a=Args(data=root/'data'/split,split=split,index=root/f'{split}.db')
                build(a);build(a)
            with connect(root/'train.db') as db:
                q=dict(db.execute("select * from queries where entity_id='S1-001'").fetchone())
                self.assertIn('S2-001',{r['entity_id'] for r in retrieve(db,q,20)})
                q['business_name']=q['business_address']=''
                self.assertEqual(retrieve(db,q,20),[])
            a=Args(index=root/'train.db',work=root/'run',sample_size=12,seed=42,k=20,epochs=2)
            train(a)
            report=json.loads((root/'run/metrics.json').read_text())
            self.assertEqual(report['sample_size_actual'],12)
            self.assertIn('holdout',report)
            predict(Args(index=root/'test.db',model=root/'run/model.joblib',output=root/'output'))
            results=read_lists(root/'output/matching_results.tsv','matched_entity_ids')
            self.assertEqual(len(results),12)
            train(a) # Completed feature shards are reusable.
            a.k=5
            with self.assertRaises(ValueError):train(a)

if __name__=='__main__':unittest.main()
