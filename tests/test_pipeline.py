import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from core import (macro_f05, normalize, Retriever, id_list, read_sources,
                  validate, write_lists, read_lists)
from make_demo import generate
from pipeline import train, predict, groups_for

class TestResolution(unittest.TestCase):
    def test_official_metric_and_singletons(self):
        self.assertAlmostEqual(macro_f05({'x': {'a','b'}}, {'x': {'a','b','c'}}), 5/7)
        self.assertEqual(macro_f05({'x': set()}, {'x': set()}), 1)
        self.assertEqual(macro_f05({'x': set()}, {'x': {'a'}}), 0)
        self.assertEqual(macro_f05({'x': {'a'}}, {'x': set()}), 0)

    def test_normalization_and_lists(self):
        self.assertEqual(normalize('Café & CO.'), 'cafe and co')
        for value in ['a,a', 'a,', ' a', 'a, b']:
            with self.assertRaises(ValueError):
                id_list(value)

    def test_no_blank_matches(self):
        r = {'entity_id': 'S2-1', 'business_name': '', 'business_address': '', 'country': 'France'}
        self.assertEqual(Retriever([r]).candidates(r), [])
        self.assertEqual(Retriever([]).candidates(r), [])

    def test_shared_target_grouping(self):
        q = [{'entity_id': key} for key in ['a','b','c']]
        groups = groups_for(q, {'a': {'t'}, 'b': {'t'}, 'c': set()})
        self.assertEqual(groups[0], groups[1])
        self.assertNotEqual(groups[0], groups[2])

    def test_end_to_end_and_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generate(root/'data')
            artifact = str(root/'model.joblib')
            train(SimpleNamespace(data=root/'data/train', model=artifact, k=20, folds=3))
            predict(SimpleNamespace(data=root/'data/test', model=artifact, output=root/'output'))
            q,t = read_sources(root/'data/test', 'test')
            matches = validate(root/'output', q, t)
            self.assertEqual(len(matches), 12)
            self.assertFalse(matches['S1-000'])
            self.assertEqual(matches['S1-001'], {'S2-001', 'S3-001'})
            matches['S1-000'] = {'S2-nonexistent'}
            write_lists(root/'output/matching_results.tsv', q, matches, 'matched_entity_ids')
            with self.assertRaises(ValueError):
                validate(root/'output', q, t)

if __name__ == '__main__':
    unittest.main()
