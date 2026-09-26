import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from prepare_data import prepare
from make_demo import generate
from data_integrity import audit


class TestPreflight(unittest.TestCase):
    def test_partial_preparation_resumes_and_changed_inputs_fail(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            generate(root/'data')
            source, target = root/'data/train', root/'ready'
            target.mkdir()
            name = 'train_source1.tsv'
            (target/name).write_bytes((source/name).read_bytes())
            prepare(source, target, 'train')
            prepare(source, target, 'train')
            with (source/name).open('a') as f:
                f.write('S1-new\tName\tAddress\tUS\n')
            with self.assertRaisesRegex(ValueError, 'differs from input'):
                prepare(source, target, 'train')

    def test_duplicate_ground_truth_fails(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            generate(root/'data')
            truth = root/'data/train/train_ground_truth.tsv'
            duplicate = truth.read_text().splitlines()[1]
            with truth.open('a') as f:
                f.write(duplicate+'\n')
            with self.assertRaisesRegex(ValueError, 'Integrity check failed'):
                audit(root/'data/train', 'train', root/'integrity.json')
