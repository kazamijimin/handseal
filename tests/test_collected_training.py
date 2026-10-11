import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from scripts.prepare_collected_training import prepare


class CollectedTrainingTests(unittest.TestCase):
    def fixture(self, root):
        base, source = root / 'base', root / 'source'
        rows = []
        rng = np.random.default_rng(42)
        for split in ['train', 'valid', 'test']:
            for label in ['dragon', 'ram', 'snake', 'tiger', 'no_seal']:
                path = base / split / label / 'sample.jpg'
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(path)
                rows.append({'path': path.relative_to(base).as_posix(), 'label': label,
                             'source_key': split + '_' + label,
                             'split': split, 'width': 32, 'height': 32,
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        (base / 'metadata.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        source.mkdir()
        return base, source, rng

    def test_uses_only_new_training_photos_and_preserves_existing_holdouts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, source, rng = self.fixture(root)
            for split in ['train', 'valid', 'test']:
                path = source / split / 'dragon' / 'capture.JPG'
                path.parent.mkdir(parents=True)
                Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)).save(path)
            (source / 'metadata.jsonl').write_text(json.dumps({'path':'train/dragon/capture.JPG',
                'participant_id':'P01','session_id':'S01'})+'\n')
            output = root / 'output'
            report = prepare(base, source, output)
            self.assertEqual(report['added'], 1)
            for split in ['valid','test']:
                for original in (base / split).glob('*/*.jpg'):
                    self.assertEqual(original.read_bytes(), (output / original.relative_to(base)).read_bytes())
            new_row = json.loads((output / 'metadata.jsonl').read_text().splitlines()[-1])
            self.assertEqual(new_row['participant_id'], 'P01')
            self.assertEqual(new_row['split'], 'train')
            self.assertEqual(report['baseline_issues'], [])

    def test_excludes_a_new_photo_similar_to_existing_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, source, _ = self.fixture(root)
            path = source / 'train/dragon/duplicate.jpg'
            path.parent.mkdir(parents=True)
            path.write_bytes((base / 'valid/dragon/sample.jpg').read_bytes())
            (source / 'metadata.jsonl').write_text('')
            report = prepare(base, source, root / 'output')
            self.assertEqual(report['added'], 0)
            self.assertGreater(report['excluded_held_out_similarity'], 0)
