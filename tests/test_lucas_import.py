import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from scripts.import_lucas import enrich


class LucasImportTests(unittest.TestCase):
    def test_new_source_only_enters_training_and_preserves_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / 'base'
            (base / 'valid' / 'dragon').mkdir(parents=True)
            image = Image.fromarray(np.random.default_rng(42).integers(0, 256, (224, 224, 3), dtype=np.uint8))
            image.save(base / 'valid' / 'dragon' / 'original.jpg')
            original = (base / 'valid' / 'dragon' / 'original.jpg').read_bytes()
            (base / 'metadata.jsonl').write_text('')
            archive = root / 'source.zip'
            buffer = io.BytesIO()
            Image.fromarray(np.random.default_rng(13).integers(0, 256, (224, 224, 3), dtype=np.uint8)).save(buffer, format='JPEG')
            with zipfile.ZipFile(archive, 'w') as handle:
                handle.writestr('repo/data/images/test/class_1_0.JPG', buffer.getvalue())
                handle.writestr('repo/data/labels/test/class_1_0.txt', '1 0.5 0.5 0.6 0.6')
            result = enrich(base, archive, root / 'output')
            self.assertEqual(result['added'], 1)
            self.assertEqual((root / 'output/valid/dragon/original.jpg').read_bytes(), original)
            row = json.loads((root / 'output/metadata.jsonl').read_text())
            self.assertEqual(row['split'], 'train')
            self.assertEqual(row['label'], 'dragon')
            self.assertIsNone(row['participant_id'])
            with self.assertRaises(ValueError):
                enrich(base, archive, root / 'output')

    def test_held_out_near_duplicate_is_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / 'base'
            (base / 'valid' / 'dragon').mkdir(parents=True)
            Image.fromarray(np.random.default_rng(42).integers(0, 256, (224, 224, 3), dtype=np.uint8)).save(base / 'valid/dragon/original.PNG')
            (base / 'metadata.jsonl').write_text('')
            archive = root / 'source.zip'
            with zipfile.ZipFile(archive, 'w') as handle:
                handle.writestr('repo/data/images/train/class_1_0.png', (base / 'valid/dragon/original.PNG').read_bytes())
                handle.writestr('repo/data/labels/train/class_1_0.txt', '1 0.5 0.5 0.6 0.6')
            result = enrich(base, archive, root / 'output')
            self.assertEqual(result['added'], 0)
            self.assertEqual(result['excluded_duplicates_or_held_out_neighbors'], 1)
