import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

root = Path('local_setup/datasets/hagrid')
output = root / 'extracted'
output.mkdir(exist_ok=True)
annotations = {}
subject_splits = defaultdict(set)
with zipfile.ZipFile(root / 'annotations.zip') as archive:
    for upstream, split in [('train', 'train'), ('val', 'valid'), ('test', 'test')]:
        content = archive.read(f'annotations/{upstream}/no_gesture.json')
        (root / f'no_gesture_{split}.json').write_bytes(content)
        values = json.loads(content)
        for image_id, row in values.items():
            assert image_id not in annotations
            assert row['user_id']
            assert set(row['labels']) == {'no_gesture'}, row['labels']
            annotations[image_id] = (split, row)
            subject_splits[row['user_id']].add(split)
assert all(len(splits) == 1 for splits in subject_splits.values()), 'Subject overlap'
counts = Counter()
records = []
seen = set()
with zipfile.ZipFile(root / 'no_gesture.zip') as archive:
    for member in archive.infolist():
        if member.is_dir() or '.ipynb_checkpoints' in Path(member.filename).parts or Path(member.filename).suffix.lower() != '.jpg':
            continue
        name = Path(member.filename).name
        image_id = Path(name).stem
        assert image_id in annotations, image_id
        assert image_id not in seen
        seen.add(image_id)
        split, row = annotations[image_id]
        data = archive.read(member)
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            width, height = image.size
        destination = output / split / 'no_seal' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        records.append({'path': destination.relative_to(output).as_posix(), 'label': 'no_seal', 'original_label': 'no_gesture', 'split': split, 'participant_id': 'hagrid:' + row['user_id'], 'session_id': None, 'width': width, 'height': height, 'sha256': hashlib.sha256(data).hexdigest(), 'source_dataset': 'HaGRIDv2', 'source_url': 'https://github.com/hukenovs/hagrid', 'annotation': row})
        counts[split] += 1
assert seen == set(annotations), 'Missing annotated images'
(output / 'provenance.jsonl').write_text(''.join(json.dumps(record) + '\n' for record in records), encoding='utf-8')
summary = {'total_images': len(records), 'counts': dict(counts), 'subjects': len(subject_splits), 'subject_overlap': 0, 'unreadable_images': 0, 'mapping': {'no_gesture': 'no_seal'}, 'note': 'Candidate negative samples; natural hand postures, not comprehensive no-hand or transition coverage. Missing source session/environment metadata remain null. Raw images are unchanged.'}
(root / 'verification.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))

