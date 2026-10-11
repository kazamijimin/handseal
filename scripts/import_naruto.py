import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from PIL import Image

root = Path('local_setup/datasets/naruto_kaggle')
archive_path = root / 'naruto-hand-sign-dataset.zip'
output = root / 'extracted'
output.mkdir(exist_ok=True)
labels = {'dragon', 'ram', 'snake', 'tiger', 'zero'}
all_counts = Counter()
selected_counts = Counter()
records = []
existing = {}
existing_names = set()
def normalized_name(name):
    stem = Path(name).stem
    stem = re.sub(r'\.rf\.[0-9a-f]+$', '', stem, flags=re.I)
    return re.sub(r'_jpg$', '', stem, flags=re.I)
for path in Path('handseal_dataset').rglob('*.jpg'):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    existing.setdefault(digest, []).append(path.as_posix())
    existing_names.add(normalized_name(path.name))
exact_overlap = 0
source_name_overlap = 0
with zipfile.ZipFile(archive_path) as archive:
    documentation = [m.filename for m in archive.infolist() if not m.is_dir() and ('license' in m.filename.lower() or 'readme' in m.filename.lower())]
    for member in archive.infolist():
        if member.is_dir():
            continue
        relative = PurePosixPath(member.filename)
        assert not relative.is_absolute() and '..' not in relative.parts
        is_doc = member.filename in documentation
        if relative.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            if is_doc:
                destination = output.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(member))
            continue
        label = relative.parent.name.lower()
        split = relative.parent.parent.name.lower()
        all_counts[f'{split}/{label}'] += 1
        if label not in labels:
            continue
        data = archive.read(member)
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            width, height = image.size
        destination = output.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        overlaps = existing.get(digest, [])
        exact_overlap += bool(overlaps)
        source_name_overlap += normalized_name(relative.name) in existing_names
        records.append({'path': relative.as_posix(), 'original_label': label, 'original_split': split, 'width': width, 'height': height, 'sha256': digest, 'participant_id': None, 'session_id': None, 'source_url': 'https://www.kaggle.com/datasets/vikranthkanumuru/naruto-hand-sign-dataset', 'license_status': 'Unknown in Kaggle metadata', 'exact_matches_in_existing_repo': overlaps})
        selected_counts[f'{split}/{label}'] += 1
(output / 'provenance.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
report = {'archive_bytes': archive_path.stat().st_size, 'all_image_counts': dict(sorted(all_counts.items())), 'extracted_image_counts': dict(sorted(selected_counts.items())), 'extracted_images': len(records), 'exact_overlap_with_repo': exact_overlap, 'source_filename_overlap_with_repo': source_name_overlap, 'documentation_members': documentation, 'note': 'Four target sign labels extracted, plus zero if present for label review. Original folder labels retained. Participant and session metadata unavailable, left null. The full raw archive retains the other sign classes.'}
(root / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
