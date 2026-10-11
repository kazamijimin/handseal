"""Add Lucas Fernando's public source photos to training only, preserving held-out splits."""
import argparse
import hashlib
import io
import json
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image
from data_tools.prepare_external_dataset import image_record
from training.datasets import IMAGE_EXTENSIONS

LABELS = {1: 'dragon', 4: 'snake', 8: 'tiger', 10: 'ram'}
SOURCE = 'lucas_fernando_github'


def average_hash(data):
    with Image.open(io.BytesIO(data)) as image:
        pixels = list(image.convert('L').resize((8, 8), Image.Resampling.BILINEAR).getdata())
    mean = sum(pixels) / len(pixels)
    return sum(int(value >= mean) << (63-index) for index, value in enumerate(pixels))


def enrich(base, archive, output):
    if output.exists():
        raise ValueError(f'Refusing to overwrite {output}')
    # Collect hashes before writing. Keep both existing evaluation splits byte-identical.
    held_out, existing = [], set()
    for path in base.glob('*/*/*'):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        data = path.read_bytes()
        existing.add(hashlib.sha256(data).hexdigest())
        if path.relative_to(base).parts[0] in ('valid', 'test'):
            held_out.append(average_hash(data))
    shutil.copytree(base, output)
    added, excluded = 0, 0
    rows = []
    with zipfile.ZipFile(archive) as handle, tempfile.TemporaryDirectory() as temporary:
        member_names = {name.casefold(): name for name in handle.namelist()}
        for name in sorted(handle.namelist()):
            member = PurePosixPath(name)
            if '/data/images/' not in name.lower() or member.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            annotation_key = name.lower().replace('/data/images/', '/data/labels/')
            annotation_key = str(PurePosixPath(annotation_key).with_suffix('.txt'))
            annotation_name = member_names[annotation_key.casefold()]
            annotations = handle.read(annotation_name).decode().splitlines()
            ids = {int(line.split()[0]) for line in annotations if line.strip()}
            if len(ids) != 1:
                raise ValueError(f'Ambiguous class annotation: {name}')
            label = LABELS.get(next(iter(ids)))
            if label is None:
                continue
            path = Path(temporary) / Path(name).name
            path.write_bytes(handle.read(name))
            record = image_record(path, label, SOURCE)
            digest, phash = record['sha256'], record['perceptual_hash']
            if digest in existing or any((phash ^ other).bit_count() <= 4 for other in held_out):
                excluded += 1
                continue
            destination = output / 'train' / label / (SOURCE + '_' + digest + '.jpg')
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(record['image_bytes'])
            existing.add(digest)
            rows.append({'path': destination.relative_to(output).as_posix(), 'label': label,
                         'split': 'train', 'width': 224, 'height': 224, 'sha256': digest,
                         'participant_id': None, 'session_id': None, 'lighting': None,
                         'background': None, 'distance': None, 'hand_position': None,
                         'source_dataset': SOURCE, 'source_path': name,
                         'source_key': record['source_key'], 'original_label': label,
                         'split_policy': 'entire_new_source_training_only'})
            added += 1
    with (output / 'metadata.jsonl').open('a', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')
    # Old audit/preparation reports describe the base dataset, so don't carry stale reports.
    for filename in ('audit.json', 'audit.md', 'preparation.json'):
        (output / filename).unlink(missing_ok=True)
    result = {'source': SOURCE, 'added': added, 'excluded_duplicates_or_held_out_neighbors': excluded,
              'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
              'source_repository': 'https://github.com/lucasfernandoprojects/hand-sign-detection',
              'repository_license': 'GPL-3.0; no separate image license found',
              'policy': 'all new-source samples are training only; original valid/test unchanged'}
    (output / 'enrichment.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('handseal_dataset_external_v2'))
    parser.add_argument('--archive', type=Path, default=Path('local_setup/datasets/lucas/source.zip'))
    parser.add_argument('--output', type=Path, default=Path('handseal_dataset_external_v3'))
    args = parser.parse_args()
    print(json.dumps(enrich(args.base, args.archive, args.output), indent=2))
