"""Add collected training photos while retaining the existing public evaluation splits."""
import argparse
import json
import shutil
from pathlib import Path

from data_tools.audit_dataset import audit_dataset, markdown_report
from data_tools.metadata import METADATA_FIELDS, load_metadata
from data_tools.prepare_external_dataset import baseline_issues, image_record
from training.config import DEFAULT_CLASSES
from training.datasets import IMAGE_EXTENSIONS


def prepare(base, source, output):
    if output.exists():
        raise ValueError(f'Refusing to overwrite {output}')
    original = audit_dataset(base, DEFAULT_CLASSES)
    if baseline_issues(original):
        raise ValueError(f'Base dataset failed audit: {baseline_issues(original)}')
    metadata, errors = load_metadata(source)
    if errors:
        raise ValueError(f'Invalid collected metadata: {errors}')
    existing_hashes = {row['sha256'] for row in original['records']}
    held_hashes = [int(row['perceptual_hash'], 16) for row in original['records']
                   if row['split'] in ('valid', 'test')]
    candidates, excluded, duplicate = [], 0, 0
    for label in DEFAULT_CLASSES:
        directory = source / 'train' / label
        for path in sorted(directory.glob('*')):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            record = image_record(path, label, 'handseal_collected')
            if record['sha256'] in existing_hashes:
                duplicate += 1
                continue
            if any((record['perceptual_hash'] ^ value).bit_count() <= 4 for value in held_hashes):
                excluded += 1
                continue
            existing_hashes.add(record['sha256'])
            relative = path.relative_to(source).as_posix()
            candidates.append((record, metadata.get(relative, {}), relative))
    shutil.copytree(base, output)
    new_rows = []
    for record, capture, relative in candidates:
        destination = output / 'train' / record['label'] / ('handseal_collected_' + record['sha256'] + '.jpg')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(record['image_bytes'])
        new_rows.append({'path':destination.relative_to(output).as_posix(), 'split':'train',
                         'label':record['label'], 'width':224, 'height':224, 'sha256':record['sha256'],
                         **{field:capture.get(field) for field in METADATA_FIELDS},
                         'source_dataset':'handseal_collected', 'source_key':record['source_key'],
                         'source_path':relative, 'split_policy':'collected_train_only_public_holdouts',
                         'captured_at':capture.get('captured_at')})
    with (output / 'metadata.jsonl').open('a', encoding='utf-8') as handle:
        for row in new_rows:
            handle.write(json.dumps(row)+'\n')
    # Remove new-source members of any transitive cross-split similarity group as well.
    report = audit_dataset(output, DEFAULT_CLASSES)
    new_paths = {row['path'] for row in new_rows}
    remove = {row['path'] for group in report['cross_split_near_duplicate_groups'] for row in group
              if row['path'] in new_paths}
    if remove:
        for relative in remove:
            (output / relative).unlink()
        rows = [json.loads(line) for line in (output / 'metadata.jsonl').read_text().splitlines()]
        (output / 'metadata.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows
                                                       if row['path'] not in remove), encoding='utf-8')
        report = audit_dataset(output, DEFAULT_CLASSES)
    issues = baseline_issues(report)
    (output / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (output / 'audit.md').write_text(markdown_report(report), encoding='utf-8')
    result = {'added':len(new_rows)-len(remove), 'excluded_exact_duplicates':duplicate,
              'excluded_held_out_similarity':excluded+len(remove), 'counts':report['counts'],
              'baseline_issues':issues, 'policy':'collected train only; original public valid/test unchanged'}
    (output / 'preparation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    if issues:
        raise ValueError(f'Combined dataset failed audit: {issues}')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('handseal_dataset_external_v3'))
    parser.add_argument('--source', type=Path, default=Path('handseal_collected'))
    parser.add_argument('--output', type=Path, default=Path('handseal_dataset_external_collected_v1'))
    args = parser.parse_args()
    print(json.dumps(prepare(args.base, args.source, args.output), indent=2))
