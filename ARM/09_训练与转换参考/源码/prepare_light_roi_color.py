"""Create traceable lamp crops using existing train/val splits; never alter labels."""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path

from light_roi_common import ROOT, NAMES, read_image, write_image, crop_head
from official_light_config import POOL, S2TLD, load_manifest
from official_light_tiles import read_boxes

VERSION = 'light_roi_color_v1_padding008'


def prepare():
    sources = []
    for split, target in [('train', 'train'), ('val', 's2_val')]:
        for image in sorted((S2TLD/f'images/{split}').glob('*.jpg')):
            sources.append({'image': image, 'label': S2TLD/f'labels/{split}/{image.stem}.txt',
                            'split': target, 'domain': 's2tld', 'source_split': split})
    for row in load_manifest()['images']:
        if row['status'] == 'reviewed' and row['split'] in {'train', 'val'}:
            image = POOL/row['image']
            sources.append({'image': image, 'label': POOL/f"labels/{row['split']}/{row['stem']}.txt",
                            'split': 'train' if row['split'] == 'train' else 'official_val',
                            'domain': 'official', 'source_split': row['split'],
                            'video': row['video'], 'frame': row['frame']})
    fingerprint = hashlib.sha256(VERSION.encode())
    fingerprint.update(json.dumps(NAMES).encode())
    for row in sources:
        assert row['image'].is_file() and row['label'].is_file(), row
        fingerprint.update(str(row['image'].resolve()).encode())
        fingerprint.update(str((row['image'].stat().st_size, row['image'].stat().st_mtime_ns)).encode())
        fingerprint.update(row['label'].read_bytes())
    identity = fingerprint.hexdigest()
    output = ROOT/'datasets/adas/prepared/light_roi_color_v1'/identity[:12]
    manifest_path = output/'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        assert manifest['fingerprint'] == identity and all((output/r['image']).is_file() for r in manifest['records'])
        print('DATASET_REUSED', output, flush=True)
        return output, manifest
    output.mkdir(parents=True, exist_ok=False)
    records, skipped, excluded, hashes = [], [], [], {}
    counts = {}
    for index, row in enumerate(sources):
        image = read_image(row['image'])
        boxes = read_boxes(row['label'], image.shape[1], image.shape[0])
        source_hash = hashlib.sha256(image.tobytes()).hexdigest()
        for box_index, box in enumerate(boxes):
            cls, *xyxy = box
            try:
                crop = crop_head(image, xyxy)
            except ValueError as exc:
                skipped.append({'source': str(row['image']), 'box_index': box_index, 'reason': str(exc)})
                continue
            digest = hashlib.sha256(str(crop.shape).encode()+crop.tobytes()).hexdigest()
            if digest in hashes:
                old = hashes[digest]
                if old['class_id'] != cls:
                    raise ValueError(f'Conflicting labels for identical pixels: {row["image"]}')
                if old['split'] != row['split']:
                    raise ValueError(f'Exact duplicate crop crosses splits: {row["image"]}')
                excluded.append({'source': str(row['image']), 'box_index': box_index, 'duplicate_of': old['image']})
                continue
            name = f"images/{row['split']}/{row['domain']}_{row['image'].stem}_b{box_index:02d}.png"
            write_image(output/name, crop)
            record = {'image': name, 'split': row['split'], 'domain': row['domain'],
                      'class_id': cls, 'name': NAMES[cls], 'source_image': str(row['image'].resolve()),
                      'source_split': row['source_split'], 'source_pixel_sha256': source_hash,
                      'source_box': xyxy, 'box_index': box_index, 'crop_pixel_sha256': digest,
                      'crop_size': [crop.shape[1], crop.shape[0]]}
            if 'video' in row:
                record.update(video=row['video'], frame=row['frame'])
            records.append(record)
            hashes[digest] = record
            counts.setdefault(row['split'], Counter())[NAMES[cls]] += 1
        if (index+1) % 300 == 0:
            print(f'PREPARE {index+1}/{len(sources)} sources; {len(records)} unique crops', flush=True)
    train_sources = {r['source_pixel_sha256'] for r in records if r['split'] == 'train'}
    val_sources = {r['source_pixel_sha256'] for r in records if r['split'] != 'train'}
    assert not train_sources & val_sources, 'Identical original image crosses training/validation'
    manifest = {'version': VERSION, 'fingerprint': identity, 'names': NAMES,
                'created_local': datetime.now().isoformat(), 'records': records,
                'counts': {s: dict(c) for s,c in counts.items()}, 'skipped': skipped,
                'exact_duplicates_removed': excluded, 'source_images': len(sources),
                'scope': 'Existing source splits preserved; no original labels changed; no test images read. '
                         'Official val is reviewed development material, not independent final testing.'}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (output/'COMPLETE').write_text(identity, encoding='utf-8')
    print(json.dumps({'dataset': str(output), 'counts': manifest['counts'],
                      'skipped': len(skipped), 'duplicates_removed': len(excluded)}, indent=2), flush=True)
    return output, manifest


if __name__ == '__main__':
    prepare()
