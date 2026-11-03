"""Build typed lane-boundary corridor data from ORIGINAL BDD train/val JSON.

The auxiliary ll_seg package is deliberately not the target: it also contains
curbs/transverse markings, and its val images belong to original BDD train.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from lane_boundary_segmentation import render_target

ROOT = Path(__file__).resolve().parent
BDD = ROOT / 'datasets/adas/raw/BDD100k'
OUT = ROOT / 'runs/lane_segmentation_20261017'
LABELS = BDD / 'bdd100k_labels/bdd100k/labels/100k'
IMAGES = BDD / 'bdd100k_images/bdd100k/images/100k'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f'Cannot encode {path}')
    path.write_bytes(encoded.tobytes())


def prepare(train_images=800, val_images=160):
    config = dict(version=2, train_images=train_images, val_images=val_images, size=[512, 288], corridor_width=5,
                  target_code_sha256=hashlib.sha256((ROOT / 'lane_boundary_segmentation.py').read_bytes()).hexdigest(),
                  split='original BDD train / original BDD val', ignore_label=255)
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    target = ROOT / 'datasets/adas/prepared/lane_boundary_segmentation_v1' / fingerprint
    manifest_path = target / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if all((target / row[key]).is_file() for row in manifest['records'] for key in ('image', 'mask')):
            return target, manifest
    records, statistics = [], {}
    hashes = {'train': set(), 'val': set()}
    for split, limit in (('train', train_images), ('val', val_images)):
        files = sorted((LABELS / split).glob('*.json'), key=lambda path: hashlib.sha256(path.name.encode()).hexdigest())
        if len(files) < limit:
            raise RuntimeError('Not enough original BDD files')
        categories, pixels = Counter(), Counter()
        rejected, accepted = [], 0
        for path in files:
            raw = path.read_bytes()
            label = json.loads(raw)
            source = IMAGES / split / (label['name'].removesuffix('.jpg') + '.jpg')
            image_bytes = source.read_bytes()
            digest = hashlib.sha256(image_bytes).hexdigest()
            other = 'train' if split == 'val' else 'val'
            if digest in hashes[other]:
                raise RuntimeError('Image content duplicated across original train/val selection')
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f'Cannot decode original image {source}')
            try:
                mask, counts = render_target(label['frames'][0]['objects'], image.shape, tuple(config['size']), config['corridor_width'])
            except ValueError as exc:
                rejected.append(dict(source_json=str(path), reason=str(exc)))
                continue
            if counts['invalid_paths']:
                rejected.append(dict(source_json=str(path), reason='invalid_lane_path'))
                continue
            hashes[split].add(digest)
            categories.update(counts)
            unique, numbers = np.unique(mask, return_counts=True)
            pixels.update(dict(zip(map(str, unique), map(int, numbers))))
            image_path = f'images/{split}/{path.stem}.png'
            mask_path = f'masks/{split}/{path.stem}.png'
            write_image(target / image_path, cv2.resize(image, tuple(config['size'])))
            write_image(target / mask_path, mask)
            records.append(dict(split=split, image=image_path, mask=mask_path, source_image=str(source),
                                source_json=str(path), source_label_sha256=hashlib.sha256(raw).hexdigest(),
                                source_image_sha256=digest, lane_counts=counts, has_visible_target=bool(((mask == 1) | (mask == 2)).any())))
            accepted += 1
            if accepted % 100 == 0:
                print('BOUNDARY_DATA', split, accepted, '/', limit, 'quarantined', len(rejected), flush=True)
            if accepted >= limit:
                break
        if accepted != limit:
            raise RuntimeError('Not enough valid original BDD image/label pairs')
        if not pixels['1'] or not pixels['2']:
            raise RuntimeError('Missing a true lane style in the selected split')
        statistics[split] = dict(images=accepted, lane_counts=dict(categories), pixels=dict(pixels), quarantined=rejected)
    manifest = dict(config=config, fingerprint=fingerprint, records=records, statistics=statistics,
                    names=['background', 'solid_boundary', 'dashed_boundary'],
                    source_format_reference='https://github.com/ucbdrive/bdd100k/blob/master/bdd100k/show_labels.py',
                    scope='Typed continuous boundary corridors, not exact painted pixels. Excluded lane types, '
                          'overlapping styles and annotated vehicles/persons are ignored. Official clips not used.')
    save(manifest_path, manifest)
    # Fixed first examples, including negatives; not selected for prediction quality.
    panels = []
    for split in ('train', 'val'):
        for row in [r for r in records if r['split'] == split][:4]:
            image = cv2.imdecode(np.fromfile(target / row['image'], np.uint8), cv2.IMREAD_COLOR)
            mask = cv2.imdecode(np.fromfile(target / row['mask'], np.uint8), cv2.IMREAD_GRAYSCALE)
            for label, color in ((1, (0, 180, 255)), (2, (80, 230, 80))):
                chosen = mask == label
                image[chosen] = (.3 * image[chosen] + .7 * np.array(color)).astype(np.uint8)
            cv2.putText(image, split + ' ' + Path(row['image']).stem, (8, 22), 0, .45, (255, 255, 255), 1)
            panels.append(image)
    write_image(target / 'target_review.jpg', np.vstack([np.hstack(panels[i:i + 2]) for i in range(0, len(panels), 2)]))
    return target, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-images', type=int, default=800)
    parser.add_argument('--val-images', type=int, default=160)
    args = parser.parse_args()
    if min(args.train_images, args.val_images) < 1:
        parser.error('Image counts must be positive')
    cv2.setNumThreads(1)
    dataset, manifest = prepare(args.train_images, args.val_images)
    save(OUT / 'latest_data.json', dict(dataset=str(dataset), manifest=str(dataset / 'manifest.json')))
    print('BOUNDARY_DATA_READY', dataset, manifest['statistics'], flush=True)


if __name__ == '__main__':
    main()
