"""Prepare unlabelled review proposals from unused real sources only.

No source annotation is relabelled as countdown/camera by this script. Explicit
visual decisions must be stored separately before any proposal enters training.
"""
from collections import defaultdict
import json
import math
from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageOps

from light_roi_common import ROOT
from prepare_type_direction_repair_pycharm import digest, save

OUT = ROOT / 'runs/signal_hard_examples_20261022'
DOMESTIC = ROOT / 'runs/domestic_direction_20261022'
PRIOR = ROOT / 'runs/signal_type_direction_repair_20261022'


def panels(rows, prefix, columns, per_page, size):
    cw, ch = size
    for start in range(0, len(rows), per_page):
        subset = rows[start:start+per_page]
        canvas = Image.new('RGB', (columns*cw, math.ceil(len(subset)/columns)*ch), '#252930')
        draw = ImageDraw.Draw(canvas)
        for j, row in enumerate(subset):
            with Image.open(row['proposal']) as im:
                thumb = ImageOps.contain(im.convert('RGB'), (cw-10, ch-40), Image.Resampling.NEAREST)
            x, y = j%columns*cw, j//columns*ch
            canvas.paste(thumb, (x+(cw-thumb.width)//2, y+3))
            draw.text((x+4, y+ch-34), f"{row['id']} {row['role']} {row['group']}", fill='white')
            draw.text((x+4, y+ch-18), row['display'], fill='#ccddee')
        canvas.save(OUT / f'{prefix}_{start//per_page:02d}.jpg', quality=96)


def main():
    OUT.mkdir(exist_ok=True)
    target = OUT / 'unlabelled_proposals.json'
    if target.exists():
        print('HARD_PROPOSALS_ALREADY_READY', flush=True)
        return
    existing = json.loads((PRIOR / 'manifest.json').read_text(encoding='utf8'))['records']
    used_sources = {str(Path(r['source_image']).resolve()).lower() for r in existing}
    inventory = json.loads((DOMESTIC / 'source_inventory.json').read_text(encoding='utf8'))
    split = json.loads((DOMESTIC / 'recording_split.json').read_text(encoding='utf8'))['recordings']
    assignments = {g: ('train' if role == 'train' else 'new_calibration' if role == 'calibration' else 'new_review')
                   for role, groups in split.items() for g in groups}
    by_group = defaultdict(list)
    for row in inventory['frames']:
        if str(Path(row['image']).resolve()).lower() not in used_sources:
            by_group[row['group']].append(row)
    folder = OUT / 'proposals'; folder.mkdir(exist_ok=True)
    contexts = []
    # Large author-annotated heads tend to provide legible nearby countdowns.
    for group in sorted(by_group):
        frames = sorted(by_group[group], key=lambda r: -max((b['xywh'][2]*b['xywh'][3] for b in r['boxes']), default=0))
        if not frames:
            continue
        row = frames[0]
        with Image.open(row['image']) as im:
            im = im.convert('RGB'); w, h = im.size
            boxes = [((b['xywh'][0]-b['xywh'][2]/2)*w, (b['xywh'][1]-b['xywh'][3]/2)*h,
                      (b['xywh'][0]+b['xywh'][2]/2)*w, (b['xywh'][1]+b['xywh'][3]/2)*h) for b in row['boxes']]
            if not boxes:
                continue
            region = [max(0, math.floor(min(b[0] for b in boxes)-180)), max(0, math.floor(min(b[1] for b in boxes)-120)),
                      min(w, math.ceil(max(b[2] for b in boxes)+180)), min(h, math.ceil(max(b[3] for b in boxes)+160))]
            index = len(contexts); path = folder / f'context_{index:03d}.png'
            im.crop(region).save(path)
        contexts.append(dict(id=f'context_{index:03d}', proposal=str(path), source_image=row['image'],
             role=assignments[group], group=group, context_xyxy=region, display=Path(row['image']).name,
             source_label=row['label'], source_label_sha256=row['label_sha256'], label_status='unlabelled'))
    panels(contexts, 'context_review', 3, 12, (440, 300))
    # S2 author train only. These source class IDs are NOT countdown/type truth.
    source = ROOT / 'datasets/adas/prepared/s2tld_light_states'
    candidates = []
    for label in sorted((source / 'labels/train').glob('*.txt')):
        image = source / 'images/train' / (label.stem+'.jpg')
        if str(image.resolve()).lower() in used_sources:
            continue
        for number, line in enumerate(label.read_text(encoding='utf8').splitlines(), 1):
            fields = line.split()
            if len(fields) != 5:
                continue
            c, x, y, w, h = map(float, fields)
            if .65 <= w/h <= 3.0 and w*h >= .00045:
                candidates.append(dict(source_image=str(image), source_label=str(label), line=number, xywh=[x,y,w,h]))
    random.Random(20261023).shuffle(candidates)
    rois = []; seen = set()
    for row in candidates:
        if row['source_image'] in seen:
            continue
        with Image.open(row['source_image']) as im:
            im = im.convert('RGB'); iw, ih = im.size
            x, y, w, h = row['xywh']; pad = max(2, round(max(w*iw,h*ih)*.08))
            box = [max(0, math.floor((x-w/2)*iw)-pad), max(0, math.floor((y-h/2)*ih)-pad),
                   min(iw, math.ceil((x+w/2)*iw)+pad), min(ih, math.ceil((y+h/2)*ih)+pad)]
            if min(box[2]-box[0], box[3]-box[1]) < 16:
                continue
            index = len(rois); path = folder / f's2_{index:03d}.png'
            im.crop(box).save(path)
        seen.add(row['source_image'])
        rois.append(dict(**row, id=f's2_{index:03d}', proposal=str(path), source_box=box,
             role='train', group='S2_author_train_recording_unknown', display=Path(row['source_image']).name,
             label_status='unlabelled'))
        if len(rois) >= 112:
            break
    panels(rois, 's2_roi_review', 8, 56, (160, 165))
    save(target, dict(records=contexts+rois, proposal_counts=dict(domestic_context=len(contexts), s2_roi=len(rois)),
         prior_manifest_sha256=digest(PRIOR / 'manifest.json'),
         domestic_inventory_sha256=digest(DOMESTIC / 'source_inventory.json'),
         recording_assignment_sha256=digest(DOMESTIC / 'recording_split.json'),
         explicit_visual_labels_required=True, source_annotations_are_not_new_type_labels=True,
         all_selected_paths_unused_in_prior_manifest=True, final_pixel_deduplication_pending=True,
         s2_recording_identity_unknown=True, new_review_recordings_already_known_to_prior_evaluation=True))
    print('HARD_PROPOSALS_READY', len(contexts), len(rois), flush=True)


if __name__ == '__main__':
    main()
