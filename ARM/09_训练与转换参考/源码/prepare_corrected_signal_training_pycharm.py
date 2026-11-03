"""Reviewed label correction + real timer/camera/small-arrow crop additions.

The sealed earlier datasets are unchanged. The eleven former 'pure display'
labels are superseded here: eight contain lit round elements; three remain
ambiguous whole housings and are excluded, not recycled as pure-digit truth.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

from PIL import Image

from prepare_signal_hard_examples_pycharm import ROOT, OUT, PRIOR, DOMESTIC, panels
from prepare_type_direction_repair_pycharm import save, digest

CORRECT_TO_MOTOR = {'repair_1614','repair_1616','repair_1617','repair_1618','repair_1628','repair_1629','repair_1630','repair_1640'}
EXCLUDE_MIXED_UNKNOWN = {'repair_1613','repair_1636','repair_1637'}
TRAIN_LEFT = {18,19,23,25,26,27,29,30,32,34}
TRAIN_STRAIGHT = {37,39,40,42,43,44,45,46,48,51,52,53,54}
CALIB_LEFT = {1,2,3,4,6}
CALIB_STRAIGHT = {8,9}
CAMERA_REJECT = {'camera_003','camera_004','camera_005'}
TIMER_TOP_TRIM = {'timer_035':4, 'timer_047':5, 'timer_069':3, 'timer_070':4, 'timer_071':2}
BICYCLE_WHOLE = {'detail_053','detail_062','detail_073'}


def rgb_digest(path):
    with Image.open(path) as im:
        return hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()


def main():
    target=OUT/'corrected_manifest.json'
    if target.exists():
        result=json.loads(target.read_text(encoding='utf8'))
        for p,h in result['inputs'].items(): assert digest(p)==h
        print('CORRECTED_SIGNAL_DATA_ALREADY_READY',result['counts'],flush=True)
        return result
    previous=json.loads((PRIOR/'manifest.json').read_text(encoding='utf8'))
    regions=json.loads((OUT/'region_proposals.json').read_text(encoding='utf8'))['records']
    parents=json.loads((OUT/'readable_housings.json').read_text(encoding='utf8'))['records']
    parents+=json.loads((OUT/'readable_housings_validation.json').read_text(encoding='utf8'))['records']
    parent_by_id={r['id']:r for r in parents}
    rows=[];corrections=[];exclusions=[]
    old_digit={r['id'] for r in previous['records'] if r['type_id']==3 and r['origin']=='corrected_small_type_review'}
    assert old_digit==CORRECT_TO_MOTOR|EXCLUDE_MIXED_UNKNOWN
    for index,row in enumerate(previous['records']):
        row=dict(row,cache_index=index)
        if row['id'] in EXCLUDE_MIXED_UNKNOWN:
            exclusions.append(dict(id=row['id'],image=row['image'],role=row['role'],reason='Whole housing/digit detail ambiguous; cannot be a pure-digit negative'))
            continue
        if row['id'] in CORRECT_TO_MOTOR:
            corrections.append(dict(id=row['id'],image=row['image'],role=row['role'],previous_type=3,corrected_type=0,
                 reason='Native nearest-neighbor review visibly contains active circular signal lens; entire housing is not a pure display'))
            row.update(type_id=0,direction_id=0,target_origin='2026-10-22 native audit: lit round element in whole vehicle housing',
                 label_corrected=True,previous_type_id=3)
        rows.append(row)
    folder=OUT/'reviewed_crops';folder.mkdir(exist_ok=True)
    fresh=[];whole_seen=set()
    def append_crop(candidate,kind,type_id,direction_id=-100):
        with Image.open(candidate['source_image']) as im:
            box=list(candidate['source_box']);box[1]+=TIMER_TOP_TRIM.get(candidate['id'],0)
            crop=im.convert('RGB').crop(box)
            path=folder/(candidate['id']+'.png');crop.save(path)
        row=dict(id='new_'+candidate['id'],image=str(path),source_image=candidate['source_image'],source_box=box,
             role=candidate['role'],recording_group=candidate['group'],type_id=type_id,direction_id=direction_id,
             origin='native_review_20261022',other_kind=kind,synthetic=False,
             label_origin='Explicit review of native crop and parent context; proposals never used as automatic labels',
             proposal_id=candidate['id'],image_sha256=digest(path))
        fresh.append(row)
    for candidate in regions:
        id=candidate['id']
        if id.startswith('timer_'):
            append_crop(candidate,'pure_countdown_region',3)
            p=parent_by_id[candidate['parent_id']]
            if p['id'] not in whole_seen:
                p=dict(p,role=candidate['role'])
                # A whole housing with digits remains motor/bicycle; its active
                # direction is not guessed from dark circular lenses.
                append_crop(p,'paired_whole_bicycle' if p['id'] in BICYCLE_WHOLE else 'paired_whole_motor',2 if p['id'] in BICYCLE_WHOLE else 0)
                whole_seen.add(p['id'])
        elif id.startswith('camera_'):
            if id in CAMERA_REJECT:
                exclusions.append(dict(id=id,image=candidate['proposal'],role=candidate['role'],reason='Transferred ROI includes sign/support; camera body not isolated'))
                continue
            append_crop(candidate,'camera_device',3)
        elif id.startswith('arrow_'):
            n=int(id.split('_')[1]);di=-100
            if candidate['role']=='train':
                if n in TRAIN_LEFT:di=1
                elif n in TRAIN_STRAIGHT:di=2
            else:
                if n in CALIB_LEFT:di=1
                elif n in CALIB_STRAIGHT:di=2
            append_crop(candidate,'clear_small_arrow' if di>=0 else 'motor_housing_direction_uncertain',0,di)
    # Five more visibly explicit bicycle pictograms improve the scarce class.
    for id in ['detail_049','detail_059']:
        append_crop(parent_by_id[id],'whole_bicycle_pictogram',2)
    paths=sorted({r['source_image'] for r in fresh})
    print('HARD_EXAMPLE_PIXEL_AUDIT',len(paths),'sources',flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        hashes=dict(zip(paths,pool.map(rgb_digest,paths)))
    old_train={r['source_rgb_sha256'] for r in rows if r['role']=='train'}
    old_held={r['source_rgb_sha256'] for r in rows if r['role']!='train'}
    fresh=[dict(r,source_rgb_sha256=hashes[r['source_image']]) for r in fresh]
    allowed=[]
    new_held={r['source_rgb_sha256'] for r in fresh if r['role']!='train'}
    for r in fresh:
        blocked=(r['role']=='train' and r['source_rgb_sha256'] in old_held|new_held) or (r['role']!='train' and r['source_rgb_sha256'] in old_train)
        if blocked:
            exclusions.append(dict(id=r['id'],image=r['image'],role=r['role'],reason='Source RGB overlaps opposite role or prior training'))
            continue
        allowed.append(r)
    by_hash={r['image_sha256']:r for r in rows}
    for r in allowed:
        if r['image_sha256'] in by_hash:
            old=by_hash[r['image_sha256']]
            assert old['type_id']==r['type_id'],'Conflicting identical-crop type truth'
            exclusions.append(dict(id=r['id'],image=r['image'],role=r['role'],reason='Identical crop already retained'))
            continue
        by_hash[r['image_sha256']]=r;rows.append(r)
    train={r['source_rgb_sha256'] for r in rows if r['role']=='train'}
    held={r['source_rgb_sha256'] for r in rows if r['role']!='train'}
    assert not train&held
    assert not any(r.get('synthetic') for r in rows)
    # Keep the known 18 diagnostic source images outside every TRAIN crop.
    audit=json.loads((ROOT/'runs/direction_path_relation_20261021/visual_direction_audit.json').read_text(encoding='utf8'))['records']
    protected={rgb_digest(r['image']) for r in audit if r['direction_label'] is not None}
    assert not train&protected
    counts={role:dict(Counter(str(r['type_id']) for r in rows if r['role']==role)) for role in sorted({r['role'] for r in rows})}
    fresh_counts={role:dict(Counter(r['other_kind'] for r in rows if r['role']==role and r['origin']=='native_review_20261022')) for role in counts}
    input_paths=[PRIOR/'manifest.json',OUT/'region_proposals.json',OUT/'readable_housings.json',OUT/'readable_housings_validation.json',DOMESTIC/'recording_split.json']
    result=dict(records=rows,counts=counts,fresh_counts=fresh_counts,corrections=corrections,exclusions=exclusions,
         preparer_sha256=digest(Path(__file__)),
         inputs={str(p):digest(p) for p in input_paths},source_pixel_leakage=False,
         former_digit_labels_superseded=11,corrected_vehicle_housings=8,excluded_ambiguous_housings=3,
         manual_timer_top_trims=TIMER_TOP_TRIM,prior_metrics_with_digit_truth_no_longer_valid=True,
         fresh_right_arrows_added=0,uncertain_arrow_directions_never_guessed=True,
         s2_recording_independence_unverified=True,all_validation_is_development_not_blind=True,
         physical_lane_association_labels=False)
    save(target,result)
    # Final-contact pages contain exactly the pixel rectangles used for training.
    reviewed=[dict(r,proposal=r['image'],group=r['recording_group'],display=r.get('other_kind','legacy')) for r in rows if r['origin']=='native_review_20261022']
    panels([r for r in reviewed if r.get('other_kind')=='pure_countdown_region'],'verified_timer',8,48,(160,130))
    panels([r for r in reviewed if r.get('other_kind')=='camera_device'],'verified_camera',8,48,(160,130))
    save(OUT/'annotation_review.json',dict(reviewer='Assistant native image review',reviewed_contacts=[
         'prior_digit_label_audit.jpg','readable_detail_review_00.jpg','readable_detail_review_01.jpg',
         'readable_detail_review_02.jpg','readable_detail_review_03.jpg','readable_val_review_00.jpg','readable_val_review_01.jpg',
         'timer_region_review_00.jpg','timer_region_review_01.jpg','camera_region_review_00.jpg','small_arrow_review_00.jpg','small_arrow_review_01.jpg'],
         corrected_old_ids=sorted(CORRECT_TO_MOTOR),excluded_old_ids=sorted(EXCLUDE_MIXED_UNKNOWN),
         camera_rejected=sorted(CAMERA_REJECT),train_left=sorted(TRAIN_LEFT),train_straight=sorted(TRAIN_STRAIGHT),
         calibration_left=sorted(CALIB_LEFT),calibration_straight=sorted(CALIB_STRAIGHT),
         scope='Native visual crop review; uncertain shape never promoted to direction truth; no new physical lane labels'))
    print('CORRECTED_SIGNAL_DATA_READY',counts,fresh_counts,flush=True)
    return result


if __name__=='__main__':main()
