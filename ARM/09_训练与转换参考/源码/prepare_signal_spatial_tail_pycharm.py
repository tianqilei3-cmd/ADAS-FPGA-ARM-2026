"""Immutable corrected data plus TRAIN-only fixture proposals/readability labels."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path

from PIL import Image,ImageDraw,ImageOps

from light_roi_common import ROOT
from prepare_type_direction_repair_pycharm import digest,save
from prepare_domestic_direction_pycharm import recording_id

OUT=ROOT/'runs/signal_spatial_tail_20261022'
PREVIOUS=ROOT/'runs/signal_hard_examples_20261022'
SOURCE_MANIFEST=PREVIOUS/'corrected_manifest.json'


def rgb_digest(path):
    with Image.open(path) as im:return hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()


def fixture_proposals():
    OUT.mkdir(exist_ok=True)
    target=OUT/'fixture_proposals.json'
    if target.exists():return json.loads(target.read_text(encoding='utf8'))
    contexts=json.loads((PREVIOUS/'unlabelled_proposals.json').read_text(encoding='utf8'))['records'][:73]
    inventory=json.loads((ROOT/'runs/domestic_direction_20261022/source_inventory.json').read_text(encoding='utf8'))['frames']
    current=json.loads(SOURCE_MANIFEST.read_text(encoding='utf8'))['records']
    used={str(Path(r['source_image']).resolve()).lower() for r in current}
    folder=OUT/'fixture_proposals';folder.mkdir(exist_ok=True)
    rows=[]
    # Visually inspected non-signal hanging fixtures. Do not assert that these
    # uncertain camera/street-light subtypes are all surveillance cameras.
    for index,box in [(52,[133,57,158,84]),(60,[7,53,25,75])]:
        p=contexts[index];assert p['role']=='train'
        original=next(r for r in inventory if r['image']==p['source_image'])
        with Image.open(original['image']) as im:iw,ih=im.size
        def anchor(row):
            b=max(row['boxes'],key=lambda b:b['xywh'][2]*b['xywh'][3])['xywh']
            return [(b[0]-b[2]/2)*iw,(b[1]-b[3]/2)*ih,b[2]*iw,b[3]*ih]
        ax,ay,aw,ah=anchor(original);ox,oy=p['context_xyxy'][:2]
        relative=[(box[0]+ox-ax)/aw,(box[1]+oy-ay)/ah,(box[2]+ox-ax)/aw,(box[3]+oy-ay)/ah]
        frames=[r for r in inventory if r['group']==p['group'] and str(Path(r['image']).resolve()).lower() not in used]
        frames.sort(key=lambda r:abs(math.log(max(1,anchor(r)[2])/aw))+abs(math.log(max(1,anchor(r)[3])/ah)))
        count=0
        for frame in frames:
            x,y,w,h=anchor(frame)
            if abs(math.log(w/aw))>.15 or abs(math.log(h/ah))>.15:continue
            box=[max(0,math.floor(x+relative[0]*w)),max(0,math.floor(y+relative[1]*h)),
                 min(iw,math.ceil(x+relative[2]*w)),min(ih,math.ceil(y+relative[3]*h))]
            if min(box[2]-box[0],box[3]-box[1])<8:continue
            id=f'fixture_{len(rows):03d}';path=folder/(id+'.png')
            with Image.open(frame['image']) as im:im.convert('RGB').crop(box).save(path)
            rows.append(dict(id=id,image=str(path),source_image=frame['image'],source_box=box,
                 recording_group=p['group'],role='train',anchor_context=index,label_status='requires native review'))
            count+=1
            if count>=6:break
    canvas=Image.new('RGB',(960,math.ceil(len(rows)/6)*150),'#252930');draw=ImageDraw.Draw(canvas)
    for j,r in enumerate(rows):
        with Image.open(r['image']) as im:thumb=ImageOps.contain(im.convert('RGB'),(150,110),Image.Resampling.NEAREST)
        x,y=j%6*160,j//6*150;canvas.paste(thumb,(x+(160-thumb.width)//2,y+2))
        draw.text((x+3,y+115),r['id']+' ctx '+str(r['anchor_context']),fill='white')
        draw.text((x+3,y+131),r['recording_group'],fill='#dddddd')
    canvas.save(OUT/'fixture_proposal_review.jpg',quality=96)
    result=dict(records=rows,scope='Proposals only; camera subtype uncertain; no automatic supervision')
    save(target,result);return result


def prepare():
    target=OUT/'manifest.json'
    if target.exists():
        result=json.loads(target.read_text(encoding='utf8'))
        assert result['source_manifest_sha256']==digest(SOURCE_MANIFEST)
        assert result['preparer_sha256']==digest(Path(__file__))
        return result
    original=json.loads(SOURCE_MANIFEST.read_text(encoding='utf8'))
    decisions=json.loads((OUT/'fixture_visual_decisions.json').read_text(encoding='utf8'))
    proposals=fixture_proposals()['records'];rows=[dict(r) for r in original['records']]
    held={r['source_rgb_sha256'] for r in rows if r['role']!='train'}
    known_path=ROOT/'runs/direction_path_relation_20261021/visual_direction_audit.json'
    known=json.loads(known_path.read_text(encoding='utf8'))['records']
    protected={rgb_digest(r['image']) for r in known if r['direction_label'] is not None}
    for r in proposals:
        if r['id'] not in decisions['accepted_non_signal_fixture_ids']:continue
        sha=rgb_digest(r['source_image'])
        assert sha not in held|protected
        rows.append(dict(r,type_id=3,direction_id=-100,source_rgb_sha256=sha,image_sha256=digest(r['image']),
             origin='native_fixture_review',other_kind='non_signal_hanging_fixture',synthetic=False,
             label_origin='Native crop and full-context review: non-signal fixture; camera/street-light subtype not asserted'))
    # Readability uses explicit newly reviewed clear/uncertain direction labels,
    # plus legacy reviewed direction and clear right-arrow TRAIN schema crops.
    quality=[]
    for i,r in enumerate(rows):
        label=None
        if r.get('other_kind')=='clear_small_arrow':label=1
        if r.get('other_kind')=='motor_housing_direction_uncertain':label=0
        if r['role']=='train' and r['origin']=='final_conservative_attributes' and r['type_id']==0 and r['direction_id']>=0:label=1
        if label is not None:
            quality.append(dict(**r,main_index=i,readability_label=label,
                 readability_origin='Previously reviewed direction available' if r['origin']=='final_conservative_attributes' else 'Explicit native clear/uncertain shape review'))
    schema_path=ROOT/'runs/domestic_direction_20261022/schema_review/review_rows.json'
    seen={r['image_sha256'] for r in quality}
    for s in json.loads(schema_path.read_text(encoding='utf8')):
        if s['class_id'] not in {9,10,11}:continue
        sha=rgb_digest(s['image']);assert sha not in held|protected
        crop_sha=digest(s['crop'])
        if crop_sha in seen:continue
        seen.add(crop_sha)
        quality.append(dict(id='quality_'+s['id'],image=s['crop'],image_sha256=crop_sha,source_image=s['image'],
             source_rgb_sha256=sha,recording_group=recording_id(Path(s['image']).stem),role='train',
             type_id=0,direction_id=3,readability_label=1,readability_origin='Previously inspected clear TRAIN schema right arrow',
             origin='reused_schema_right',synthetic=False))
    train={r['source_rgb_sha256'] for r in rows if r['role']=='train'}
    assert not train&held and not train&protected
    assert not {r['source_rgb_sha256'] for r in quality if r['role']=='train'}&held
    result=dict(records=rows,readability_records=quality,source_manifest=str(SOURCE_MANIFEST),
         source_manifest_sha256=digest(SOURCE_MANIFEST),preparer_sha256=digest(Path(__file__)),
         fixture_decisions_sha256=digest(OUT/'fixture_visual_decisions.json'),schema_manifest_sha256=digest(schema_path),
         counts={s:dict(Counter(str(r['type_id']) for r in rows if r['role']==s)) for s in sorted({r['role'] for r in rows})},
         readability_counts={s:dict(Counter(str(r['readability_label']) for r in quality if r['role']==s)) for s in sorted({r['role'] for r in quality})},
         no_synthetic_training=True,readability_unknown_is_not_a_fifth_direction=True,
         all_evaluation_is_known_development=True,physical_lane_labels=False)
    save(target,result);print('SPATIAL_TAIL_MANIFEST',result['counts'],result['readability_counts'],flush=True)
    return result


if __name__=='__main__':
    proposals=fixture_proposals()
    print('FIXTURE_PROPOSALS_READY',len(proposals['records']),flush=True)
    if (OUT/'fixture_visual_decisions.json').exists():prepare()
