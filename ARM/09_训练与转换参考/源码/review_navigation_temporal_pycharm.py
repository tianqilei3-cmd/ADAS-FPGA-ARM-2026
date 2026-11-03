"""Causal seven-frame bursts ending at previously labelled source times.

No training and no future images. Native center baseline is scored separately
from the JPEG reference baseline to avoid attributing decode differences to fusion.
"""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import time

import cv2
import numpy as np

from light_roi_common import ROOT,read_image,write_image
from navigation_card_runtime import NavigationCardPipeline
from navigation_distance_ocr import NavigationDistanceReader
from navigation_temporal_ocr import TemporalNavigationOCR
from navigation_textline_runtime import NavigationTextlineReader
from official_light_config import VIDEOS
from signal_scene_adapter import image_identity

OUT=ROOT/'runs/navigation_temporal_20261019'


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def counts(rows,name):
    positive=[r for r in rows if r['reference']['distance_m'] is not None]
    negative=[r for r in rows if r['reference']['distance_m'] is None]
    def correct(row):
        value=row[name]
        return bool(value['distance_valid'] and row['reference']['distance_m'] is not None
                    and abs(value['distance_m']-row['reference']['distance_m'])<.01)
    return dict(positive_total=len(positive),exact=sum(correct(r) for r in positive),
        unknown=sum(not r[name]['distance_valid'] for r in positive),
        wrong_accepted=sum(r[name]['distance_valid'] and not correct(r) for r in rows),
        negative_total=len(negative),negative_rejected=sum(not r[name]['distance_valid'] for r in negative))


def main():
    import torch
    torch.set_num_threads(8);cv2.setNumThreads(1)
    refs_path=ROOT/'runs/navigation_distance_20261018/distance_references.json'
    references=json.loads(refs_path.read_text(encoding='utf-8'))
    card=json.loads((ROOT/'runs/navigation_card_20261018/pc_card_candidate.json').read_text(encoding='utf-8'))
    training=json.loads((ROOT/'runs/navigation_textline_20261019/latest_training.json').read_text(encoding='utf-8'))
    ocr=ROOT/'external_models/navigation_ocr/rapidocr_v4_mobile/ch_PP-OCRv4_rec_mobile.onnx'
    models={name:dict(path=str(path),sha256=digest(path)) for name,path in
        dict(card=card['card_weights'],glyph=card['glyph_weights'],ocr=ocr,textline=training['best']).items()}
    assert models['textline']['sha256']==training['best_sha256']
    pipeline=NavigationCardPipeline(card['card_weights'],card['glyph_weights'])
    reader=NavigationDistanceReader(ocr)
    line_reader=NavigationTextlineReader(training['best'],ocr,ocr=reader.ocr)
    pipeline.predict(np.full((512,512,3),128,np.uint8))
    output=OUT/('review_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True)
    videos={r['video']:VIDEOS/f"video_20260929_{r['video']}.mp4" for r in references['records']}
    protocol=dict(refs_path=str(refs_path),refs_sha256=digest(refs_path),models=models,
        source_sha256={v:digest(p) for v,p in videos.items()},
        sources={n:dict(path=str(ROOT/n),sha256=digest(ROOT/n)) for n in
        ['navigation_temporal_ocr.py','review_navigation_temporal_pycharm.py',
         'navigation_textline_runtime.py','navigation_card_runtime.py','navigation_distance_ocr.py']},
        frames_per_burst=7,causal_only=True,maximum_age_ms=200.,minimum_support=3,
        alignment_correlation=.92,unit_end_correlation=.80,max_translation_at48px=4,
        native_digit_agreement_required=True,explicit_unit_ocr_required=True,
        old_unit_propagation=False,parameters_fixed_before_review=True,
        source_independent=False,training_performed=False,default_changed=False,hardware_accessed=False)
    save(output/'before_inference.json',protocol)
    captures={};rows=[];tiles=[]
    try:
        for ref in references['records']:
            assert digest(ref['image'])==ref['sha256']
            path=videos[ref['video']]
            if ref['video'] not in captures:
                cap=cv2.VideoCapture(str(path))
                if not cap.isOpened():raise RuntimeError('Cannot open '+str(path))
                captures[ref['video']]=cap
            cap=captures[ref['video']];fps=float(cap.get(cv2.CAP_PROP_FPS))
            second=int(re.search(r'__(\d+)\.',Path(ref['image']).name).group(1))
            center=round(second*fps);start=max(0,center-6)
            cap.set(cv2.CAP_PROP_POS_FRAMES,start)
            temporal=TemporalNavigationOCR(reader.ocr)
            burst=[];center_image=None;last=None
            began=time.perf_counter()
            for number in range(start,center+1):
                ok,image=cap.read()
                if not ok:raise RuntimeError('Failed native burst decode')
                frame=dict(camera_id='native:'+ref['video'],frame_id=number,
                    captured_ms=1000+number/fps*1000,image_identity=image_identity(image))
                nav=dict(pipeline.predict(image),**frame)
                baseline=dict(reader.read(image,nav),**frame)
                line=dict(line_reader.read(image,nav),**frame)
                last=temporal.observe(image,frame,nav,baseline,line,now_ms=frame['captured_ms'])
                box=line.get('distance_box') or baseline.get('distance_box')
                if box:
                    a,b,c,d=map(int,box)
                    write_image(output/'evidence'/ref['id']/f'line_{number:06d}.png',image[b:d,a:c])
                burst.append(dict(frame=frame,navigation=nav,baseline=baseline,line=line,
                                  temporal=last,crop_box=box))
                if number==center:center_image=image
            # Reference images and labels below are used only for paired scoring/QA.
            original=read_image(ref['image'])
            if original.shape!=center_image.shape:raise RuntimeError('Reference/native shape mismatch')
            mae=float(np.abs(original.astype(np.float32)-center_image.astype(np.float32)).mean())
            if mae>8:raise RuntimeError('Reference/native timestamp alignment requires review: '+ref['id']+' MAE='+str(mae))
            jpeg_nav=pipeline.predict(original);jpeg_baseline=reader.read(original,jpeg_nav)
            row=dict(reference=ref,center_frame=center,source_fps=fps,center_native_reference_mae=mae,
                jpeg_baseline=jpeg_baseline,native_baseline=burst[-1]['baseline'],
                native_textline=burst[-1]['line'],temporal=last['temporal_distance'],
                proposal=last['proposal_distance'],burst=burst,
                burst_processing_ms=(time.perf_counter()-began)*1000)
            rows.append(row)
            fused=temporal.last_fused_image
            if fused is not None:write_image(output/'evidence'/ref['id']/'fused.png',fused)
            canvas=cv2.resize(center_image,(640,360))
            box=burst[-1]['line'].get('distance_box') or burst[-1]['baseline'].get('distance_box')
            line_crop=None
            if box:
                a,b,c,d=map(int,box);line_crop=center_image[b:d,a:c]
                xy=np.array(box)*np.array([640/center_image.shape[1],360/center_image.shape[0]]*2)
                cv2.rectangle(canvas,tuple(xy[:2].astype(int)),tuple(xy[2:].astype(int)),(0,255,255),2)
            bottom=np.zeros((64,640,3),np.uint8)
            if line_crop is not None:bottom[:,:320]=cv2.resize(line_crop,(320,64))
            if fused is not None:bottom[:,320:]=cv2.resize(fused,(320,64))
            header=np.zeros((100,640,3),np.uint8)
            for j,text in enumerate([ref['id']+f" expected={ref['distance_m']}m",
                f"JPEG={jpeg_baseline['distance_m']} native={row['native_baseline']['distance_m']} proposal={row['proposal']['distance_m']}",
                last['reason'],f"support={len(last['support_frames'])}; bottom: native / aligned blend; known DEV"]):
                cv2.putText(header,text[:95],(4,20+j*24),0,.43,(255,255,255),1)
            tiles.append(np.vstack([header,canvas,bottom]))
            print('TEMPORAL_CASE',ref['id'],'native',row['native_baseline']['distance_m'],
                  'proposal',row['proposal']['distance_m'],'reason',last['reason'],'mae',round(mae,3),flush=True)
    finally:
        for cap in captures.values():cap.release()
    for first in range(0,len(tiles),4):
        page=tiles[first:first+4]
        while len(page)<4:page.append(np.zeros_like(page[0]))
        write_image(output/f'page_{first//4+1:02d}.jpg',np.vstack([np.hstack(page[:2]),np.hstack(page[2:])]))
    names=['jpeg_baseline','native_baseline','native_textline','temporal','proposal']
    summary={name:counts(rows,name) for name in names}
    reasons=dict(Counter(r['burst'][-1]['temporal']['reason'] for r in rows))
    result=dict(state='completed_pending_visual_review',output=str(output),protocol=protocol,
        summary=summary,records=rows,reasons=reasons,
        source_independent_acceptance=False,default_changed=False,hardware_accessed=False)
    save(output/'results.json',result);save(OUT/'latest_review.json',dict(output=str(output),summary=summary,reasons=reasons))
    print('TEMPORAL_SUMMARY',json.dumps(summary),reasons,flush=True)


if __name__=='__main__':main()
