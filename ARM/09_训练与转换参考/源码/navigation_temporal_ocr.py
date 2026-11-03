"""Causal pixel fusion experiment, without unit propagation or motion output.

Only distinct, recent, matched card/number observations enter the image blend.
The OCR must read an explicit unit from the blended pixels. Reference labels,
video names, future images and previously accepted distances are not inputs.
"""
from collections import deque
import hashlib
import math

import cv2
import numpy as np

from competition_scene_logic import finite_number, same_frame
from navigation_distance_ocr import _distance_literal
from navigation_textline_runtime import fuse_distances, iou, resolve_readings, unknown
from signal_scene_adapter import image_identity


def correlation(a, b):
    a=np.asarray(a,dtype=np.float32).ravel();b=np.asarray(b,dtype=np.float32).ravel()
    a=a-a.mean();b=b-b.mean()
    denominator=float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.dot(a,b)/denominator) if denominator>1e-5 else 0.


def line_quality(crop):
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(3,3),.6)
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0);gy=cv2.Sobel(gray,cv2.CV_32F,0,1)
    return float(np.mean(gx*gx+gy*gy))


def align_line(current, prior, minimum_correlation=.92, minimum_right_correlation=.80):
    """Translate a same-number text line; never synthesize new text strokes."""
    h,w=current.shape[:2]
    if not .85<=(prior.shape[1]/prior.shape[0])/(w/h)<=1.15:
        return None,dict(reason='line_aspect_changed')
    # A common48px text height supports subpixel alignment, not new information.
    width=max(32,round(48*w/h));shape=(width,48)
    target=cv2.resize(current,shape).astype(np.float32)
    source=cv2.resize(prior,shape).astype(np.float32)
    t=cv2.cvtColor(target,cv2.COLOR_BGR2GRAY)/255
    s=cv2.cvtColor(source,cv2.COLOR_BGR2GRAY)/255
    transform=np.eye(2,3,dtype=np.float32)
    try:
        _,transform=cv2.findTransformECC(t,s,transform,cv2.MOTION_TRANSLATION,
            (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,1e-4),None,3)
    except cv2.error:
        return None,dict(reason='alignment_failed')
    if not np.isfinite(transform).all() or np.max(np.abs(transform[:,2]))>4:
        return None,dict(reason='alignment_displacement_too_large')
    aligned=cv2.warpAffine(source,transform,shape,flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_REPLICATE)
    gray=cv2.cvtColor(aligned,cv2.COLOR_BGR2GRAY)/255
    cc=correlation(t,gray)
    # Check the unit end as well as the large digits. Blank margins do not count.
    start=round(width*.70)
    right_cc=correlation(t[:,start:],gray[:,start:])
    if cc<minimum_correlation or right_cc<minimum_right_correlation:
        return None,dict(reason='line_or_unit_end_pixels_changed',correlation=cc,right_correlation=right_cc)
    return aligned,dict(reason='aligned',correlation=cc,right_correlation=right_cc,
                        translation=transform[:,2].tolist())


def numeric_agreement(*readers):
    hints={float(r['numeric_value_hint']) for r in readers
           if finite_number(r.get('numeric_value_hint'))}
    return next(iter(hints)) if len(hints)==1 else None


def literal_scales(*readers):
    """Include explicit out-of-range OCR literals when detecting unit changes."""
    values=set()
    for reader in readers:
        for raw in reader.get('ocr_views',[]):
            if raw.get('confidence',0)<.85 or raw.get('minimum_character_confidence',0)<.50:
                continue
            parsed=_distance_literal(raw.get('text'))
            if parsed is not None:
                values.add(1000 if parsed['observed_unit'] in ['km','KM','公里','千米'] else 1)
    return values


class TemporalNavigationOCR:
    def __init__(self, ocr, window_ms=200., minimum_frames=3):
        if not finite_number(window_ms) or not 0<window_ms<=500 or type(minimum_frames) is not int or not 3<=minimum_frames<=7:
            raise ValueError('Invalid bounded causal fusion policy')
        self.ocr=ocr;self.window_ms=window_ms;self.minimum_frames=minimum_frames
        self.history=deque(maxlen=7);self.watermarks={};self.last_camera=None
        self.last_fused_image=None;self.last_now=None

    def clear(self):
        self.history.clear()

    def observe(self, image, frame, nav, baseline, line, *, now_ms):
        self.last_fused_image=None
        candidate=unknown('not_enough_matched_recent_frames')
        result=dict(temporal_distance=candidate,baseline=baseline,
                    proposal_distance=dict(baseline),support_frames=[],alignments=[],
                    sharpest_observation=None,hardware_command=None,
                    immediate_turn_authorized=False,experimental=True)

        def finish(reason):
            result['reason']=reason
            return result

        camera,number,captured=frame.get('camera_id'),frame.get('frame_id'),frame.get('captured_ms')
        if finite_number(now_ms) and self.last_now is not None and now_ms<self.last_now:
            self.clear();self.last_now=now_ms;result['proposal_distance']=unknown('temporal_clock_moved_backwards')
            return finish('temporal_clock_moved_backwards')
        self.last_now=now_ms if finite_number(now_ms) else None
        if (not finite_number(now_ms) or not isinstance(camera,str) or not camera
                or type(number) is not int or number<0 or not finite_number(captured) or captured<0
                or not 0<=now_ms-captured<=self.window_ms
                or frame.get('image_identity')!=image_identity(image)):
            self.clear();result['proposal_distance']=unknown('invalid_or_stale_temporal_frame')
            return finish('invalid_or_stale_temporal_frame')
        previous=self.watermarks.get(camera)
        if previous and (number<=previous[0] or captured<=previous[1]):
            self.clear();result['proposal_distance']=unknown('replayed_temporal_frame')
            return finish('replayed_temporal_frame')
        self.watermarks[camera]=(number,captured)
        if not all(same_frame(row,frame) for row in [nav,baseline,line]):
            self.clear();result['proposal_distance']=unknown('temporal_result_frame_mismatch')
            return finish('temporal_result_frame_mismatch')
        if camera!=self.last_camera:
            self.clear();self.last_camera=camera
        number_hint=numeric_agreement(baseline,line)
        box=line.get('distance_box') or baseline.get('distance_box')
        card=nav.get('card_box')
        if (nav.get('direction_valid') is not True or nav.get('direction') not in ['left','straight','right']
                or number_hint is None or box is None or card is None):
            self.clear();return finish('no_current_supported_card_and_numeric_evidence')
        if (len(box)!=4 or not all(finite_number(v) for v in box)
                or len(card)!=4 or not all(finite_number(v) for v in card)):
            self.clear();return finish('invalid_text_or_card_box')
        a,b,c,d=map(int,box)
        if not (0<=a<c<=image.shape[1] and 0<=b<d<=image.shape[0] and c-a>=12 and d-b>=6
                and card[0]<=a<c<=card[2] and card[1]<=b<d<=card[3]):
            self.clear();return finish('text_box_outside_current_card')
        while self.history and captured-self.history[0]['frame']['captured_ms']>self.window_ms:
            self.history.popleft()
        if self.history:
            last=self.history[-1]
            if (last['number']!=number_hint or last['direction']!=nav['direction']
                    or last['card_class']!=nav.get('card_class') or iou(last['card'],card)<.60):
                self.clear()
        crop=image[b:d,a:c].copy()
        crop_hash=hashlib.sha256(crop.tobytes()).hexdigest()
        if any(h['frame']['image_identity']==frame['image_identity'] or h['crop_hash']==crop_hash for h in self.history):
            return finish('duplicate_pixels_do_not_add_fusion_support')
        self.history.append(dict(frame=dict(frame),number=number_hint,direction=nav['direction'],
            card_class=nav.get('card_class'),card=list(card),crop=crop,crop_hash=crop_hash,
            quality=line_quality(crop),scales=literal_scales(baseline,line)))
        sharpest=max(self.history,key=lambda row:row['quality'])
        result['sharpest_observation']=dict(frame=sharpest['frame'],quality=sharpest['quality'],
            age_ms=captured-sharpest['frame']['captured_ms'],current_distance_inferred=False)
        scales=set().union(*(h['scales'] for h in self.history))
        if len(scales)>1:
            self.clear();result['proposal_distance']=unknown('native_explicit_unit_conflict')
            return finish('native_explicit_unit_conflict')
        if len(self.history)<self.minimum_frames:
            return finish('not_enough_matched_recent_frames')
        current=self.history[-1]
        h,w=crop.shape[:2];size=(max(32,round(48*w/h)),48)
        aligned=[cv2.resize(crop,size).astype(np.float32)]
        supports=[current['frame']]
        for prior in sorted(list(self.history)[:-1],key=lambda row:row['quality'],reverse=True):
            image_aligned,diagnostic=align_line(crop,prior['crop'])
            result['alignments'].append(dict(frame=prior['frame'],**diagnostic))
            if image_aligned is not None:
                aligned.append(image_aligned);supports.append(prior['frame'])
            if len(aligned)==self.minimum_frames:break
        if len(aligned)<self.minimum_frames:
            return finish('insufficient_pixel_alignment_at_number_and_unit_end')
        combined=np.clip(np.mean(aligned,axis=0),0,255).astype(np.uint8)
        self.last_fused_image=combined
        gray=cv2.cvtColor(combined,cv2.COLOR_BGR2GRAY)
        enhanced=cv2.createCLAHE(clipLimit=2.,tileGridSize=(4,2)).apply(gray)
        views=[self.ocr.recognize(combined),self.ocr.recognize(cv2.cvtColor(enhanced,cv2.COLOR_GRAY2BGR))]
        candidate=resolve_readings(views,box)
        if candidate['distance_valid'] and candidate['observed_value']!=number_hint:
            candidate=unknown('fused_digits_disagree_with_current_native_digits')
        if candidate['distance_valid'] and scales:
            scale=1000 if candidate['observed_unit'] in ['km','KM','公里','千米'] else 1
            if scale not in scales:candidate=unknown('fused_unit_disagrees_with_native_explicit_unit')
        result.update(temporal_distance=candidate,proposal_distance=fuse_distances(baseline,candidate),fused_ocr_views=views,
                      support_frames=supports,evidence_oldest_ms=min(r['captured_ms'] for r in supports),
                      evidence_newest_ms=captured)
        return finish('pixel_fusion_explicit_unit_read' if candidate['distance_valid'] else 'pixel_fusion_unit_still_unreadable')
