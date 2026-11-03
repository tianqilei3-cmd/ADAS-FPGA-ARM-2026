"""Same-pixel lamp/road scheduling for declared virtual lane simulation.

One full-width lamp pass; CPU lamp classification overlaps road NPU inference.
No annotated lamp ROI, horizontal-order lane assignment or physical-lane claim.
"""
import time
from concurrent.futures import ThreadPoolExecutor,Future
from pathlib import Path

import cv2
import numpy as np
from arm_scene_a04 import frame_identity, scene_decision, box_iou
from competition_scene_logic import SceneController
from lamp_attrs_portable import PortableAttributes
from fast_types import FastTypes
from npu_signal_attributes import NpuWholeTypes
from head_continuity import HeadContinuity
from light_pipeline import merge_lights
from light_roi_common import NAMES, crop_head, input_rgb, classify_policy
from signal_s3_policy import AutomaticSignalSelector, RoundGroupSupplement, guarded_signal_rows, inside_vehicle_body, round_emitter, conservative_red_group
from vision_common import preprocess, decoded_output, decode


def unknown_signal(frame, controls, reason):
    return dict(frame, selected_state='unknown', selected_valid=False,
                simulation_association_valid=False, physical_lane_association_verified=False,
                lane_id=controls.lane_id, movement=controls.movement, reason=reason,
                head_id=None, association_scope='virtual_lane_only')


def combined_decision(frame, controls, hazard, signal, controller, now_ms):
    if controls.mode != 'signal':
        return scene_decision(frame, controls, hazard, now_ms)
    result = controller.update(frame, controls, signal=signal, now_ms=now_ms)
    result.update(own_lane_light_valid=bool(signal.get('selected_valid')),
                  physical_lane_association_verified=False, motor_command_sent=False,
                  source='current_pixels_and_key_simulated_lane', selected_state=signal.get('selected_state','unknown'))
    if result['action'] == 'HOLD': result['action'] = 'STOP'
    if hazard.get('action') == 'STOP':
        result.update(action='STOP', alarm=bool(hazard.get('alarm') or result['alarm']),
                      reason='hazard:'+hazard.get('reason','unknown'))
    elif result['action'] == 'GO' and not (hazard.get('decision_valid') and hazard.get('action') == 'GO'):
        result.update(action='STOP', decision_valid=False, reason='hazard_clearance_unconfirmed')
    return result




def focused_seed(search,previous,frame,minimum_housing_aspect=.8):
    """A prior coordinate is a search hint, never a current detection or color."""
    horizontal=[p for p in search if minimum_housing_aspect<=(p['box'][2]-p['box'][0])/(p['box'][3]-p['box'][1])<=4.5]
    strong=[p for p in horizontal if p['confidence']>=.25]
    if len(strong)>=2:return None
    weak=[p for p in horizontal if p['confidence']<.25 and all(box_iou(p['box'],q['box'])<.3 for q in strong)]
    candidates=weak or strong or horizontal
    if candidates:return max(candidates,key=lambda p:p['confidence'])
    if (previous and previous['camera_id']==frame['camera_id']
        and 0<frame['captured_ms']-previous['captured_ms']<=500 and previous['boxes']):
        return dict(box=previous['boxes'][0],confidence=0.,historical_search_hint_only=True)
    return None


class SignalEngine:
    def __init__(self, root, guard, metadata):
        from rknnlite.api import RKNNLite
        self.guard, self.metadata = guard, metadata
        self.size=metadata['image_size']
        self.root=Path(root);self.attributes=None;self.cpu_worker=None;self.fast_types=None;self.geometry_worker=None
        if metadata.get('proposal_mode')=='geometry':
            from geometry_cpu_worker import GeometryWorker
            self.geometry_worker=GeometryWorker()
        backend=metadata.get('classifiers','npu')
        if backend=='cpu_process':
            from lamp_cpu_worker import CpuLampWorker
            self.cpu_worker=CpuLampWorker(root)
        elif backend in ('batch2','batch3'):
            from batched_signal_attributes import BatchedWholeTypes
            self.fast_types=BatchedWholeTypes(Path(root)/('models/lamp_types_'+backend+'_fp.rknn'),batch_size=int(backend[-1]))
        else:self.fast_types = NpuWholeTypes(Path(root)/'models/lamp_types_fp.rknn')
        self.lamp = RKNNLite(verbose=False)
        try:
            assert self.lamp.load_rknn(str(Path(root)/'models'/metadata.get('model_name','light_fp.rknn'))) == 0
            assert self.lamp.init_runtime() == 0
            self.lamp.inference(inputs=[np.zeros((1,*((self.size,self.size) if isinstance(self.size,int) else self.size),3),np.uint8)],data_format=['nhwc'])
        except BaseException:
            self.lamp.release()
            if self.fast_types:self.fast_types.close()
            if self.cpu_worker:self.cpu_worker.close()
            if self.geometry_worker:self.geometry_worker.close()
            raise
        self.pool = ThreadPoolExecutor(max_workers=1,thread_name_prefix='same_frame_lamp_cpu')
        self.selector = AutomaticSignalSelector()
        self.rounds = RoundGroupSupplement()
        self.detector_hint = None
        self.continuity = HeadContinuity(max_gap_ms=500.)
        self.controller = SceneController(max_age_ms=500.,green_confirmations=3)

    def begin(self, image, frame, controls=None):
        tick = time.monotonic()
        if self.metadata.get('proposal_mode')=='geometry' and controls is not None and controls.movement=='straight':
            future=self.geometry_worker.submit(image,(frame['frame_id'],frame['image_identity']))
            return dict(frame=dict(frame),geometry_future=future,started=tick)
        road_top = image[:round(image.shape[0]*.6)]
        rgb, geometry = preprocess(road_top,self.size)
        outputs = self.lamp.inference(inputs=[rgb[None]],data_format=['nhwc'])
        proposals = decode(decoded_output(outputs,self.size,4,score_threshold=.10),geometry,
                           self.metadata['names'],.10,self.metadata['iou'],30)
        proposals=merge_lights(proposals,.5,30)
        if self.metadata.get('housing_refinement',False):
            from current_lamp_housing import refine
            proposals=merge_lights([refine(image,p) for p in proposals],.5,30)
        search_proposals=proposals
        proposals=[p for p in proposals if p['confidence']>=.25]
        full_count=len(proposals);focused=False
        seed=focused_seed(search_proposals,self.detector_hint,frame,1.3 if self.metadata.get('housing_refinement',False) else .8)
        # Previous coordinates can guide a fresh NPU pass, not supply its output.
        if seed is not None:
            a,b,c,d=seed['box'];cx,cy=(a+c)/2,(b+d)/2
            rw,rh=max(192.,8*(c-a)),max(64.,5*(d-b))
            if self.metadata.get('housing_refinement',False):rw=max(rw,.45*road_top.shape[1])
            left,top=max(0,int(cx-rw/2)),max(0,int(cy-rh/2))
            right,bottom=min(road_top.shape[1],int(cx+rw/2)),min(road_top.shape[0],int(cy+rh/2))
            tile=road_top[top:bottom,left:right]
            rgb2,g2=preprocess(tile,self.size)
            raw2=self.lamp.inference(inputs=[rgb2[None]],data_format=['nhwc'])
            extra=decode(decoded_output(raw2,self.size,4,score_threshold=.25),g2,
                         self.metadata['names'],.25,self.metadata['iou'],30)
            for row in extra:
                x1,y1,x2,y2=row['box']
                if ((left>0 and x1<=1) or (top>0 and y1<=1) or
                    (right<road_top.shape[1] and x2>=right-left-1) or
                    (bottom<road_top.shape[0] and y2>=bottom-top-1)):continue
                proposals.append(dict(row,box=[x1+left,y1+top,x2+left,y2+top]))
            focused=True
        proposals = merge_lights(proposals,.5,30)
        if focused and self.metadata.get('housing_refinement',False):proposals=merge_lights([refine(image,p) for p in proposals],.5,30)
        horizontal=[p for p in proposals if 1.3<=(p['box'][2]-p['box'][0])/(p['box'][3]-p['box'][1])<=4.5]
        if horizontal:self.detector_hint=dict(camera_id=frame['camera_id'],captured_ms=frame['captured_ms'],boxes=[p['box'] for p in horizontal])
        detection_ms = (time.monotonic()-tick)*1000
        # A crowded frame remains unknown; never hide omitted competing heads.
        overloaded = len(proposals)>6
        future = Future()
        if self.cpu_worker:
            token=(frame['frame_id'],frame['image_identity'])
            future=self.cpu_worker.submit(image,proposals if not overloaded else [],controls is None or controls.movement!='straight',token)
        elif getattr(self.guard,'on_npu',False):
            future.set_result(self.classify(image,proposals if not overloaded else [],allow_direction=controls is None or controls.movement!='straight'))
        else:future = self.pool.submit(self.classify, image, proposals if not overloaded else [], allow_direction=controls is None or controls.movement!='straight')
        return dict(frame=dict(frame),future=future,detector_ms=detection_ms,
                    proposal_count=len(proposals),full_proposal_count=full_count,focused_recheck=focused,overloaded=overloaded,started=tick)

    def classify(self,image,proposals,allow_direction=True):
        started=time.monotonic();good=[];crops=[]
        for proposal in proposals:
            try:crop=crop_head(image,proposal['box'])
            except ValueError:continue
            good.append(proposal);crops.append(crop)
        if not crops:return [],(time.monotonic()-started)*1000
        tensor=np.stack([input_rgb(c,128) for c in crops]).transpose(0,3,1,2).astype(np.float32)/255
        logits=self.guard.run(None,{self.guard.get_inputs()[0].name:tensor})[0]
        def softmax(z):
            p=np.exp(z-z.max(axis=1,keepdims=True));return p/p.sum(axis=1,keepdims=True)
        colors=softmax(logits[:,:4]);emissions=softmax(logits[:,4:])[:,1]
        # Joint confidence cannot exceed detector confidence or its actual
        # detector*classifier product. Only impossible candidates are removed.
        eligible=[i for i,(proposal,p) in enumerate(zip(good,colors)) if proposal['confidence']*float(p.max())>=.25]
        good=[good[i] for i in eligible];crops=[crops[i] for i in eligible]
        colors=colors[eligible];emissions=emissions[eligible]
        if not crops:return [],(time.monotonic()-started)*1000
        attributes=self.fast_types.predict(crops,batch=4);rows=[];fallback=[]
        for index,(proposal,crop,p,attr) in enumerate(zip(good,crops,colors,attributes)):
            guarded=guarded_signal_rows(image,[dict(box=proposal['box'],probabilities=p.tolist(),visible_emission_probability=float(emissions[index]))])[0]
            color=guarded['name']
            if attr['lamp_type']!='vehicle_signal' or color not in ('red','yellow','green'):continue
            evidence=round_emitter(image,proposal['box'],color)
            a,b,c,d=proposal['box']
            if evidence['valid'] and 1.3<=(c-a)/(d-b)<=4.5:
                attr.update(shape='round',shape_source='current_filled_disk_geometry',
                            shape_evidence=evidence,control_direction='unknown')
            else:fallback.append(index)
        # Direction stays current-image neural inference for non-round candidates.
        if fallback and allow_direction:
            if self.attributes is None:self.attributes=PortableAttributes(self.root/'models/lamp_attributes.onnx',threads=2)
            values=self.attributes.predict([crops[i] for i in fallback],batch=2)
            for i,value in zip(fallback,values):attributes[i]=dict(value,type_source='full_original_attributes_direction_fallback')

        for proposal,crop,p,emission,attr in zip(good,crops,colors,emissions,attributes):
            cls,reason=classify_policy(p,crop)
            rows.append(dict(proposal,name=NAMES[cls] if cls<4 else 'unknown',
                confidence=float(proposal['confidence']*p.max()),
                detector_confidence=proposal['confidence'],probabilities=p.tolist(),
                reason=reason,visible_emission_probability=float(emission),attributes=attr))
        return guarded_signal_rows(image,rows),(time.monotonic()-started)*1000

    def finish(self,job,image,roads,frame,controls):
        if 'geometry_future' in job:
            (proposals,info),detector_ms=job['geometry_future'].result(timeout=3.)
            overloaded=info['overloaded'] or len(proposals)>6
            fallback=False
            if not overloaded and len(proposals)<2:
                # A current measured emitter may have an open/occluded casing.
                # Search the same full current image with the retained detector;
                # neither its color nor its box comes from a previous frame.
                tick=time.monotonic();road_top=image[:round(image.shape[0]*.6)];rgb,g=preprocess(road_top,self.size)
                outputs=self.lamp.inference(inputs=[rgb[None]],data_format=['nhwc'])
                extra=decode(decoded_output(outputs,self.size,4,score_threshold=.25),g,self.metadata['names'],.25,self.metadata['iou'],30)
                proposals=merge_lights(proposals+extra,.5,30);detector_ms+=(time.monotonic()-tick)*1000;fallback=True
                overloaded=len(proposals)>6
            rows,cpu_ms=self.classify(image,proposals if not overloaded else [],False)
            job.update(detector_ms=detector_ms,proposal_count=len(proposals),full_proposal_count=len(proposals),focused_recheck=fallback,overloaded=overloaded,proposal_method=info['source']+'+current_neural_fallback_when_missing',component_candidates=info['component_candidates'])
        else:rows,cpu_ms=job['future'].result(timeout=5.)
        now=time.monotonic()*1000
        metrics=dict(detector_ms=job['detector_ms'],cpu_ms=cpu_ms,
                     overlap_elapsed_ms=(time.monotonic()-job['started'])*1000,
                     proposal_count=job['proposal_count'],full_proposal_count=job['full_proposal_count'],focused_recheck=job['focused_recheck'],overloaded=job['overloaded'])
        metrics.update(proposal_method=job.get('proposal_method','neural_detector'),component_candidates=job.get('component_candidates'))
        expected=frame_identity(image,'',0,0,0)['image_identity']
        if job['frame']!=frame or frame.get('image_identity')!=expected:
            selected=unknown_signal(frame,controls,'current_image_identity_mismatch')
        elif not 0<=now-frame['captured_ms']<=500:
            selected=unknown_signal(frame,controls,'signal_result_stale')
        elif job['overloaded']:
            selected=unknown_signal(frame,controls,'too_many_competing_lamp_proposals')
        else:
            for row in rows:row['inside_vehicle']=inside_vehicle_body(row['box'],roads)
            selected=self.selector.select(rows,frame,controls)
            selected=conservative_red_group(image,rows,roads,frame,controls,selected)
            selected=self.rounds.refine(image,rows,roads,frame,controls,selected)
        if selected.get('selected_state')=='green':selected=self.continuity.update(image,frame,controls,selected)
        else:self.continuity.previous=None
        return rows,selected,metrics

    def close(self):
        self.pool.shutdown(wait=True,cancel_futures=True)
        self.lamp.release()
        if self.fast_types:self.fast_types.close()
        if self.cpu_worker:self.cpu_worker.close()
        if self.geometry_worker:self.geometry_worker.close()


def display_decision(decision,captured_ms,now_ms,display_budget_ms=470.):
    """Reserve 30 ms for publication; UART keeps its original 500 ms TTL."""
    if not 0<=now_ms-captured_ms<=display_budget_ms:
        return dict(decision,action='STOP',decision_valid=False,reason='result_stale_before_display')
    return decision
