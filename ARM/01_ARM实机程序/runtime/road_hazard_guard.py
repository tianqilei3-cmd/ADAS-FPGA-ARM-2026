"""Observed vulnerable road users/cones in an explicitly virtual corridor.

Risk memory can maintain a stop, but missing detections cannot clear it. A
matched visible exit and subsequent fresh empty observations are required.
This is a PC simulation policy, not calibrated physical vehicle control.
"""
from dataclasses import asdict
import math

import cv2
import numpy as np

from competition_scene_logic import finite_number,same_frame
from arm_scene_a04 import box_iou

HAZARDS={'person':'vru','two_wheeler':'vru','traffic_cone':'cone'}


def corridor(image_shape,zone='center'):
    h,w=image_shape[:2]
    if zone not in ['center','left','right']:raise ValueError('Unknown declared virtual corridor')
    shift={'center':0.,'left':-.23,'right':.23}[zone]
    points=np.array([[.43+shift,.52],[.57+shift,.52],[.80+shift,1.],[.20+shift,1.]],np.float32)
    points[:,0]=np.clip(points[:,0],0,1)
    return (points*np.array([w,h],np.float32)).tolist()


def hazard_observation(objects,frame,image_shape,*,zone='center',kinds=('vru','cone'),crosswalks=None):
    h,w=image_shape[:2];polygon=corridor(image_shape,zone)
    candidates=[]
    for item in objects:
        family=HAZARDS.get(item.get('name'))
        if family not in kinds:continue
        confidence=item.get('confidence');box=item.get('box')
        if (not finite_number(confidence) or not .15<=confidence<=1 or not isinstance(box,(list,tuple))
            or len(box)!=4 or not all(finite_number(v) for v in box)):
            continue
        a,b,c,d=map(float,box)
        if not (0<=a<c<=w and 0<=b<d<=h):continue
        foot=[(a+c)/2,d]
        signed=float(cv2.pointPolygonTest(np.asarray(polygon,np.float32),tuple(foot),True))
        candidates.append(dict(item,family=family,foot_point=foot,in_corridor=signed>=0,
            outside_with_margin=signed < -max(3,.015*w),signed_corridor_distance=signed))
    return dict(frame,observation_valid=True,zone=zone,objects=candidates,polygon=polygon,
        image_size=[w,h],crosswalks=list(crosswalks or []),
        crosswalk_is_context_not_clearance=True,pedestrian_red_violation_inferred=False,
        physical_lane_association_verified=False,scope='declared_fixed_virtual_corridor')


class HazardStopController:
    def __init__(self,*,zone='center',max_age_ms=200.,clear_confirmations=3,exit_confirmations=2):
        if zone not in ['center','left','right']:raise ValueError('Explicit corridor required')
        if (not finite_number(max_age_ms) or max_age_ms<=0 or type(clear_confirmations)is not int
            or clear_confirmations<2 or type(exit_confirmations)is not int or exit_confirmations<2):
            raise ValueError('Invalid freshness/confirmation policy')
        self.zone=zone;self.max_age_ms=max_age_ms
        self.required_clear=clear_confirmations;self.required_exit=exit_confirmations
        self.camera=None;self.watermarks={};self.last_now=None;self.last_capture=None
        self.active={};self.serial=0;self.clear_count=0;self.had_intruder=False;self.visible_exits=0

    def reset_after_manual_review(self,controls):
        if controls.requested_motion!='stop':raise ValueError('Stop before manual risk reset')
        self.active={};self.clear_count=0;self.last_capture=None;self.had_intruder=False

    def update(self,frame,observation,controls,*,now_ms):
        result=dict(simulation_only=True,hardware_command=None,hardware_command_sent=False,
            physical_lane_association_verified=False,physical_turn_authorized=False,
            action='HOLD',alarm=False,decision_valid=False,zone=self.zone,
            reason='unknown_current_hazard_observation',active_tracks=[],clear_confirmations=0,
            visible_exit_total=self.visible_exits,controls=asdict(controls))
        def finish(reason,action='HOLD',alarm=False,valid=False):
            return dict(result,reason=reason,action=action,alarm=alarm,decision_valid=valid,
                active_tracks=[dict(v) for v in self.active.values()],clear_confirmations=self.clear_count,
                visible_exit_total=self.visible_exits)
        if controls.requested_motion not in ['go','stop']:raise ValueError('Invalid motion')
        if not finite_number(now_ms) or now_ms<0:raise ValueError('Invalid monotonic time')
        if self.last_now is not None and now_ms<self.last_now:
            self.last_now=now_ms;self.clear_count=0
            return finish('clock_moved_backwards',alarm=bool(self.active))
        self.last_now=now_ms
        camera=frame.get('camera_id');fid=frame.get('frame_id');captured=frame.get('captured_ms')
        if (not isinstance(camera,str) or not camera or type(fid)is not int or fid<0
            or not finite_number(captured) or captured<0 or not isinstance(frame.get('image_identity'),str)
            or len(frame['image_identity'])!=64):
            self.clear_count=0;return finish('invalid_frame_metadata')
        old=self.watermarks.get(camera)
        if old and (fid<=old[0] or captured<=old[1]):
            self.clear_count=0;return finish('replayed_or_out_of_order_frame',alarm=bool(self.active))
        self.watermarks[camera]=(fid,captured)
        if not 0<=now_ms-captured<=self.max_age_ms:
            self.clear_count=0
            return finish('stale_or_future_frame','STOP' if controls.requested_motion=='stop' else 'HOLD',bool(self.active))
        if (not same_frame(observation,frame) or not observation.get('observation_valid')
            or observation.get('zone')!=self.zone or not isinstance(observation.get('objects'),list)):
            self.clear_count=0;return finish('hazard_result_frame_or_zone_mismatch',alarm=bool(self.active))
        if self.camera!=camera:
            self.camera=camera;self.active={};self.clear_count=0;self.last_capture=None
            self.had_intruder=False;self.visible_exits=0
        contiguous=self.last_capture is not None and 0<captured-self.last_capture<=self.max_age_ms
        if not contiguous:self.clear_count=0
        self.last_capture=captured
        objects=observation['objects'];used=set();matches={}
        # Mutual unique best matching; ambiguity never supplies exit evidence.
        ids=list(self.active)
        scores=np.full((len(ids),len(objects)),-1.,float)
        for i,key in enumerate(ids):
            track=self.active[key]
            if not 0<captured-track['last_seen_ms']<=500:continue
            a,b,c,d=track['box'];scale=max(c-a,d-b,1.)
            for j,item in enumerate(objects):
                if item['family']!=track['family']:continue
                distance=math.dist(track['foot_point'],item['foot_point'])
                if distance>scale*1.1:continue
                scores[i,j]=.7*box_iou(track['box'],item['box'])+.3*max(0.,1-distance/(1.5*scale))
            # Weak overlapping class guesses must not fragment a strong match.
            # Prefer a plausible confident same-class observation, then confident
            # family observations, before using weak observations for retention.
            strong=[j for j,r in enumerate(objects) if scores[i,j]>=.25 and r['confidence']>=.35]
            same=[j for j in strong if objects[j]['name']==track['name']]
            preferred=same or strong
            if preferred:
                for j in range(len(objects)):
                    if j not in preferred:scores[i,j]=-1.
        for i,key in enumerate(ids):
            if not len(objects):break
            j=int(np.argmax(scores[i]));best=scores[i,j]
            if best<.25 or int(np.argmax(scores[:,j]))!=i:continue
            alternatives=np.delete(scores[i],j)
            competitors=np.delete(scores[:,j],i)
            if (len(alternatives) and best-float(alternatives.max())<.05) or (len(competitors) and best-float(competitors.max())<.05):continue
            matches[key]=j;used.add(j)
        exited=[]
        for key,track in self.active.items():
            j=matches.get(key)
            if j is None:
                track['outside_count']=0;track['currently_visible']=False
                continue
            item=objects[j];gap=captured-track['last_seen_ms']
            count=track['outside_count'] if gap<=self.max_age_ms and contiguous else 0
            outside=item['outside_with_margin'] and item['confidence']>=.35
            track.update(name=item['name'],box=list(item['box']),foot_point=item['foot_point'],last_seen_ms=captured,
                currently_visible=True,outside_count=count+1 if outside else 0)
            if track['outside_count']>=self.required_exit:exited.append(key)
        for key in exited:self.active.pop(key);self.visible_exits+=1
        for j,item in enumerate(objects):
            if j in used or not item['in_corridor'] or item['confidence']<.35:continue
            self.serial+=1;self.had_intruder=True
            self.active[self.serial]=dict(track_id=self.serial,name=item['name'],family=item['family'],box=list(item['box']),
                foot_point=item['foot_point'],last_seen_ms=captured,outside_count=0,currently_visible=True)
        intrusion=any(r['in_corridor'] and r['confidence']>=.35 for r in objects)
        if self.active or intrusion:self.clear_count=0
        else:self.clear_count+=1
        if controls.requested_motion=='stop':return finish('stop_key','STOP',valid=True)
        if self.active:
            missing=any(not r['currently_visible'] for r in self.active.values())
            return finish('hazard_missing_detection_retained' if missing else 'observed_hazard_or_exit_confirmation',
                'STOP',alarm=True,valid=True)
        if self.clear_count<self.required_clear:return finish('confirming_current_no_intrusion')
        basis='matched_visible_exit_then_confirmed_empty' if self.had_intruder else 'confirmed_no_detected_intrusion'
        return dict(finish(basis,'GO',valid=True),clearance_scope='virtual_detector_observations_not_physical_road_clear')


def restrict_motion(base,center,*,requested_side=None,forced=False):
    """Apply hazard vetoes; never turn an existing red/unknown decision into GO."""
    result=dict(base,hazard_guard_applied=True,hardware_command=None,hardware_command_sent=False,
                physical_turn_authorized=False)
    if base['action']=='STOP':return dict(result,alarm=bool(base.get('alarm') or center.get('alarm')))
    if center['action']=='STOP':return dict(result,action='STOP',alarm=True,decision_valid=True,reason='center_hazard:'+center['reason'])
    if center['action']!='GO':return dict(result,action='HOLD',decision_valid=False,reason='center_hazard_unconfirmed')
    if requested_side is not None and base['action'] in ['CHANGE_LEFT','CHANGE_RIGHT']:
        if requested_side['action']=='STOP':
            return dict(result,action='KEEP_LANE',alarm=True,reason='requested_corridor_hazard')
        if requested_side['action']!='GO':return dict(result,action='HOLD',decision_valid=False,reason='requested_corridor_unconfirmed')
    return result
