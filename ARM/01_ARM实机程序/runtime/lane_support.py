import json,math,time
from dataclasses import asdict
from pathlib import Path
import cv2,numpy as np
from vision_common import draw,write_image
def atomic(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(path)

def snapshot_json(snapshot):
    return dict(snapshot,controls=asdict(snapshot['controls']))

def save_evidence(folder,count,raw_image,image,canvas):
    folder.mkdir(exist_ok=True)
    write_image(folder/(f'{count:06d}_raw.png'),raw_image)
    write_image(folder/(f'{count:06d}_processed.png'),image)
    write_image(folder/(f'{count:06d}_preview.jpg'),canvas)

def decide(frame,controls,boundaries,hazard,controller,now_ms):
    if controls.mode=='boundary':
        result=controller.update(frame,controls,boundaries=boundaries,now_ms=now_ms)
        if result['action']=='HOLD':result=dict(result,action='STOP')
    else:
        result=dict(action='STOP',alarm=False,decision_valid=True,
            reason='select_boundary_scene_with_KEY2' if controls.mode!='pedestrian' else 'risk_scene_not_tested_in_this_lane_run')
    if hazard['action']=='STOP':
        result.update(action='STOP',alarm=bool(hazard['alarm']),decision_valid=bool(hazard['decision_valid']),reason='hazard:'+hazard['reason'])
    elif result['action'] in ('KEEP_LANE','CHANGE_LEFT','CHANGE_RIGHT') and not (hazard['action']=='GO' and hazard['decision_valid']):
        result.update(action='STOP',decision_valid=False,reason='hazard_clearance_unconfirmed')
    result.update(physical_lane_association_verified=False,hardware_motion_control_connected=False)
    return result

def make_context(controls,boundaries,decision,observation,hazard,source_epoch,captured,key_unchanged=True):
    side=controls.requested_turn
    value=boundaries.get(side,{}) if boundaries and side in ('left','right') else {}
    boundary_valid=bool(value.get('state_valid') and value.get('simulation_association_valid') and value.get('line_type') in ('solid','dashed'))
    intrusion=any(r['in_corridor'] and r['confidence']>=.35 for r in observation['objects'])
    retained=bool(hazard['active_tracks']) and not intrusion
    valid=1|64|(32 if key_unchanged else 0)|(8 if boundary_valid and key_unchanged else 0)
    flags=int(controls.requested_motion=='go')|(int(controls.forced)<<1)|(int(side!='none')<<2)|(int(side=='right')<<3)|(int(controls.forced and side!='none')<<4)
    permitted=key_unchanged and decision['decision_valid'] and decision['action'] in ('KEEP_LANE','CHANGE_LEFT','CHANGE_RIGHT')
    return dict(camera_id=0,scene=2,valid_flags=valid,risk_flags=int(intrusion)|(int(retained)<<3),
        light_state=0,navigation=0,control_flags=flags if key_unchanged else 0,
        movement={'left':1,'straight':2,'right':3}[controls.movement],boundary={'dashed':1,'solid':2}.get(value.get('line_type'),0),
        epoch=source_epoch,age_units=min(255,math.ceil((time.monotonic()-captured)*200)),arm_action=2 if permitted else 1)

def render(image,roads,paths,controls,selection,decision,key,receipt,feedback,age):
    canvas=cv2.resize(draw(image,[r for r in roads if r['name']!='traffic_light']),(1280,720))
    sx,sy=1280/image.shape[1],720/image.shape[0]
    for path in paths:
        points=np.asarray(path['points_native'],np.float32)
        color={'solid':(30,200,255),'dashed':(255,210,30)}.get(path['line_type'],(160,160,160))
        if len(points)>1:
            pixels=np.rint(points*[sx,sy]).astype(np.int32)
            cuts=np.flatnonzero(np.diff(points[:,1])>image.shape[0]/288*1.5)+1
            strokes=[part.reshape(-1,1,2) for part in np.split(pixels,cuts) if len(part)>1]
            if strokes:cv2.polylines(canvas,strokes,False,color,3)
    cv2.rectangle(canvas,(0,0),(1280,130),(22,26,30),-1)
    color=(40,70,255) if decision['alarm'] else (40,230,100) if decision['action'].startswith('CHANGE') else (80,180,250)
    left=(selection or {}).get('left',{}).get('line_type','unknown');right=(selection or {}).get('right',{}).get('line_type','unknown')
    text=[
        f"ARM {decision['action']} | ALARM {int(decision['alarm'])} | {decision['reason']}",
        f"{'REMOTE TEST DRIVER' if key.get('source')=='remote_test_driver' else 'PHYSICAL KEY'} seq={key.get('sequence','?')} | scene={controls.mode} | turn={controls.requested_turn} | drive={controls.requested_motion} | force={int(controls.forced)}",
        f"CURRENT IMAGE left={left} right={right} | result age {age:.0f} ms | virtual image-center lane",
    ]
    for i,line in enumerate(text):cv2.putText(canvas,line,(12,30+i*35),0,.62 if i==0 else .54,color if i==0 else (240,240,240),2 if i==0 else 1,cv2.LINE_AA)
    cv2.rectangle(canvas,(0,620),(1280,720),(22,26,30),-1)
    link='CONFIRMED' if receipt and receipt['status_acknowledged'] else 'UNCONFIRMED / STOP'
    extra=f"action={feedback['action']} alarm={feedback['alarm_code']} frame={feedback['frame_id']}" if feedback else 'no identified feedback'
    cv2.putText(canvas,'UART '+link+' | FPGA '+extra,(12,646),0,.58,(80,230,100) if link=='CONFIRMED' else (80,90,255),1)
    cv2.putText(canvas,'KEY2 scene | KEY3 left | KEY4 straight | KEY5 right | KEY6 drive/stop | KEY7 force',(12,676),0,.55,(240,240,240),1)
    cv2.putText(canvas,'TEST: STOP before choosing next direction. Physical lane mapping / vehicle actuation unverified.',(12,705),0,.51,(220,220,220),1)
    return canvas
