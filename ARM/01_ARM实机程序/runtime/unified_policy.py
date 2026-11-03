"""Image-derived decisions and key-derived driver intents remain distinct."""
import math,time
from lane_support import decide as lane_decide
from signal_engine import combined_decision


def choose(frame,controls,key,boundaries,risk,selected,controller,now_ms):
    if not key.get('supported',False):
        result=dict(action='STOP',alarm=False,decision_valid=False,reason='navigation_not_implemented')
    elif controls.mode=='signal':
        result=combined_decision(frame,controls,risk,selected,controller,now_ms)
    elif controls.mode=='boundary':
        result=lane_decide(frame,controls,boundaries,risk,controller,now_ms)
    else:
        result=dict(action='GO' if risk['action']=='GO' and risk['decision_valid'] else 'STOP',
            alarm=risk['alarm'],decision_valid=risk['decision_valid'],reason='hazard:'+risk['reason'])
    return dict(result,physical_lane_association_verified=False,hardware_motion_control_connected=False)


def context(controls,key,boundaries,selected,decision,observation,risk,captured,unchanged,quality):
    side=controls.requested_turn;line=(boundaries or {}).get(side,{})
    line_valid=bool(line.get('state_valid') and line.get('simulation_association_valid') and line.get('line_type') in ('solid','dashed'))
    lamp_valid=bool(key.get('scene_index')==0 and selected.get('selected_valid') and selected.get('simulation_association_valid'))
    intrusion=any(r['in_corridor'] and r['confidence']>=.35 for r in observation['objects'])
    retained=bool(risk['active_tracks']) and not intrusion
    valid=int(quality['valid'])|(64 if quality['valid'] else 0)|(32 if unchanged else 0)
    valid|=2 if lamp_valid and unchanged else 0
    valid|=8 if line_valid and unchanged else 0
    driver=int(controls.requested_motion=='go')|(int(controls.forced)<<1)|(int(side!='none')<<2)|(int(side=='right')<<3)|(int(controls.forced and side!='none')<<4)
    permit=unchanged and quality['valid'] and decision['decision_valid'] and decision['action'] in ('GO','KEEP_LANE','CHANGE_LEFT','CHANGE_RIGHT')
    return dict(camera_id=0,scene=key.get('scene_index',0)+1,valid_flags=valid,
        risk_flags=int(intrusion)|(int(retained)<<3),light_state={'red':1,'yellow':2,'green':3,'off':4}.get(selected.get('selected_state'),0) if lamp_valid else 0,
        navigation=0,control_flags=driver if unchanged else 0,movement={'left':1,'straight':2,'right':3}[controls.movement],
        boundary={'dashed':1,'solid':2}.get(line.get('line_type'),0) if line_valid else 0,
        epoch=key.get('sequence',0),age_units=min(255,math.ceil((time.monotonic()-captured)*200)),arm_action=2 if permit else 1)


def legal_driver_state(state):
    return type(state)is int and (state==255 or (0<=state<64 and state&3!=3 and (state>>2)&3!=2 and not(state&32 and not state&16)))
