"""Six-byte key snapshots are driver intent, never image answers."""
from dataclasses import replace
from competition_scene_logic import Controls

def decode_keys(raw):
    if len(raw)!=6 or raw[:3]!=bytes([0x55,3,2]) or sum(raw[:-1])&255!=raw[-1]:
        raise ValueError('Checked six-byte FPGA key snapshot required')
    state,sequence=raw[3:5]
    if state&128 or state&3==3:raise ValueError('Reserved key-state bits')
    scene=(state>>2)&3;direction={0:'straight',1:'left',2:'right'}[state&3]
    drive=bool(state&16);forced=bool(state&32)
    if forced and not drive:raise ValueError('Forced request without drive request')
    supported=scene in (0,1,3)
    mode={0:'signal',1:'boundary',2:'signal',3:'pedestrian'}[scene]
    controls=Controls(mode=mode,lane_id=direction+'_virtual_lane',movement=direction,
        requested_motion='go' if drive and supported else 'stop',
        requested_turn=direction if scene==1 and direction!='straight' else 'none',
        forced=forced and supported)
    return dict(state=state,sequence=sequence,scene_index=scene,supported=supported,controls=controls,
        source='remote_test_driver' if state&64 else 'physical_keys')

class KeyTracker:
    def __init__(self):self.previous=None;self.revision=0
    def update(self,snapshot):
        old=self.previous;changed=old is None or (snapshot['sequence'],snapshot['state'])!=(old['sequence'],old['state'])
        if old and old['sequence']==snapshot['sequence'] and old['state']!=snapshot['state']:
            raise ValueError('State changed without a key event')
        reset=False
        if old and snapshot['sequence']!=old['sequence']:
            delta=(snapshot['sequence']-old['sequence'])&255
            if delta>=128:
                if snapshot['sequence']==0 and not snapshot['state']&16:reset=True
                else:raise ValueError('Out-of-order or ambiguous key event')
        if changed:self.revision+=1
        self.previous=snapshot
        return dict(snapshot,changed=changed,board_reset_observed=reset,revision=self.revision)
