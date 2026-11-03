"""Experimental type adaptation and emitter-only direction, with fixed gates."""
import numpy as np
import torch

from light_roi_common import ROOT, read_image
from prepare_type_direction_repair_pycharm import TYPE_NAMES, digest
from signal_attribute_model import reject, DIRECTION_NAMES
from signal_attribute_consensus import ConsensusAttributeClassifier
from signal_dual_view_residual import encode

PARENT = ROOT / 'runs/adas/signal_attributes_consensus_20fc4d9bebcd/weights/best.pt'
PARENT_SHA = 'b55b10e57cef0b2e090932824d2733da499d13956621175325f41115cc99a1ad'


def make_type_head(reader):
    old = reader.model.model[-1].linear
    head = torch.nn.Linear(1280, 8)
    with torch.no_grad():
        for new_start, old_start in ((0,0),(4,3)):
            head.weight[new_start:new_start+3].copy_(old.weight[old_start:old_start+3])
            head.bias[new_start:new_start+3].copy_(old.bias[old_start:old_start+3])
            head.weight[new_start+3].zero_(); head.bias[new_start+3] = -6.
    return head


def make_direction_head(reader):
    old = reader.model.model[-1].linear; head = torch.nn.Linear(1280,4)
    with torch.no_grad():
        head.weight.copy_(old.weight[6:10]); head.bias.copy_(old.bias[6:10])
    return head


def decode(type_logits, direction_logits, quality):
    n = type_logits.shape[1]//2
    a,b = type_logits[:,:n].softmax(1).detach().cpu().numpy(), type_logits[:,n:].softmax(1).detach().cpu().numpy()
    d = direction_logits.softmax(1).detach().cpu().numpy()
    output=[]
    for pa,pb,pd,q in zip(a,b,d,quality.flatten().tolist()):
        ta,tb=reject(pa,.85,.25),reject(pb,.85,.25)
        t=ta if ta==tb and ta>=0 else -1
        di=reject(pd,.8,.2) if t==0 and q else -1
        output.append(dict(lamp_type=TYPE_NAMES[t] if t>=0 else 'unknown',type_guard=t,
            type_head_probabilities=[pa.tolist(),pb.tolist()],direction_probabilities=pd.tolist(),
            type_argmax=TYPE_NAMES[int(((pa+pb)/2).argmax())],direction_argmax=DIRECTION_NAMES[int(pd.argmax())],
            direction_guard=di,shape='round' if di==0 else 'arrow' if di>0 else 'unknown',
            control_direction=DIRECTION_NAMES[di] if di>0 else 'unknown',
            emitting_available=bool(q),physical_lane_association_verified=False,trusted_for_lane_association=False))
    return output


def summarize(rows, predictions):
    result=dict(records=len(rows),vehicle_known=0,vehicle_type_accepted=0,negative_known=0,negative_vehicle_accepted=0,
                negative_direction_accepted=0,known_direction=0,raw_direction_correct=0,
                correct=0,wrong=0,unknown=0,type_correct=0,type_unknown=0,by_direction={},negative_by_kind={})
    for r,p in zip(rows,predictions):
        t=r['type_id']; di=r['direction_id']; accepted=p['type_guard']
        result['type_correct'] += accepted==t
        result['type_unknown'] += accepted<0
        if t==0:
            result['vehicle_known']+=1; result['vehicle_type_accepted']+=accepted==0
        else:
            result['negative_known']+=1; result['negative_vehicle_accepted']+=accepted==0
            result['negative_direction_accepted']+=p['direction_guard']>=0
            stat=result['negative_by_kind'].setdefault(TYPE_NAMES[t],dict(known=0,false_vehicle=0))
            stat['known']+=1; stat['false_vehicle']+=accepted==0
        if di<0: continue
        value=p['direction_guard']; raw=p['direction_argmax']; expected=DIRECTION_NAMES[di]
        result['known_direction']+=1; result['raw_direction_correct']+=raw==expected
        result['correct']+=value==di; result['wrong']+=value>=0 and value!=di; result['unknown']+=value<0
        stat=result['by_direction'].setdefault(expected,dict(known=0,correct=0,wrong=0,unknown=0))
        stat['known']+=1;stat['correct']+=value==di;stat['wrong']+=value>=0 and value!=di;stat['unknown']+=value<0
    return result


class RepairAttributeClassifier:
    def __init__(self,type_weights,direction_weights=None):
        if digest(PARENT)!=PARENT_SHA: raise RuntimeError('Retained parent changed')
        self.reader=ConsensusAttributeClassifier(PARENT)
        state=torch.load(type_weights,map_location='cpu',weights_only=True)
        if state['format']!='type_dual4_repair_v1': raise ValueError('Wrong type checkpoint')
        self.types=make_type_head(self.reader).eval();self.types.load_state_dict(state['model'])
        self.direction=None
        if direction_weights:
            state=torch.load(direction_weights,map_location='cpu',weights_only=True)
            if state['format']!='emitting_direction_retention_v1':raise ValueError('Wrong direction checkpoint')
            self.direction=make_direction_head(self.reader).eval();self.direction.load_state_dict(state['model'])

    def predict(self,images):
        if not images:return []
        f,z,q=encode(self.reader,images)
        with torch.inference_mode():
            t=self.types(f[:,:1280]);d=z[:,6:] if self.direction is None else self.direction(f[:,1280:])
        return decode(t,d,q)
