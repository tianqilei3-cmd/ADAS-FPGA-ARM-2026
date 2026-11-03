"""Two trained type heads share an identical trunk; disagreement is unknown.

This is a reviewed-development candidate, not a verified physical-lane selector.
The v3 and v4 linear heads were trained locally; the direction head is frozen v2.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from light_roi_common import ROOT, logits_of
from signal_attribute_model import (AttributeClassifier, make_model, preprocess,
                                    direction_input, reject, TYPE_NAMES, DIRECTION_NAMES)
from train_light_roi_color import save_checkpoint, save_json

FORMAT='signal_attributes_consensus_v1'
POINTER=ROOT/'runs/signal_attributes_20261017/local_consensus_candidate.json'


def build():
    names=['latest_emitting','latest_red_pedestrian_v3','latest_type_only_v4']
    pointers=[json.loads((POINTER.parent/(n+'.json')).read_text(encoding='utf-8')) for n in names]
    states=[torch.load(p['best'],map_location='cpu',weights_only=False) for p in pointers]
    for s in states:
        assert s['input_mode']=='whole_plus_emitting_v2'
        assert s['type_names']==TYPE_NAMES and s['direction_names']==DIRECTION_NAMES
    trunk=states[0]['model_state'];linear=['model.9.linear.weight','model.9.linear.bias']
    for state in states[1:]:
        assert set(trunk)==set(state['model_state'])
        assert all(torch.equal(v,state['model_state'][k]) for k,v in trunk.items() if k not in linear), 'Trunks differ: cannot merge'
    assert all(torch.equal(trunk[k][3:],states[2]['model_state'][k][3:]) for k in linear)
    merged={k:v.clone() for k,v in trunk.items()}
    for k in linear:
        merged[k]=torch.cat([states[1]['model_state'][k][:3],states[2]['model_state'][k][:3],trunk[k][3:]],0)
    sources=[dict(pointer=str(POINTER.parent/(n+'.json')),weights=p['best'],epoch=s['epoch'],
                  sha256=hashlib.sha256(Path(p['best']).read_bytes()).hexdigest()) for n,p,s in zip(names,pointers,states)]
    digest=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    folder=ROOT/'runs/adas'/('signal_attributes_consensus_'+digest[:12]);folder.mkdir(exist_ok=True,parents=True)
    weights=folder/'weights/best.pt'
    weights.parent.mkdir(exist_ok=True)
    save_checkpoint(weights,dict(format=FORMAT,model_state=merged,type_names=TYPE_NAMES,direction_names=DIRECTION_NAMES,
         input_mode='whole_plus_emitting_v2',sources=sources,plan=states[2]['plan'],
         backbone_identical_verified=True,direction_identical_to_v2_verified=True,
         policy='Both type heads must pass p>=.85,margin>=.25 and agree; otherwise unknown. Direction unchanged v2.',
         lane_association_verified=False,board_verified=False))
    pointer=dict(state='completed',run=str(folder),best=str(weights),format=FORMAT,
                 sources=sources,sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                 experimental=True,board_verified=False,lane_association_verified=False)
    save_json(POINTER,pointer)
    return pointer


class ConsensusAttributeClassifier:
    def __init__(self,weights):
        self.checkpoint=torch.load(weights,map_location='cpu',weights_only=False)
        assert self.checkpoint['format']==FORMAT
        self.model=make_model()
        old=self.model.model[-1].linear
        self.model.model[-1].linear=torch.nn.Linear(old.in_features,10)
        self.model.load_state_dict(self.checkpoint['model_state']);self.model.eval()

    def probabilities(self,images):
        if not images:return np.empty((0,3)),np.empty((0,3)),np.empty((0,4)),[]
        whole=np.stack([preprocess(im) for im in images]).transpose(0,3,1,2).copy()
        shapes=[direction_input(im) for im in images]
        emission=np.stack([s[0] for s in shapes]).transpose(0,3,1,2).copy()
        tensor=torch.from_numpy(np.concatenate([whole,emission],0)).float()/255
        with torch.inference_mode():
            z=logits_of(self.model(tensor));n=len(images)
            a=z[:n,:3].softmax(1).numpy();b=z[:n,3:6].softmax(1).numpy();d=z[n:,6:].softmax(1).numpy()
        return a,b,d,[s[1] for s in shapes]

    def predict(self,images):
        a,b,d,quality=self.probabilities(images);rows=[]
        for pa,pb,pd,good in zip(a,b,d,quality):
            ta,tb=reject(pa,.85,.25),reject(pb,.85,.25)
            t=ta if ta==tb and ta>=0 else -1
            direction=reject(pd,.8,.2) if t==0 and good else -1
            mean=(pa+pb)/2
            rows.append(dict(lamp_type=TYPE_NAMES[t] if t>=0 else 'unknown',
                shape='round' if direction==0 else 'arrow' if direction>0 else 'unknown',
                control_direction=DIRECTION_NAMES[direction] if direction>0 else 'unknown',
                type_probabilities=mean.tolist(),type_head_probabilities=[pa.tolist(),pb.tolist()],
                type_argmax=TYPE_NAMES[int(mean.argmax())],type_head_guard_ids=[ta,tb],
                type_reason='agreement' if t>=0 else 'head_disagreement_or_low_confidence',
                direction_probabilities=pd.tolist(),direction_argmax=DIRECTION_NAMES[int(pd.argmax())],
                trusted_for_lane_association=False,
                scope='Experimental local consensus; unknown on type disagreement; no verified lane mapping.'))
        return rows


def load_candidate(weights):
    state=torch.load(weights,map_location='cpu',weights_only=False)
    return ConsensusAttributeClassifier(weights) if state.get('format')==FORMAT else AttributeClassifier(weights)


if __name__=='__main__':print(json.dumps(build(),ensure_ascii=False,indent=2))
