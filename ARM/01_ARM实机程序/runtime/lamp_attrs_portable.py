"""Portable frozen lamp type/direction reader. No lane or movement authority."""
from pathlib import Path
import numpy as np
from lamp_attrs_preprocess import preprocess,direction_input,native_descriptors

TYPE_NAMES=['vehicle_signal','pedestrian_signal','bicycle_signal','other_object']
DIRECTION_NAMES=['round','left','straight','right']


def softmax(value):
    value=np.asarray(value,dtype=np.float32)
    p=np.exp(value-value.max(axis=1,keepdims=True))
    return p/p.sum(axis=1,keepdims=True)


def reject(p,threshold,margin):
    a,b=map(int,np.argsort(-p)[:2])
    return a if p[a]>=threshold and p[a]-p[b]>=margin else -1


def decode(types,directions,quality):
    n=types.shape[1]//2;a,b=softmax(types[:,:n]),softmax(types[:,n:]);d=softmax(directions)
    rows=[]
    for pa,pb,pd,q in zip(a,b,d,quality):
        ta,tb=reject(pa,.85,.25),reject(pb,.85,.25)
        typ=ta if ta==tb and ta>=0 else -1
        direction=reject(pd,.8,.2) if typ==0 and q else -1
        rows.append(dict(lamp_type=TYPE_NAMES[typ] if typ>=0 else 'unknown',type_guard=typ,
            type_head_probabilities=[pa.tolist(),pb.tolist()],direction_probabilities=pd.tolist(),
            type_argmax=TYPE_NAMES[int(((pa+pb)/2).argmax())],direction_argmax=DIRECTION_NAMES[int(pd.argmax())],
            direction_guard=direction,shape='round' if direction==0 else 'arrow' if direction>0 else 'unknown',
            control_direction=DIRECTION_NAMES[direction] if direction>0 else 'unknown',
            emitting_available=bool(q),physical_lane_association_verified=False,trusted_for_lane_association=False))
    return rows


def guarded_decode(types,original,directions,quality,delta):
    predictions=decode(types,directions,quality);old=decode(original,directions,quality)
    for p,o,change in zip(predictions,old,delta.tolist()):
        pa,pb=p['type_head_probabilities']
        agree_motor=int(np.asarray(pa).argmax())==int(np.asarray(pb).argmax())==0
        p.update(original_type_retained=False,promotion_without_emission_blocked=False,
                 original_type_head_probabilities=o['type_head_probabilities'])
        if o['type_guard']==0 and p['type_guard']<0 and agree_motor:
            p.update(type_guard=0,lamp_type='vehicle_signal',original_type_retained=True,
                     direction_guard=o['direction_guard'],shape=o['shape'],control_direction=o['control_direction'])
        if o['type_guard']!=0 and p['type_guard']==0 and not p['emitting_available']:
            p.update(type_guard=-1,lamp_type='unknown',promotion_without_emission_blocked=True,
                     direction_guard=-1,shape='unknown',control_direction='unknown')
        p['direction_residual_logits']=change
    return predictions


def inputs(images):
    whole=np.stack([preprocess(im) for im in images]).transpose(0,3,1,2).astype(np.float32)/255
    shapes=[direction_input(im) for im in images]
    emitter=np.stack([s[0] for s in shapes]).transpose(0,3,1,2).astype(np.float32)/255
    descriptors=np.stack([native_descriptors(im) for im in images]).astype(np.float32)
    quality=[s[1] for s in shapes]
    return dict(whole=whole,emitter=emitter,descriptors=descriptors),quality


class PortableAttributes:
    def __init__(self,model=None,threads=1):
        import onnxruntime as ort
        options=ort.SessionOptions();options.intra_op_num_threads=threads;options.inter_op_num_threads=1
        options.add_session_config_entry('session.intra_op.allow_spinning','0')
        model=Path(model) if model else Path(__file__).parent/'models/lamp_attributes.onnx'
        self.session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])

    def predict(self,images,batch=4):
        rows=[]
        for start in range(0,len(images),batch):
            values,quality=inputs(images[start:start+batch])
            types,original,directions,delta=self.session.run(None,values)
            rows+=guarded_decode(types,original,directions,quality,delta)
        return rows
