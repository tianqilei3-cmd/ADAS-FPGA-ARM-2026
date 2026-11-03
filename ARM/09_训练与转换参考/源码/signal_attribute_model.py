"""Experimental color-independent lamp attributes; no trusted lane assignment."""
from pathlib import Path
import cv2
import numpy as np
import torch

from light_roi_common import ROOT,input_rgb,logits_of,color_evidence

TYPE_NAMES=['vehicle_signal','pedestrian_signal','bicycle_signal']
DIRECTION_NAMES=['round','left','straight','right']


def preprocess(image,size=128):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return input_rgb(cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR),size)


def direction_input(image,size=128):
    """Normalize luminous shape brightness separately from the full housing."""
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);h,s,v=cv2.split(hsv)
    state=color_evidence(image)['state']
    if state is None:return preprocess(image,size),False
    hue=[(h<=12)|(h>=165),(h>=15)&(h<=34),(h>=35)&(h<=95)][state]
    mask=(hue&(s>=80)&(v>=80)).astype(np.uint8)
    joined=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    n,labels,stats,_=cv2.connectedComponentsWithStats(joined,8)
    if n<=1:return preprocess(image,size),False
    component=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
    x,y,w,height,area=map(int,stats[component])
    if area<4:return preprocess(image,size),False
    pad=max(2,round(max(w,height)*.2));left,top=max(0,x-pad),max(0,y-pad)
    right,bottom=min(image.shape[1],x+w+pad),min(image.shape[0],y+height+pad)
    # Use V instead of RGB luminance: red and green arrows retain similar brightness.
    shape=(v*(labels==component).astype(np.uint8))[top:bottom,left:right]
    return input_rgb(cv2.cvtColor(shape,cv2.COLOR_GRAY2BGR),size),True


def task_logits(model,x,u=None):
    if u is None:
        z=logits_of(model(x));return z[:,:3],z[:,3:]
    z=logits_of(model(torch.cat([x,u],dim=0)));n=len(x)
    return z[:n,:3],z[n:,3:]


def make_model():
    from ultralytics import YOLO
    from ultralytics.nn.tasks import ClassificationModel
    model=YOLO(str(ROOT/'yolov8n-cls.pt')).model.float()
    ClassificationModel.reshape_outputs(model,7)
    model.names=dict(enumerate(TYPE_NAMES+DIRECTION_NAMES));model.yaml['nc']=7
    model.requires_grad_(True)
    return model


def reject(p,threshold,margin):
    order=np.argsort(-np.asarray(p));a,b=map(int,order[:2])
    return a if p[a]>=threshold and p[a]-p[b]>=margin else -1


class AttributeClassifier:
    def __init__(self,weights):
        self.checkpoint=torch.load(weights,map_location='cpu',weights_only=False)
        assert self.checkpoint['type_names']==TYPE_NAMES and self.checkpoint['direction_names']==DIRECTION_NAMES
        self.model=make_model();self.model.load_state_dict(self.checkpoint['model_state']);self.model.eval()
    def predict(self,images):
        if not images:return []
        x=np.stack([preprocess(im) for im in images]).transpose(0,3,1,2).copy()
        emitting=self.checkpoint.get('input_mode')=='whole_plus_emitting_v2'
        shapes=[direction_input(im) for im in images] if emitting else None
        u=torch.from_numpy(np.stack([p[0] for p in shapes]).transpose(0,3,1,2).copy()).float()/255 if emitting else None
        with torch.inference_mode():
            a,b=task_logits(self.model,torch.from_numpy(x).float()/255,u)
            types=a.softmax(1).numpy();directions=b.softmax(1).numpy()
        rows=[]
        for index,(p,q) in enumerate(zip(types,directions)):
            t=reject(p,.85,.25);d=reject(q,.80,.20)
            # Direction of a non-vehicle or uncertain lamp cannot establish vehicle control.
            if t!=0:d=-1
            if emitting and not shapes[index][1]:d=-1
            rows.append({'lamp_type':TYPE_NAMES[t] if t>=0 else 'unknown',
                         'shape':('round' if d==0 else 'arrow' if d>0 else 'unknown'),
                         'control_direction':DIRECTION_NAMES[d] if d>0 else 'unknown',
                         'type_probabilities':p.tolist(),'direction_probabilities':q.tolist(),
                         'type_argmax':TYPE_NAMES[int(p.argmax())],'direction_argmax':DIRECTION_NAMES[int(q.argmax())],
                         'trusted_for_lane_association':False,
                         'scope':'Experimental attributes, limited reviewed data; right trained with mirrors only'})
        return rows
