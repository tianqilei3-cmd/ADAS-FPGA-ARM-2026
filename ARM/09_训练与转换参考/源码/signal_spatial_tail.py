"""Experimental spatial type adaptation plus independent directional readability.

The encoder prefix, original direction projection and color weights are frozen.
No class or output establishes physical ego-lane association or motion authority.
"""
from copy import deepcopy

import cv2
import numpy as np
import torch

from light_roi_common import ROOT,color_evidence,logits_of
from signal_attribute_model import preprocess,direction_input
from signal_attribute_consensus import ConsensusAttributeClassifier
from signal_type_direction_repair import PARENT,PARENT_SHA,make_direction_head
from calibrate_corrected_signal_guard_pycharm import guarded_decode
from prepare_signal_spatial_tail_pycharm import OUT,PREVIOUS
from prepare_type_direction_repair_pycharm import digest

DIRECTION_WEIGHTS=ROOT/'runs/signal_type_direction_repair_20261022/direction/best.pt'


class SpatialTypeTail(torch.nn.Module):
    def __init__(self,reader):
        super().__init__()
        self.conv=deepcopy(reader.model.model[-1].conv)
        self.pool=torch.nn.AdaptiveAvgPool2d(1)
        self.linear=torch.nn.Linear(1280,8)
        state=torch.load(PREVIOUS/'control_guard/selected_types.pt',map_location='cpu',weights_only=True)
        self.linear.load_state_dict(state['model'])
        self.eval()

    def forward(self,x):
        return self.linear(self.pool(self.conv(x)).flatten(1))


def encode_spatial(reader,images):
    whole=np.stack([preprocess(im) for im in images]).transpose(0,3,1,2).copy()
    shapes=[direction_input(im) for im in images]
    emitting=np.stack([s[0] for s in shapes]).transpose(0,3,1,2).copy()
    tensor=torch.from_numpy(np.concatenate([whole,emitting])).float()/255
    spatial=[];pooled=[]
    head=reader.model.model[-1]
    a=head.conv.register_forward_pre_hook(lambda module,args:spatial.append(args[0].detach().clone()))
    b=head.linear.register_forward_pre_hook(lambda module,args:pooled.append(args[0].detach().clone()))
    try:
        with torch.inference_mode():z=logits_of(reader.model(tensor)).clone()
    finally:a.remove();b.remove()
    assert len(spatial)==len(pooled)==1
    n=len(images)
    return (torch.stack([spatial[0][:n],spatial[0][n:]],1),
         torch.cat([pooled[0][:n],pooled[0][n:]],1),torch.cat([z[:n,:6],z[n:,6:]],1),
         torch.tensor([s[1] for s in shapes],dtype=torch.float32).reshape(-1,1))


def native_descriptors(image):
    h,w=image.shape[:2]
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);hue,s,v=cv2.split(hsv)
    state=color_evidence(image)['state']
    select=[(hue<=12)|(hue>=165),(hue>=15)&(hue<=34),(hue>=35)&(hue<=95)]
    mask=((s>=80)&(v>=80)&(select[state] if state is not None else False)).astype(np.uint8)
    joined=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    n,labels,stats,_=cv2.connectedComponentsWithStats(joined,8)
    bw,bh,extent=0,0,0.
    if n>1:
        index=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));_,_,bw,bh,_=stats[index]
        extent=float((mask&(labels==index)).sum())/max(1,int(bw*bh))
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    filled=float(((joined>0)&(mask==0)).sum())/max(1,int(mask.sum()))
    return np.asarray([np.log1p(w),np.log1p(h),np.log1p(mask.sum()),float(mask.mean()),filled,
         float(s.mean()/255),float(v.mean()/255),float(v.std()/255),float(cv2.Laplacian(gray,cv2.CV_32F).var()),
         np.log1p(bw),np.log1p(bh),extent],dtype=np.float32)


def readability_transform(features,descriptors,transform):
    pc=(features-transform['feature_mean'])@transform['projection']
    raw=torch.cat([pc,descriptors],1)
    return (raw-transform['mean'])/transform['std']


def apply_readability(predictions,probabilities,threshold):
    for p,probability in zip(predictions,probabilities.tolist()):
        p.update(readability_probability=float(probability),readability_threshold=float(threshold),readability_vetoed=False)
        if p['direction_guard']>=0 and probability<threshold:
            p.update(direction_guard=-1,shape='unknown',control_direction='unknown',readability_vetoed=True)
    return predictions


class SpatialTailClassifier:
    def __init__(self,use_readability=True):
        if digest(PARENT)!=PARENT_SHA:raise RuntimeError('Retained parent changed')
        self.reader=ConsensusAttributeClassifier(PARENT)
        self.types=SpatialTypeTail(self.reader).eval()
        state=torch.load(OUT/'types/best.pt',map_location='cpu',weights_only=True)
        self.types.load_state_dict(state['model'])
        self.direction=make_direction_head(self.reader).eval()
        self.direction.load_state_dict(torch.load(DIRECTION_WEIGHTS,map_location='cpu',weights_only=True)['model'])
        self.use_readability=use_readability
        if use_readability:
            state=torch.load(OUT/'readability/best.pt',map_location='cpu',weights_only=True)
            self.transform=state['transform'];self.threshold=state['threshold']
            self.quality=torch.nn.Linear(len(self.transform['mean']),2).eval();self.quality.load_state_dict(state['model'])

    def predict(self,images):
        if not images:return []
        s,f,z,q=encode_spatial(self.reader,images)
        with torch.inference_mode():
            predictions=guarded_decode(self.types(s[:,0]),z[:,:6],self.direction(f[:,1280:]),q)
            if self.use_readability:
                descriptors=torch.from_numpy(np.stack([native_descriptors(im) for im in images]))
                quality=self.quality(readability_transform(f,descriptors,self.transform)).softmax(1)[:,1]
                return apply_readability(predictions,quality,self.threshold)
        return predictions
