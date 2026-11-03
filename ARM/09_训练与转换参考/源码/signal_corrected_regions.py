"""Corrected region classifier and detail-preserving luminous preprocessing."""
import cv2
import numpy as np
import torch

from light_roi_common import color_evidence, input_rgb, logits_of
from signal_attribute_model import preprocess
from signal_dual_view_residual import encode
from signal_type_direction_repair import PARENT, PARENT_SHA, make_type_head, make_direction_head, decode
from signal_attribute_consensus import ConsensusAttributeClassifier
from prepare_type_direction_repair_pycharm import digest
from prepare_signal_hard_examples_pycharm import OUT


def detail_direction_input(image, size=128):
    """Close to locate the component only; never paint closed-in shape pixels."""
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
    native=mask&(labels==component).astype(np.uint8)
    if area<4 or int(native.sum())<4:return preprocess(image,size),False
    pad=max(2,round(max(w,height)*.2))
    left,top=max(0,x-pad),max(0,y-pad)
    right,bottom=min(image.shape[1],x+w+pad),min(image.shape[0],y+height+pad)
    shape=(v*native)[top:bottom,left:right]
    return input_rgb(cv2.cvtColor(shape,cv2.COLOR_GRAY2BGR),size),True


def encode_detail(reader, images):
    if not images:
        return torch.empty((0,1280)),torch.empty((0,1))
    shapes=[detail_direction_input(im) for im in images]
    tensor=torch.from_numpy(np.stack([r[0] for r in shapes]).transpose(0,3,1,2).copy()).float()/255
    captured=[]
    hook=reader.model.model[-1].linear.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
    try:
        with torch.inference_mode():logits_of(reader.model(tensor))
    finally:hook.remove()
    assert len(captured)==1
    return captured[0],torch.tensor([r[1] for r in shapes],dtype=torch.float32).reshape(-1,1)


class CorrectedRegionClassifier:
    def __init__(self, direction_mode='detail'):
        if digest(PARENT)!=PARENT_SHA:raise RuntimeError('Retained encoder changed')
        self.reader=ConsensusAttributeClassifier(PARENT)
        self.types=make_type_head(self.reader).eval()
        state=torch.load(OUT/'type_training/best.pt',map_location='cpu',weights_only=True)
        self.types.load_state_dict(state['model'])
        self.direction_mode=direction_mode
        self.direction=make_direction_head(self.reader).eval()
        path=OUT/'direction_training/best.pt' if direction_mode=='detail' else OUT.parent/'signal_type_direction_repair_20261022/direction/best.pt'
        self.direction.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model'])

    def predict(self, images):
        if not images:return []
        f,z,q=encode(self.reader,images)
        if self.direction_mode=='detail':
            d,q=encode_detail(self.reader,images)
        else:d=f[:,1280:]
        with torch.inference_mode():
            return decode(self.types(f[:,:1280]),self.direction(d),q)
