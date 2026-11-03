"""Add a trained visible-emission head while preserving all four color logits."""
from pathlib import Path

import numpy as np
import torch

from light_roi_common import ROOT,input_rgb,logits_of,make_model,NAMES

FORMAT='color4_plus_visible_emission_v1'
POINTER=ROOT/'runs/signal_remaining_repair_20261017/emission_guard.json'


class EmissionGuardClassifier:
    def __init__(self,weights):
        self.checkpoint=torch.load(weights,map_location='cpu',weights_only=False)
        assert self.checkpoint['format']==FORMAT
        self.model=make_model();old=self.model.model[-1].linear
        self.model.model[-1].linear=torch.nn.Linear(old.in_features,6)
        self.model.load_state_dict(self.checkpoint['model_state']);self.model.eval()
        self.size=128

    def predict_with_guard(self,crops,batch=32):
        colors=[];valid=[]
        for start in range(0,len(crops),batch):
            rgb=np.stack([input_rgb(c,128) for c in crops[start:start+batch]]).transpose(0,3,1,2).copy()
            with torch.inference_mode():
                z=logits_of(self.model(torch.from_numpy(rgb).float()/255))
                colors.extend(z[:,:4].softmax(1).numpy());valid.extend(z[:,4:].softmax(1)[:,1].numpy())
        return np.asarray(colors,np.float32).reshape(-1,4),np.asarray(valid,np.float32)

    def predict(self,crops,batch=32):return self.predict_with_guard(crops,batch)[0]
