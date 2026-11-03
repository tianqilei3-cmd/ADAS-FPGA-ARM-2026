"""Independent whole/emitter residual for direction; retained path is frozen."""
from copy import deepcopy

import numpy as np
import torch

from signal_spatial_tail import SpatialTailClassifier,encode_spatial,native_descriptors
from calibrate_corrected_signal_guard_pycharm import guarded_decode
from prepare_signal_spatial_tail_pycharm import OUT


class SpatialDirectionResidual(torch.nn.Module):
    def __init__(self,reader):
        super().__init__()
        self.conv=deepcopy(reader.model.model[-1].conv)
        self.pool=torch.nn.AdaptiveAvgPool2d(1)
        self.norm=torch.nn.LayerNorm(2560)
        self.hidden=torch.nn.Linear(2572,32)
        self.out=torch.nn.Linear(32,4)
        self.register_buffer('descriptor_mean',torch.zeros(12))
        self.register_buffer('descriptor_std',torch.ones(12))
        torch.nn.init.zeros_(self.out.weight);torch.nn.init.zeros_(self.out.bias)
        self.eval()

    def forward(self,spatial,descriptors):
        n=len(spatial)
        features=self.pool(self.conv(spatial.reshape(-1,*spatial.shape[2:]))).flatten(1).reshape(n,2560)
        native=(descriptors-self.descriptor_mean)/self.descriptor_std
        return self.out(torch.nn.functional.silu(self.hidden(torch.cat([self.norm(features),native],1))))


class DirectionResidualClassifier(SpatialTailClassifier):
    def __init__(self):
        super().__init__(use_readability=False)
        state=torch.load(OUT/'direction_residual/best.pt',map_location='cpu',weights_only=True)
        self.residual=SpatialDirectionResidual(self.reader).eval();self.residual.load_state_dict(state['model'])

    def predict(self,images):
        if not images:return []
        s,f,z,q=encode_spatial(self.reader,images)
        desc=torch.from_numpy(np.stack([native_descriptors(im) for im in images]))
        with torch.inference_mode():
            delta=self.residual(s,desc)
            predictions=guarded_decode(self.types(s[:,0]),z[:,:6],self.direction(f[:,1280:])+delta,q)
        for row,change in zip(predictions,delta.tolist()):row['direction_residual_logits']=change
        return predictions
