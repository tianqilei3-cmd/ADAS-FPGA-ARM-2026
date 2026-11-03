"""Experimental lane-conditioned scorer; no controller or hardware commands."""
import math
from pathlib import Path

import numpy as np
import torch

from expanded_relation_data import digest
from signal_lane_topology_data import pair_geometry
from visual_lane_relation_features import (FrozenLaneVisualEncoder,fit_projection,apply_projection)

FORMAT='frozen_visual_lane_relation_v1'
SEED=2026102140


def make_head(dimensions):
    return torch.nn.Sequential(torch.nn.Linear(dimensions,32),torch.nn.ReLU(),
                               torch.nn.Dropout(.2),torch.nn.Linear(32,1))


def fit_training_transform(cache,mode):
    if mode not in ('geometry','visual'):raise ValueError('Unsupported relation mode')
    indices=[i for i,r in enumerate(cache['metadata']) if r['split']=='train']
    if not indices:raise ValueError('Training partition required')
    transform=dict(mode=mode,fit_pair_indices=torch.tensor(indices,dtype=torch.int64))
    parts=[cache['geometry']]
    if mode=='visual':
        seen=set();head_indices=[]
        for i in indices:
            r=cache['metadata'][i];key=r['frame_key'],r['variant'],r['element_index']
            if key not in seen:seen.add(key);head_indices.append(i)
        transform['head_fit_indices']=torch.tensor(head_indices,dtype=torch.int64)
        transform['head']=fit_projection(cache['head'][head_indices],16,SEED+1)
        transform['road']=fit_projection(cache['road'][indices],32,SEED+2)
        parts.extend([apply_projection(cache['head'],transform['head']),
                      apply_projection(cache['road'],transform['road'])])
    x=torch.cat(parts,1)
    transform['mean']=x[indices].mean(0);transform['std']=x[indices].std(0).clamp_min(.05)
    if not torch.isfinite(transform['mean']).all() or not torch.isfinite(transform['std']).all():
        raise ValueError('Invalid training feature normalization')
    return transform


def transformed_features(cache,transform):
    parts=[cache['geometry']]
    if transform['mode']=='visual':
        parts.extend([apply_projection(cache['head'],transform['head']),
                      apply_projection(cache['road'],transform['road'])])
    return (torch.cat(parts,1)-transform['mean'])/transform['std']


class VisualLaneRelationModel:
    def __init__(self,weights):
        checkpoint=torch.load(weights,map_location='cpu',weights_only=True)
        if checkpoint.get('format')!=FORMAT:raise ValueError('Wrong visual relation format')
        self.mode=checkpoint['mode'];self.transform=checkpoint['transform'];self.config=checkpoint['config']
        self.threshold=checkpoint['threshold']
        if self.threshold is not None and (not math.isfinite(self.threshold) or not 0<self.threshold<1):
            raise ValueError('Invalid calibrated threshold')
        if self.mode not in ('geometry','visual') or self.transform['mode']!=self.mode:
            raise ValueError('Mode/normalization mismatch')
        expected=45 if self.mode=='geometry' else 93
        if self.transform['mean'].shape!=(expected,) or self.transform['std'].shape!=(expected,):
            raise ValueError('Wrong relation transform dimensions')
        if (not torch.isfinite(self.transform['mean']).all() or not torch.isfinite(self.transform['std']).all()
                or not (self.transform['std']>0).all()):raise ValueError('Nonfinite relation normalization')
        for name in ('road','head'):
            if digest(self.config[name+'_parent'])!=self.config[name+'_sha256']:
                raise ValueError('Frozen parent changed')
        self.head=make_head(expected).eval().requires_grad_(False)
        self.head.load_state_dict(checkpoint['model'])
        self.encoder=FrozenLaneVisualEncoder(self.config['road_parent'],self.config['head_parent']) if self.mode=='visual' else None
        self.checkpoint=checkpoint

    def predict_frame(self,image,queries,boxes):
        if not queries or not boxes:return []
        encoded=self.encoder.encode_frame(image,queries,boxes) if self.encoder else None
        features=dict(head=[],road=[],geometry=[])
        keys=[]
        for li,query in enumerate(queries):
            for hi,box in enumerate(boxes):
                if encoded:
                    row=self.encoder.pair(encoded,li,hi,query,box,image.shape)
                    for k in features:features[k].append(row[k])
                else:features['geometry'].append(torch.from_numpy(pair_geometry(query,box,image.shape)))
                keys.append((li,hi))
        values={k:torch.stack(v) for k,v in features.items() if v}
        x=transformed_features(values,self.transform)
        with torch.inference_mode():probability=self.head(x).flatten().sigmoid().tolist()
        return [dict(lane_slot=li,head_slot=hi,lane_light_probability=p,
            relation_state='candidate_positive' if self.threshold is not None and p>=self.threshold else 'unknown',
            calibrated_positive_threshold=self.threshold,input_query_source='caller_supplied_unverified',
            physical_lane_association_verified=False,motion_authorized=False)
            for (li,hi),p in zip(keys,probability)]
