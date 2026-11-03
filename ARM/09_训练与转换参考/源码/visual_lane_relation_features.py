"""Frozen grayscale road/head features explicitly conditioned on a lane query.

Only pixels, normalized query points and current head boxes enter this module.
Source IDs, poses, labels, lane names and adjacency matrices are never inputs.
"""
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from light_roi_common import ROOT,crop_head,input_rgb,logits_of
from signal_approach_relevance import validate_image_box
from signal_emission_guard import EmissionGuardClassifier
from signal_lane_topology_data import pair_geometry

ROAD_PARENT=ROOT/'runs/adas/bdd100k_4class_stage42/weights/best.pt'
ROAD_SHA256='a9af921e423563bcd8c61246eed8bb991784cda091c0da761a57b8b282104ad1'
HEAD_PARENT=ROOT/'runs/adas/signal_emission_guard_20261017_172936_118725/weights/best.pt'
HEAD_SHA256='4c51451b57ac5177525d1b4a1053e440885f4bc6fa431349246ed370862c6a3b'
HEAD_DIM=1280
ROAD_DIM=1536
GEOMETRY_DIM=45
SIZE=512


def checked_query(query):
    q=np.asarray(query,dtype=np.float32)
    if q.shape!=(16,2) or not np.isfinite(q).all() or np.any(q<0) or np.any(q>1):
        raise ValueError('Finite normalized 16-point in-image query required')
    return q


def augment_frame(image,queries,boxes,variant):
    for q in queries:checked_query(q)
    for b in boxes:validate_image_box(image,b)
    if variant not in ('base','dark','bright','mirror'):
        raise ValueError('Unknown declared augmentation')
    q=[np.asarray(a,dtype=np.float32).copy() for a in queries]
    b=[list(map(float,a)) for a in boxes]
    if variant=='mirror':
        transformed=cv2.flip(image,1)
        for a in q:a[:,0]=1-a[:,0]
        w=image.shape[1]
        b=[[w-c,y,w-a,d] for a,y,c,d in b]
    elif variant in ('dark','bright'):
        transformed=np.clip(image.astype(np.float32)*(.8 if variant=='dark' else 1.2),0,255).astype(np.uint8)
    else:transformed=image.copy()
    return transformed,q,b


def grayscale_letterbox(image,size=SIZE):
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
        raise ValueError('uint8 BGR image required')
    h,w=image.shape[:2]
    if min(h,w)<2:raise ValueError('Positive scene dimensions required')
    gain=size/max(h,w);rw,rh=max(1,round(w*gain)),max(1,round(h*gain))
    left,top=(size-rw)//2,(size-rh)//2
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    resized=cv2.resize(gray,(rw,rh))
    canvas=np.full((size,size),114,np.uint8);canvas[top:top+rh,left:left+rw]=resized
    tensor=torch.from_numpy(np.repeat(canvas[None,:,:],3,axis=0).copy()).float().unsqueeze(0)/255
    transform=dict(scale_x=rw/size,scale_y=rh/size,offset_x=left/size,offset_y=top/size)
    return tensor,transform


def transformed_grid(points,transform):
    p=np.asarray(points,dtype=np.float32).copy()
    if not np.isfinite(p).all() or p.shape[-1]!=2:raise ValueError('Finite query grid required')
    p[...,0]=p[...,0]*transform['scale_x']+transform['offset_x']
    p[...,1]=p[...,1]*transform['scale_y']+transform['offset_y']
    return torch.from_numpy(2*p-1)


def roi_grid(box,image_shape,multiplier=1.,minimum_fraction=.0,count=4):
    h,w=image_shape[:2];a,b,c,d=map(float,box)
    cx,cy=(a+c)/2/w,(b+d)/2/h
    width=max((c-a)/w*multiplier,minimum_fraction);height=max((d-b)/h*multiplier,minimum_fraction)
    a,b=max(0,cx-width/2),max(0,cy-height/2)
    c,d=min(1,cx+width/2),min(1,cy+height/2)
    # Cell-center samples avoid putting every sample exactly on a box edge.
    fraction=(np.arange(count,dtype=np.float32)+.5)/count
    xx,yy=np.meshgrid(a+(c-a)*fraction,b+(d-b)*fraction)
    return np.stack([xx,yy],-1)


def lane_grid(query):
    q=checked_query(query)
    grid=np.repeat(q[:,None,:],3,axis=1)
    grid[:,:,0]+=np.asarray([-.025,0,.025],np.float32)[None,:]
    return np.clip(grid,0,1)


@torch.inference_mode()
def sample_features(maps,grids,transform):
    if not grids:return torch.empty((0,192,1,1))
    grid=torch.stack([transformed_grid(g,transform) for g in grids])
    values=[F.grid_sample(m.expand(len(grids),-1,-1,-1),grid,
                           mode='bilinear',padding_mode='border',align_corners=False)
            for m in maps]
    return torch.cat(values,1).clone()


def assemble_pair_features(scene,head_roi,context_roi,lane_bins,head_tight,query,box,shape):
    mean_lane=lane_bins.mean(0)
    road=torch.cat([head_roi,context_roi,scene,lane_bins.flatten(),
                    (mean_lane-context_roi).abs(),mean_lane*context_roi])
    if road.shape!=(ROAD_DIM,) or head_tight.shape!=(HEAD_DIM,):
        raise ValueError('Unexpected frozen feature dimensions')
    geometry=torch.from_numpy(pair_geometry(query,box,shape))
    if not torch.isfinite(road).all() or not torch.isfinite(head_tight).all():
        raise ValueError('Nonfinite visual feature')
    return dict(head=head_tight,road=road,geometry=geometry)


class FrozenLaneVisualEncoder:
    def __init__(self,road_parent=ROAD_PARENT,head_parent=HEAD_PARENT):
        from ultralytics import YOLO
        source=YOLO(str(road_parent)).model.model
        if any(layer.f!=-1 for layer in source[:7]):raise ValueError('Unexpected frozen road backbone')
        self.road=torch.nn.ModuleList(list(source[:7])).float().eval().requires_grad_(False)
        self.head=EmissionGuardClassifier(head_parent).model.eval().requires_grad_(False)
        self.head.model[-1].linear=torch.nn.Identity()

    @torch.inference_mode()
    def encode_frame(self,image,queries,boxes):
        for q in queries:checked_query(q)
        for b in boxes:validate_image_box(image,b)
        tensor,transform=grayscale_letterbox(image)
        maps=[];x=tensor
        for index,layer in enumerate(self.road):
            x=layer(x)
            if index in (4,6):maps.append(x)
        if [m.shape[1] for m in maps]!=[64,128]:raise ValueError('Unexpected road feature channels')
        scene=torch.cat([m.mean((2,3))[0] for m in maps]).clone()
        lane_values=sample_features(maps,[lane_grid(q) for q in queries],transform)
        lane_bins=torch.stack([lane_values[:,:,:5].mean((2,3)),
                               lane_values[:,:,5:11].mean((2,3)),
                               lane_values[:,:,11:].mean((2,3))],1) if queries else torch.empty((0,3,192))
        tight=sample_features(maps,[roi_grid(b,image.shape,minimum_fraction=.02) for b in boxes],transform).mean((2,3)) if boxes else torch.empty((0,192))
        context=sample_features(maps,[roi_grid(b,image.shape,multiplier=6,minimum_fraction=.08) for b in boxes],transform).mean((2,3)) if boxes else torch.empty((0,192))
        inputs=[]
        for box in boxes:
            gray=cv2.cvtColor(crop_head(image,box),cv2.COLOR_BGR2GRAY)
            inputs.append(input_rgb(cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR),128))
        head_vectors=[]
        for first in range(0,len(inputs),32):
            array=np.stack(inputs[first:first+32]).transpose(0,3,1,2).copy()
            values=logits_of(self.head(torch.from_numpy(array).float()/255))
            head_vectors.append(values.clone())
        head_vectors=torch.cat(head_vectors) if head_vectors else torch.empty((0,HEAD_DIM))
        return dict(scene=scene,lanes=lane_bins,head_roi=tight,head_context=context,head_tight=head_vectors)

    def pair(self,encoded,lane_slot,head_slot,query,box,shape):
        return assemble_pair_features(encoded['scene'],encoded['head_roi'][head_slot],
            encoded['head_context'][head_slot],encoded['lanes'][lane_slot],
            encoded['head_tight'][head_slot],query,box,shape)


def fit_projection(values,components,seed):
    """Called on training rows only; validation/review never enter fitting."""
    if values.ndim!=2 or len(values)<=components or not torch.isfinite(values).all():
        raise ValueError('Insufficient finite training feature rows')
    mean=values.mean(0);std=values.std(0).clamp_min(.05)
    standardized=(values-mean)/std
    center=standardized.mean(0)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        _,_,directions=torch.pca_lowrank(standardized,q=components,center=True,niter=3)
    return dict(mean=mean,std=std,center=center,components=directions.T.contiguous())


def apply_projection(values,projection):
    return (((values-projection['mean'])/projection['std'])-projection['center'])@projection['components'].T
