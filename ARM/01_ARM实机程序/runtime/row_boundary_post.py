"""Observed row paths using the existing V2/V3 geometry and type thresholds.

Skip diagnostic paint grouping and pair enumeration unused by the board's
nearest-side policy. No new mask pixels or style-from-gap classification.
"""
import cv2,numpy as np
import lane_boundary_instances as base
from lane_boundary_segments_v3 import split_unknown_end
from compact_boundary_post import current_masks,filter_classes
from boundary_policy import requested_boundaries


def trace(mask,blocked,p):
    tracks=[];complete=[]
    ys,xs=np.nonzero(mask);by_row={}
    if not len(xs):return []
    starts=np.flatnonzero(np.r_[True,(ys[1:]!=ys[:-1])|(xs[1:]-xs[:-1]>p.row_merge_gap+1)])
    stops=np.r_[starts[1:],len(xs)]
    prefix=np.r_[0,np.cumsum(mask[ys,xs]==1,dtype=np.int32)]
    for begin,end in zip(starts,stops):
        a,b=int(xs[begin]),int(xs[end-1])
        if b-a+1>p.max_run_width:continue
        n=int(end-begin);y=int(ys[begin]);solid=int(prefix[end]-prefix[begin])
        center=(int(xs[begin+(n-1)//2])+int(xs[begin+n//2]))/2
        by_row.setdefault(y,[]).append(dict(x=center,y=y,a=a,b=b,counts=[solid,n-solid]))
    blocked_any=bool(blocked.any())
    expanded=cv2.dilate(blocked.astype(np.uint8),np.ones((1,3),np.uint8)) if blocked_any else None
    for y,nodes in by_row.items():
        active=[]
        for track in tracks:
            if y-track['points'][-1][1]<=p.max_gap_rows+1:active.append(track)
            else:complete.append(track)
        tracks=active
        if len(nodes)>64 or len(complete)+len(tracks)>512:return []
        links=[]
        for ti,track in enumerate(tracks):
            x0,y0=track['points'][-1]
            if y-y0>1 and len(track['points'])<4:continue
            slope=0.
            if len(track['points'])>=3:
                xp,yp=track['points'][-3]
                slope=min(p.max_dx_dy,max(-p.max_dx_dy,(x0-xp)/(y0-yp)))
            for ni,node in enumerate(nodes):
                dy=y-y0
                if abs(node['x']-(x0+slope*dy))>3+p.max_dx_dy*dy:continue
                crossed=False
                if blocked_any:
                    for iy in range(y0+1,y+1):
                        ix=round(x0+(node['x']-x0)*(iy-y0)/dy)
                        if expanded[iy,ix]:crossed=True;break
                if not crossed:links.append((ti,ni))
        attached=set();retired=set()
        for ti,track in enumerate(tracks):
            choices=[n for t,n in links if t==ti]
            if len(choices)==1 and sum(n==choices[0] for _,n in links)==1:
                node=nodes[choices[0]];track['points'].append((node['x'],y))
                track['counts'].append(node['counts']);track['widths'].append(node['b']-node['a']+1);attached.add(choices[0])
            elif choices:
                track['ambiguous']=True;complete.append(track);retired.add(ti)
        tracks=[t for i,t in enumerate(tracks) if i not in retired]
        for ni,node in enumerate(nodes):
            if ni not in attached:
                tracks.append(dict(points=[(node['x'],y)],counts=[node['counts']],widths=[node['b']-node['a']+1],ambiguous=any(n==ni for _,n in links)))
    return complete+tracks


def from_mask(mask,image,objects,frame):
    policy=base.InstancePolicy()
    extracted=base.extract_instances(mask,image,objects,policy,trace_function=trace)
    paths=[]
    for parent in extracted['instances']:
        split=split_unknown_end(parent,mask,policy)
        paths.append(split[0] if split else parent)
    extracted.update(instances=paths,algorithm_version='bounded_existing_V3_row_paths_no_diagnostic_groups')
    return extracted,requested_boundaries(extracted,frame,image.shape)


def process(probabilities,image,objects,frame):
    mask,_=current_masks(probabilities,image.shape,objects)
    return from_mask(mask,image,objects,frame)


def process_classes(classes,image,objects,frame):
    mask,_=filter_classes(classes,image.shape,objects)
    return from_mask(mask,image,objects,frame)
