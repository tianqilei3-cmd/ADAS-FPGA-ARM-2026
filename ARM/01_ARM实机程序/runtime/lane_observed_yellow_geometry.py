"""Geometry-only yellow pair proposals near existing solid semantic support.

This follows an unsuccessful strict0.85 paint-point experiment. Evidence within
3work pixels can guide observed paint geometry, but CANNOT confirm line type.
Every added path remains unknown, invalid for control; all V4 outputs persist.
"""
import copy
from dataclasses import asdict,dataclass

import cv2
import numpy as np

import lane_boundary_instances as base
from lane_boundary_instances_v2 import trace_without_stale_fragments
import lane_boundary_wide_recovery as v4
from lane_observed_yellow_pairs import YellowPairPolicy,_pair


@dataclass(frozen=True)
class GeometryPolicy:
    semantic_tolerance: int=3
    minimum_yellow_chroma: int=20
    minimum_paint_contrast: int=20
    minimum_rows: int=10
    minimum_span: int=12
    minimum_density: float=.85
    maximum_slope: float=4.5
    maximum_fit_error: float=1.5
    maximum_paint_width: int=12
    minimum_common_rows: int=10
    minimum_separation: float=2.
    # Same close-pair limit as V4 group_solid_markings, at512 work pixels.
    # Region average-width24 is a different quantity and is not a pair limit.
    maximum_separation: float=.065*512
    maximum_angle_difference_degrees: float=12.

    def __post_init__(self):
        if type(self.semantic_tolerance) is not int or not 0<=self.semantic_tolerance<=3:
            raise ValueError('Invalid semantic tolerance')
        # Reuse numerical validation only, not the strict probability assertion.
        for field in ['minimum_yellow_chroma','minimum_paint_contrast']:
            if type(getattr(self,field)) is not int or getattr(self,field)<=0:
                raise ValueError('Invalid paint contrast')
        YellowPairPolicy(**{k:v for k,v in asdict(self).items()
                           if k not in ['semantic_tolerance','minimum_yellow_chroma','minimum_paint_contrast']})


def overlaps(candidate,existing):
    points=np.asarray(candidate['points_work'])
    for old in existing:
        shared=points[(points[:,1]>=old['y_range'][0])&(points[:,1]<=old['y_range'][1])]
        if len(shared)<max(5,.75*len(points)):
            continue
        distance=np.abs(shared[:,0]-np.polyval(old['x_of_y'],shared[:,1]))
        if np.mean(distance<=3)>=.75:
            return True
    return False


def augment(before,classes,strong_solid,image,occluders,policy=None):
    p=policy or GeometryPolicy()
    classes=np.asarray(classes)
    if classes.dtype!=np.uint8 or classes.ndim!=2 or not np.isin(classes,[0,1,2]).all():
        raise ValueError('Invalid semantic classes')
    strong_solid=np.asarray(strong_solid)
    if strong_solid.shape!=classes.shape or not np.isin(strong_solid,[0,1]).all():
        raise ValueError('Invalid strong-solid evidence')
    h,w=classes.shape
    sh,sw=image.shape[:2]
    solid=(classes==1).astype(np.uint8)
    radius=p.semantic_tolerance
    near=cv2.dilate(solid,np.ones((2*radius+1,2*radius+1),np.uint8))
    near[:round(.4*h)]=0;near[round(.94*h):]=0
    for obj in occluders:
        if obj.get('name') not in ['vehicle','person','two_wheeler'] or obj.get('confidence',0)<.35:
            continue
        x1,y1,x2,y2=obj['box']
        if not np.isfinite([x1,y1,x2,y2]).all() or x2<=x1 or y2<=y1:
            raise ValueError('Invalid current occluder')
        if x2<=0 or y2<=0 or x1>=sw or y1>=sh:continue
        a,b=max(0,int(x1*w/sw)-1),max(0,int(y1*h/sh)-1)
        c,d=min(w,round(x2*w/sw)+2),min(h,round(y2*h/sh)+2)
        near[b:d,a:c]=0
    small=cv2.resize(image,(w,h))
    hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
    channels=small.astype(np.int16)
    # Windshield footage contains pale yellow paint whose S is only40--60.
    # Require both red and green above blue, plus local contrast. This is a
    # development color revision; it never grants a solid/dashed type.
    chroma=np.minimum(channels[:,:,1],channels[:,:,2])-channels[:,:,0]
    yellow=(hsv[:,:,0]>=15)&(hsv[:,:,0]<=40)&(hsv[:,:,2]>=105)&(chroma>=p.minimum_yellow_chroma)
    contrast=cv2.morphologyEx(hsv[:,:,2],cv2.MORPH_TOPHAT,np.ones((1,31),np.uint8))
    observed=(yellow&(contrast>=p.minimum_paint_contrast)&(near>0)).astype(np.uint8)
    count,components,stats,_=cv2.connectedComponentsWithStats(near,8)
    strict=base.InstancePolicy(row_merge_gap=0,max_gap_rows=1,max_run_width=p.maximum_paint_width,
        min_span=p.minimum_span,min_rows=p.minimum_rows,max_dx_dy=p.maximum_slope,max_fit_error=p.maximum_fit_error)
    after=copy.deepcopy(before)
    additions,audits=[],[]
    originals=before['instances']
    next_id=max((path['id'] for path in originals),default=-1)+1
    for component in range(1,count):
        if stats[component,3]<p.minimum_span:continue
        mask=((components==component)&(observed>0)).astype(np.uint8)
        if mask.sum()<2*p.minimum_rows:continue
        n,stripes,stripe_stats,_=cv2.connectedComponentsWithStats(mask,8)
        candidates=[]
        for stripe in range(1,n):
            if stripe_stats[stripe,3]<p.minimum_span:continue
            support=(stripes==stripe).astype(np.uint8)
            # A binary1 here supplies geometry nodes, not a semantic solid label.
            result=base.extract_instances(support,image,[],strict,trace_function=trace_without_stale_fragments)
            if len(result['instances'])!=1:continue
            candidate=result['instances'][0]
            span=candidate['y_range'][1]-candidate['y_range'][0]+1
            if candidate['ambiguous'] or candidate['supported_rows']/span<p.minimum_density:continue
            xy=np.round(candidate['points_work']).astype(int)
            if not np.all(observed[xy[:,1],xy[:,0]]):continue
            candidate.update(id=len(candidates),line_type='unknown',state_valid=False,
                reason='paired_observed_paint_near_semantic_region_type_unconfirmed',
                type_scope='unknown_geometry_only',mean_type_fraction=None,significant_type_rows=None,
                geometry_source='v5_geometry_only_observed_yellow_pair',
                semantic_support_scope=f'within_{radius}_work_pixels_not_exact_probability',
                source_solid85_fraction=float(np.mean(strong_solid[xy[:,1],xy[:,0]]>0)))
            candidates.append(candidate)
        audit=dict(component=component,observed_paint_pixels=int(mask.sum()),
            eligible_stripes=len(candidates),accepted=False,reason='requires_exactly_two_observed_stripes')
        if len(candidates)!=2:
            audits.append(audit);continue
        pair=_pair(*candidates,p)
        if pair is None:
            audit['reason']='not_supported_parallel_pair';audits.append(audit);continue
        added=[]
        for candidate in candidates:
            if overlaps(candidate,originals+additions):continue
            candidate=copy.deepcopy(candidate);candidate['id']=next_id;next_id+=1
            additions.append(candidate);added.append(candidate['id'])
        audit.update(accepted=bool(added),reason='geometry_only_pair' if added else 'already_covered',
                     pair=pair,added_path_ids=added)
        audits.append(audit)
    after['instances'].extend(additions)
    after.update(algorithm_version='v5_geometry_only_observed_yellow_pair_candidate',
        yellow_pair_added_ids=[p['id'] for p in additions],yellow_pair_audit=audits,
        yellow_pair_policy=asdict(p),runtime_pair_preserved=True)
    return after
