"""Bounded V5 proposal: observed yellow pairs, including short/oblique paint.

All V4 paths, their types, and runtime pairing remain unchanged. Additions
require two independent thin yellow paint stripes in the same solid semantic
component. Type applies only to current observed points, never an entire lane.
"""
import copy
from dataclasses import asdict,dataclass
import math

import cv2
import numpy as np

import lane_boundary_instances as base
from lane_boundary_instances_v2 import trace_without_stale_fragments
import lane_boundary_wide_recovery as v4


@dataclass(frozen=True)
class YellowPairPolicy:
    minimum_probability: float=.85
    minimum_rows: int=10
    minimum_span: int=12
    minimum_density: float=.85
    maximum_slope: float=4.5
    maximum_fit_error: float=1.5
    maximum_paint_width: int=12
    minimum_common_rows: int=10
    minimum_separation: float=2.
    maximum_separation: float=24.
    maximum_angle_difference_degrees: float=12.

    def __post_init__(self):
        if (not .5<self.minimum_probability<=1 or not 0<self.minimum_density<=1
                or not 0<self.minimum_separation<self.maximum_separation
                or not 0<self.maximum_angle_difference_degrees<45):
            raise ValueError('Invalid confidence or pair limits')
        for field in ['minimum_rows','minimum_span','maximum_paint_width','minimum_common_rows']:
            if type(getattr(self,field)) is not int or getattr(self,field)<2:
                raise ValueError('Invalid observed point limit')
        if not all(math.isfinite(v) and v>0 for v in [self.maximum_slope,self.maximum_fit_error]):
            raise ValueError('Invalid geometry limit')


def _overlaps(candidate,existing):
    points=np.asarray(candidate['points_work'])
    for old in existing:
        shared=points[(points[:,1]>=old['y_range'][0])&(points[:,1]<=old['y_range'][1])]
        if len(shared)<5:
            continue
        distance=np.abs(shared[:,0]-np.polyval(old['x_of_y'],shared[:,1]))
        if np.mean(distance<=3)>=.75:
            return True
    return False


def _pair(a,b,policy):
    ap={int(y):float(x) for x,y in a['points_work']}
    bp={int(y):float(x) for x,y in b['points_work']}
    common=sorted(set(ap)&set(bp))
    if len(common)<policy.minimum_common_rows:
        return None
    difference=np.array([bp[y]-ap[y] for y in common])
    if not (np.all(difference>0) or np.all(difference<0)):
        return None
    separation=np.abs(difference)
    if separation.min()<policy.minimum_separation or separation.max()>policy.maximum_separation:
        return None
    if separation[-1]<.75*separation[0]:
        return None
    slopes=[np.polyval(np.polyder(path['x_of_y']),common) for path in [a,b]]
    angle=np.rad2deg(np.abs(np.arctan(slopes[0])-np.arctan(slopes[1])))
    if angle.max()>policy.maximum_angle_difference_degrees:
        return None
    return dict(common_rows=len(common),common_y=[common[0],common[-1]],
                separation_work=separation.tolist(),maximum_angle_difference_degrees=float(angle.max()))


def augment(before,classes,strong_solid,image,occluders,policy=None):
    p=policy or YellowPairPolicy()
    # The caller obtains strong_solid through V4's fixed0.85 probability gate.
    if p.minimum_probability!=v4.RecoveryPolicy().minimum_probability:
        raise ValueError('Strong mask threshold must match fixed V4 evidence')
    classes=np.asarray(classes)
    strong=np.asarray(strong_solid)
    if (classes.dtype!=np.uint8 or classes.ndim!=2 or not np.isin(classes,[0,1,2]).all()
            or strong.shape!=classes.shape or not np.isin(strong,[0,1]).all()
            or np.any((strong>0)&(classes!=1))):
        raise ValueError('Invalid class/strong-solid evidence')
    h,w=classes.shape
    sh,sw=image.shape[:2]
    eligible=(classes==1).astype(np.uint8)
    eligible[:round(.4*h)]=0
    eligible[round(.94*h):]=0
    blocked=np.zeros_like(eligible)
    for obj in occluders:
        if obj.get('name') not in ['vehicle','person','two_wheeler'] or obj.get('confidence',0)<.35:
            continue
        x1,y1,x2,y2=obj['box']
        if not np.isfinite([x1,y1,x2,y2]).all() or x2<=x1 or y2<=y1:
            raise ValueError('Invalid same-frame occluder')
        if x2<=0 or y2<=0 or x1>=sw or y1>=sh:
            continue
        a,b=max(0,int(x1*w/sw)-1),max(0,int(y1*h/sh)-1)
        c,d=min(w,round(x2*w/sw)+2),min(h,round(y2*h/sh)+2)
        blocked[b:d,a:c]=1
    eligible[blocked>0]=0
    small=cv2.resize(image,(w,h))
    hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
    yellow=(hsv[:,:,0]>=15)&(hsv[:,:,0]<=40)&(hsv[:,:,1]>=80)&(hsv[:,:,2]>=105)
    contrast_paint=v4._paint_evidence(image,classes.shape,v4.RecoveryPolicy())
    painted=(yellow&contrast_paint&(strong>0)&(eligible>0)).astype(np.uint8)
    count,components,stats,_=cv2.connectedComponentsWithStats(eligible,8)
    strict=base.InstancePolicy(row_merge_gap=0,max_gap_rows=1,max_run_width=p.maximum_paint_width,
        min_span=p.minimum_span,min_rows=p.minimum_rows,max_dx_dy=p.maximum_slope,
        max_fit_error=p.maximum_fit_error)
    after=copy.deepcopy(before)
    originals=copy.deepcopy(before['instances'])
    additions,audits=[],[]
    next_id=max((path['id'] for path in originals),default=-1)+1
    for component in range(1,count):
        x,y,cw,ch,area=map(int,stats[component])
        if area<20 or ch<p.minimum_span:
            continue
        support=((components==component)&(painted>0)).astype(np.uint8)
        if support.sum()<p.minimum_rows*2:
            continue
        n,stripes,stripe_stats,_=cv2.connectedComponentsWithStats(support,8)
        candidates=[]
        for stripe in range(1,n):
            if stripe_stats[stripe,3]<p.minimum_span:
                continue
            mask=(stripes==stripe).astype(np.uint8)
            extracted=base.extract_instances(mask,image,[],strict,trace_function=trace_without_stale_fragments)
            if len(extracted['instances'])!=1:
                continue
            path=extracted['instances'][0]
            span=path['y_range'][1]-path['y_range'][0]+1
            # Yellow aspect heuristics are not a source of style here: every
            # listed point must still have actual0.85 solid probability below.
            if path['ambiguous'] or path['supported_rows']/span<p.minimum_density:
                continue
            points=np.asarray(path['points_work']).round().astype(int)
            if not (strong[points[:,1],points[:,0]]>0).all():
                continue
            if not (painted[points[:,1],points[:,0]]>0).all():
                continue
            path.update(id=len(candidates),line_type='solid',state_valid=True,
                reason='two_actual_yellow_paint_stripes_with_strong_current_solid_support',
                type_scope='observed_points_only',geometry_source='v5_observed_yellow_pair',
                candidate_source='current_frame_only')
            candidates.append(path)
        audit=dict(component=component,box_work=[x,y,cw,ch],paint_pixels=int(support.sum()),
                   eligible_stripes=len(candidates),accepted=False,reason='requires_exactly_two_observed_stripes')
        if len(candidates)!=2:
            audits.append(audit)
            continue
        pair=_pair(*candidates,p)
        if pair is None:
            audit['reason']='not_a_supported_parallel_pair'
            audits.append(audit)
            continue
        added=[]
        for candidate in candidates:
            if _overlaps(candidate,originals+additions):
                continue
            candidate=copy.deepcopy(candidate)
            candidate['id']=next_id
            next_id+=1
            additions.append(candidate)
            added.append(candidate['id'])
        audit.update(accepted=bool(added),reason='paired_observed_yellow_stripes' if added else 'already_covered',
                     pair=pair,added_path_ids=added)
        audits.append(audit)
    after['instances'].extend(additions)
    after.update(algorithm_version='v5_observed_yellow_pair_candidate',
        yellow_pair_added_ids=[path['id'] for path in additions],yellow_pair_audit=audits,
        yellow_pair_policy=asdict(p),runtime_pair_preserved=True)
    return after


def extract_from_evidence(filtered,classes,strong_solid,image,occluders,policy=None):
    before=v4.extract_from_evidence(filtered,classes,strong_solid,image,occluders)
    return augment(before,classes,strong_solid,image,occluders,policy)
