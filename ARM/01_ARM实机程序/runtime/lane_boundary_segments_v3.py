"""Trim a short conflicting end from a stable, observed typed subsegment.

This never classifies the discarded end or the entire parent boundary. Both
long competing types, branches, transverse yellow intersections and missing
geometry remain unresolved. No new pixels, extrapolation or control actions.
"""
import copy

import numpy as np

import lane_boundary_instances as base
import lane_boundary_instances_v2 as v2

MIN_PARENT_SUPPORT_FRACTION = .65


def row_type_counts(mask, points, policy):
    counts=[]
    for xf,yf in points:
        y=int(yf); choices=[]
        for a,b in base.row_runs(mask[y],policy.row_merge_gap):
            pixels=np.flatnonzero(mask[y,a:b+1])+a
            if len(pixels) and abs(float(np.median(pixels))-xf)<1e-5:
                choices.append([int((mask[y,a:b+1]==c).sum()) for c in (1,2)])
        if len(choices)!=1:
            return None
        counts.append(choices[0])
    return np.asarray(counts,dtype=float)


def split_unknown_end(parent,mask,policy):
    """Accept exactly one long consistent end segment under the old type rules."""
    if (parent['line_type']!='unknown' or parent['reason']!='conflicting_type_segments'
            or parent['ambiguous'] or parent['yellow_transverse_components']):
        return None
    points=np.asarray(parent['points_work'],dtype=float)
    counts=row_type_counts(mask,points,policy)
    if counts is None:return None
    fractions=counts/np.maximum(1.,counts.sum(1,keepdims=True))
    labels=fractions.argmax(1)+1
    labels[fractions.max(1)<policy.min_type_fraction]=0
    runs=[];start=0
    for index in range(1,len(points)+1):
        if index<len(points) and labels[index]==labels[start] and points[index,1]-points[index-1,1]==1:
            continue
        span=points[index-1,1]-points[start,1]+1
        if labels[start]>0 and index-start>=policy.min_rows and span>=policy.min_span:
            runs.append((start,index,int(labels[start])))
        start=index
    # A real long solid/dashed transition stays unknown as a parent boundary.
    if len(runs)!=1:return None
    a,b,label=runs[0]
    if (a!=0 and b!=len(points)) or (b-a)/len(points)<MIN_PARENT_SUPPORT_FRACTION:
        return None
    chosen=points[a:b];span=chosen[-1,1]-chosen[0,1]+1
    if len(chosen)/span<.70:return None
    coefficients=np.polyfit(chosen[:,1],chosen[:,0],2 if span>=40 else 1)
    error=float(np.sqrt(np.mean((np.polyval(coefficients,chosen[:,1])-chosen[:,0])**2)))
    slopes=np.polyval(np.polyder(coefficients),chosen[:,1])
    if error>policy.max_fit_error or np.abs(slopes).max()>policy.max_dx_dy:return None
    shares=fractions[a:b].mean(0)
    significant=(fractions[a:b]>=policy.conflict_row_fraction).sum(0)
    if shares.max()<policy.min_type_fraction or (significant>=policy.min_conflict_rows).all():return None
    child=copy.deepcopy(parent)
    child.update(line_type='solid' if label==1 else 'dashed',state_valid=True,
        reason='consistent_observed_subsegment_after_unknown_end',type_scope='observed_subsegment_only',
        x_of_y=coefficients.tolist(),y_range=chosen[[0,-1],1].astype(int).tolist(),supported_rows=len(chosen),
        points_work=chosen.tolist(),points_native=parent['points_native'][a:b],fit_rms_work=error,
        mean_type_fraction=shares.tolist(),significant_type_rows=significant.tolist())
    unknown=[]
    for begin,end in ((0,a),(b,len(points))):
        if begin==end:continue
        unknown.append(dict(parent_id=parent['id'],line_type='unknown',state_valid=False,
            reason='conflicting_end_not_classified',points_work=points[begin:end].tolist(),
            points_native=parent['points_native'][begin:end],y_range=points[[begin,end-1],1].astype(int).tolist()))
    audit=dict(parent_id=parent['id'],original_parent=copy.deepcopy(parent),
               retained_rows=len(chosen),original_rows=len(points),retained_row_fraction=len(chosen)/len(points),
               accepted_range=[a,b],accepted_type=child['line_type'],unknown_ends=unknown,
               original_type_threshold=policy.min_type_fraction)
    return child,audit


def extract_instances(mask,image,occluders,policy=None):
    policy=policy or base.InstancePolicy()
    result=v2.extract_instances(mask,image,occluders,policy)
    original=copy.deepcopy(result)
    instances=[];audits=[];unknown=[]
    for parent in result['instances']:
        split=split_unknown_end(parent,mask,policy)
        if split is None:
            instances.append(parent);continue
        child,audit=split;instances.append(child);audits.append(audit);unknown.extend(audit['unknown_ends'])
    if audits:
        shape=tuple(result['mask_size'][::-1])
        representatives,groups,supports=v2.group_solid_markings(instances,image,shape,policy.camera_center_fraction)
        result.update(instances=instances,pair=v2.propose_inner_pair(representatives,shape,policy.camera_center_fraction),
                      marking_groups=groups,paint_support=supports,representative_path_ids=[p['id'] for p in representatives])
        pair_key=lambda r:(r['pair']['state'],r['pair']['left_id'],r['pair']['right_id'])
        if pair_key(result)!=pair_key(original):
            original.update(blocked_type_splits=audits,blocked_type_split_reason='preserve_original_geometric_pair_identity')
            result=original;audits=[];unknown=[]
    result.update(type_segment_splits=audits,unresolved_type_ends=unknown,
                  algorithm_version='v3_consistent_observed_type_subsegment',
                  type_scope='Every accepted style applies only to listed current-frame points')
    return result
