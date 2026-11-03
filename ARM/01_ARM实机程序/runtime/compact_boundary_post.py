"""Bounded current-mask geometry for the board prototype.

Types come only from the trained solid/dashed channels. Observed gaps never
create a dashed label. Branches, mixed styles, transverse paint and rejected
nearer semantic components remain unknown.
"""
import cv2,numpy as np
from lane_boundary_wide_recovery import prepare_evidence
from lane_boundary_instances import transverse_yellow
from boundary_policy import requested_boundaries


def current_masks(probabilities,source_shape,objects):
    """Same typed-pixel/CC acceptance as filter_regions, without unused contours."""
    p=np.asarray(probabilities)
    if p.ndim!=3 or p.shape[0]!=3 or not np.isfinite(p).all():
        raise ValueError('Expected finite three-class probabilities')
    epsilon=4*np.finfo(np.float32).eps
    if p.min() < -epsilon or p.max()>1+epsilon or not np.allclose(p.sum(0),1.,atol=.001):
        raise ValueError('Expected normalized probabilities')
    p=np.clip(p,0,1)
    # Explicit comparisons preserve argmax's lower-index tie behavior.
    one=(p[1]>p[0])&(p[1]>=p[2]);two=(p[2]>p[0])&(p[2]>p[1])
    classes=np.zeros(p.shape[1:],np.uint8)
    classes[one&(p[1]>=.5)]=1;classes[two&(p[2]>=.5)]=2
    return filter_classes(classes,source_shape,objects)


def filter_classes(classes,source_shape,objects):
    if classes.dtype!=np.uint8 or classes.ndim!=2 or classes.max()>2:
        raise ValueError('Expected current three-class label image')
    classes=classes.copy()
    h,w=classes.shape;sh,sw=source_shape[:2]
    classes[:round(h*.4)]=0;classes[round(h*.94):]=0
    for obj in objects:
        if obj.get('name') not in ('vehicle','person','two_wheeler') or obj.get('confidence',0)<.35:continue
        a,b,c,d=obj['box']
        if not np.isfinite([a,b,c,d]).all() or c<=a or d<=b:raise ValueError('Invalid current occluder')
        if c<=0 or d<=0 or a>=sw or b>=sh:continue
        a,b=max(0,int(a*w/sw)-1),max(0,int(b*h/sh)-1)
        c,d=min(w,round(c*w/sw)+2),min(h,round(d*h/sh)+2)
        classes[b:d,a:c]=0
    accepted=np.zeros_like(classes)
    for label in (1,2):
        count,components,stats,_=cv2.connectedComponentsWithStats((classes==label).astype(np.uint8),8)
        heights=stats[:,cv2.CC_STAT_HEIGHT];areas=stats[:,cv2.CC_STAT_AREA]
        keep=(areas>=20)&(heights>=12)&(areas<=24*heights);keep[0]=False
        accepted[keep[components]]=label
    return accepted,classes

def from_mask(filtered,classes,image,frame):
    h,w=filtered.shape;sh,sw=image.shape[:2]
    count,labels,stats,_=cv2.connectedComponentsWithStats((filtered>0).astype(np.uint8),8)
    paths=[];transverse=transverse_yellow(cv2.resize(image,(w,h)))
    if count>65:
        extraction=dict(instances=[],mask_size=[w,h],overloaded=True)
        return extraction,requested_boundaries(extraction,frame,image.shape)
    for index in range(1,count):
        a,b,c,d,area=map(int,stats[index])
        if area<20 or d<18:continue
        ys,xs=np.nonzero(labels[b:b+d,a:a+c]==index);ys=ys+b;xs=xs+a
        totals=np.bincount(ys,minlength=h);observed=np.flatnonzero(totals)
        if len(observed)<14:continue
        sum_x=np.bincount(ys,weights=xs,minlength=h)
        centers=sum_x[observed]/totals[observed]
        lo=np.full(h,w,np.int32);hi=np.zeros(h,np.int32)
        np.minimum.at(lo,ys,xs);np.maximum.at(hi,ys,xs)
        widths=hi[observed]-lo[observed]+1
        yy=observed.astype(float);xx=centers.astype(float)
        dy=yy-yy.mean();den=float(dy@dy)
        if den<=0:continue
        slope=float(dy@(xx-xx.mean())/den);offset=float(xx.mean()-slope*yy.mean())
        rms=float(np.sqrt(np.mean((xx-(slope*yy+offset))**2)))
        style_counts=np.bincount(filtered[ys,xs],minlength=3)[1:3]
        label=int(style_counts.argmax())+1
        purity=float(style_counts.max()/max(1,style_counts.sum()))
        wide=bool((widths>28).any())
        blocked=False
        for x,y,ww,hh in transverse:
            near=(yy>=y-1)&(yy<=y+hh+1)&(xx>=x-2)&(xx<=x+ww+2)
            if near.any():blocked=True;break
        valid=purity>=.85 and not wide and abs(slope)<=2.25 and rms<=3 and not blocked
        reason=('typed_current_component' if valid else 'mixed_style_component' if purity<.85 else
                'branch_or_wide_component' if wide else 'transverse_yellow_paint' if blocked else 'unstable_component_geometry')
        paths.append(dict(id=len(paths),line_type=('solid' if label==1 else 'dashed') if valid else 'unknown',
            state_valid=valid,ambiguous=not valid,reason=reason,x_of_y=[slope,offset],
            y_range=[int(observed[0]),int(observed[-1])],supported_rows=len(observed),
            points_work=np.stack([xx,yy],1).tolist(),points_native=np.stack([xx*sw/w,yy*sh/h],1).tolist(),
            fit_rms_work=rms,mean_type_fraction=(style_counts/max(1,style_counts.sum())).tolist(),
            ego_lane_association_verified=False))
    extraction=dict(instances=paths,mask_size=[w,h],algorithm_version='bounded_component_geometry_prototype')
    boundaries=requested_boundaries(extraction,frame,image.shape)
    row=round(.82*h);center=.5*w
    unsupported=(classes[max(0,row-2):min(h,row+3)]>0)&(filtered[max(0,row-2):min(h,row+3)]==0)
    xs=np.flatnonzero(unsupported.sum(0)>=3)
    for side in ('left','right'):
        selected=boundaries[side]
        if not selected.get('state_valid'):continue
        native=selected['x_at_reference_native']*w/sw
        competing=xs[(xs<center-2)&(xs>native+2)] if side=='left' else xs[(xs>center+2)&(xs<native-2)]
        if len(competing):
            selected.update(line_type='unknown',state_valid=False,simulation_association_valid=False,
                reason='nearer_untyped_semantic_component')
    return extraction,boundaries

def process(probabilities,image,objects,frame):
    filtered,classes=current_masks(probabilities,image.shape,objects)
    extraction,boundaries=from_mask(filtered,classes,image,frame)
    if boundaries is None:boundaries=requested_boundaries(extraction,frame,image.shape)
    return extraction,boundaries
