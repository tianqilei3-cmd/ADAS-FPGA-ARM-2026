"""Refine an emitter-only detection using a stable current dark housing.

This supplies a measured crop/search box, never a color, type or lane answer.
The original detector box/confidence are retained for audit; normal classifiers
and the full two-head/filled-emitter/group/traffic gates still run afterward.
"""
import cv2,numpy as np
from arm_scene_a04 import box_iou
from signal_s3_policy import round_emitter


def edge_housing(image,proposal,emitter,region,thresholds=(30,50,70)):
    x0,y0,x1,y1=region;crop=image[y0:y1,x0:x1];gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    value=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)[:,:,2];cx,cy=emitter['center'];a,b,c,d=proposal['box'];h=d-b;w=c-a;values=[];emitter_checks={}
    for low in thresholds:
        edges=cv2.Canny(gray,low,3*low)
        contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x,y,bw,bh=cv2.boundingRect(contour)
            if x<=0 or y<=0 or x+bw>=x1-x0 or y+bh>=y1-y0:continue
            if not 1.3<=bw/bh<=4.5 or not .6*h<=bh<=1.8*h or cv2.contourArea(contour)<.6*bw*bh:continue
            if cv2.pointPolygonTest(contour,(cx-x0,cy-y0),False)<0:continue
            if (value[y:y+bh,x:x+bw]<=140).mean()<.35:continue
            box=[x+x0-1,y+y0-1,x+x0+bw+1,y+y0+bh+1]
            key=tuple(box)
            if key not in emitter_checks:emitter_checks[key]=round_emitter(image,box,proposal['name'])
            check=emitter_checks[key]
            if not check['valid'] or np.linalg.norm(np.asarray(check['center'])-emitter['center'])>2.:continue
            values.append(dict(box=box,threshold=low))
    stable=[q for q in values if len({r['threshold'] for r in values if box_iou(q['box'],r['box'])>=.8})==len(thresholds)]
    if not stable or any(box_iou(stable[0]['box'],r['box'])<.6 for r in stable):return proposal
    box=max((r['box'] for r in stable),key=lambda b:(b[2]-b[0])*(b[3]-b[1]))
    if box[2]-box[0]<=1.1*w:return proposal
    return dict(proposal,detector_box=list(proposal['box']),box=box,housing_refinement=dict(source='current_closed_housing_contours',
        thresholds=list(thresholds),original_emitter=emitter,measurement_only=True,confidence_not_increased=True))


def refine(image,proposal):
    a,b,c,d=map(float,proposal['box']);w,h=c-a,d-b
    if min(w,h)<6 or not .8<=w/h<2.:return proposal
    emitter=round_emitter(image,proposal['box'],proposal['name'])
    if not emitter['valid']:return proposal
    height,width=image.shape[:2];x0,y0=max(0,int(a-4*w)),max(0,int(b-h));x1,y1=min(width,int(c+4*w+1)),min(height,int(d+h+1))
    if min(x1-x0,y1-y0)<8:return proposal
    hsv=cv2.cvtColor(image[y0:y1,x0:x1],cv2.COLOR_BGR2HSV);hue,sat,value=cv2.split(hsv);cx,cy=emitter['center'];candidates=[]
    allowed={'green':(hue>=35)&(hue<=95),'red':(hue<=12)|(hue>=165),'yellow':(hue>=15)&(hue<=34)}[proposal['name']]
    yy,xx=np.ogrid[y0:y1,x0:x1]
    emitting=allowed&(sat>=80)&(value>=160)&((xx-cx)**2+(yy-cy)**2<=(emitter['radius']+1.)**2)
    for threshold in range(40,141,5):
        dark=value<=threshold
        mask=cv2.morphologyEx((dark|emitting).astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        n,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
        for x,y,bw,bh,area in stats[1:]:
            x,y,bw,bh,area=map(int,(x,y,bw,bh,area));left,top=x+x0,y+y0
            if x<=0 or y<=0 or x+bw>=x1-x0 or y+bh>=y1-y0:continue
            if not 1.3<=bw/bh<=4.5 or not .6*h<=bh<=1.8*h or int(dark[y:y+bh,x:x+bw].sum())<.35*bw*bh:continue
            if not left<cx<left+bw or not top<cy<top+bh:continue
            if min(c,left+bw)-max(a,left)<.5*w or min(d,top+bh)-max(b,top)<.5*h:continue
            box=[max(0,left-1),max(0,top-1),min(width,left+bw+1),min(height,top+bh+1)]
            check=round_emitter(image,box,proposal['name'])
            if not check['valid'] or np.linalg.norm(np.asarray(check['center'])-emitter['center'])>2.:continue
            candidates.append(dict(box=box,threshold=threshold,dark_fill=area/(bw*bh)))
    stable=[q for q in candidates if len({r['threshold'] for r in candidates if box_iou(q['box'],r['box'])>=.8})>=3]
    if not stable:return edge_housing(image,proposal,emitter,(x0,y0,x1,y1))
    # All accepted thresholds must describe the same local object, rather than
    # selecting a different enclosing object at whichever threshold is helpful.
    if any(box_iou(stable[0]['box'],r['box'])<.8 for r in stable):return proposal
    box=np.median(np.asarray([q['box'] for q in stable]),axis=0).tolist()
    if box[2]-box[0]<=1.1*w:return proposal
    return dict(proposal,detector_box=list(proposal['box']),box=box,
        housing_refinement=dict(source='current_pixels_stable_dark_component_and_same_filled_emitter',thresholds=sorted({q['threshold'] for q in stable}),
            original_emitter=emitter,measurement_only=True,confidence_not_increased=True))
