"""Current-pixel round emitters plus measured housings, before CNN validation.

Quality is geometric evidence, not a calibrated neural detector probability.
No labels, per-video coordinates, previous colors or previous boxes are used.
"""
import cv2,numpy as np
from current_lamp_housing import refine
from signal_s3_policy import round_emitter
from light_pipeline import merge_lights


def round_component_stats(stats,width):
    data=stats[1:]
    if not len(data):return data
    w,h,area=data[:,2],data[:,3],data[:,4];ratio=w/np.maximum(1,h)
    return data[(area>=12)&(np.minimum(w,h)>=6)&(np.maximum(w,h)<=.08*width)&(ratio>=.65)&(ratio<=1.55)]


def propose_reference(image):
    top=image[:round(image.shape[0]*.6)];hsv=cv2.cvtColor(top,cv2.COLOR_BGR2HSV);h,s,v=cv2.split(hsv)
    base=(s>=80)&(v>=160);masks=[base&((h<=12)|(h>=165)),base&(h>=15)&(h<=34),base&(h>=35)&(h<=95)]
    candidates=[]
    for cls,mask in enumerate(masks):
        n,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
        for x,y,w,height,area in stats[1:]:
            x,y,w,height,area=map(int,(x,y,w,height,area))
            if area<12 or min(w,height)<6 or max(w,height)>.08*image.shape[1] or not .65<=w/height<=1.55:continue
            candidates.append((cls,[max(0,x-1),max(0,y-1),min(top.shape[1],x+w+1),min(top.shape[0],y+height+1)]))
    if len(candidates)>128:return [],dict(overloaded=True,component_candidates=len(candidates),source='current_emitter_geometry')
    rows=[]
    for cls,box in candidates:
        name=['red','yellow','green'][cls];e=round_emitter(image,box,name)
        if not e['valid']:continue
        a,b,c,d=box;px=max(2.,.3*(c-a));py=max(2.,.3*(d-b))
        context_box=[max(0,a-px),max(0,b-py),min(top.shape[1],c+px),min(top.shape[0],d+py)]
        seed=dict(class_id=cls,name=name,confidence=min(1.,e['circularity'],e['disk_fill']),box=context_box,emitter_seed_box=box,proposal_source='current_emitter_geometry',confidence_is_geometric_quality_not_probability=True)
        row=refine(image,seed)
        if not row.get('housing_refinement'):continue
        rows.append(row)
    return merge_lights(rows,.5,30),dict(overloaded=False,component_candidates=len(candidates),source='current_emitter_geometry_and_stable_dark_housing')


def propose_global_reference(image):
    top=image[:round(image.shape[0]*.6)];hsv=cv2.cvtColor(top,cv2.COLOR_BGR2HSV);h,s,v=cv2.split(hsv)
    base=(s>=80)&(v>=160);masks=[base&((h<=12)|(h>=165)),base&(h>=15)&(h<=34),base&(h>=35)&(h<=95)];candidates=[]
    for cls,mask in enumerate(masks):
        n,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
        for x,y,w,height,area in stats[1:]:
            x,y,w,height,area=map(int,(x,y,w,height,area))
            if area<12 or min(w,height)<6 or max(w,height)>.08*image.shape[1] or not .65<=w/height<=1.55:continue
            candidates.append((cls,[max(0,x-1),max(0,y-1),min(top.shape[1],x+w+1),min(top.shape[0],y+height+1)]))
    if len(candidates)>128:return [],dict(overloaded=True,component_candidates=len(candidates),source='global_current_housing_edges')
    emitters=[]
    for cls,box in candidates:
        name=['red','yellow','green'][cls];e=round_emitter(image,box,name)
        if e['valid']:emitters.append(dict(class_id=cls,name=name,seed=box,e=e,matches=[]))
    if not emitters:return [],dict(overloaded=False,component_candidates=len(candidates),source='global_current_housing_edges')
    gray=cv2.cvtColor(top,cv2.COLOR_BGR2GRAY);dark=cv2.integral((v<=140).astype(np.uint8));centers=np.asarray([e['e']['center'] for e in emitters]);diameters=np.asarray([2*e['e']['radius'] for e in emitters]);contour_count=0
    for low in (30,50,70):
        edges=cv2.Canny(gray,low,3*low);contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);contour_count+=len(contours)
        if len(contours)>8192:return [],dict(overloaded=True,component_candidates=len(candidates),source='global_current_housing_edges')
        for contour in contours:
            x,y,w,height=cv2.boundingRect(contour)
            if x<=0 or y<=0 or x+w>=top.shape[1] or y+height>=top.shape[0] or not 1.3<=w/height<=4.5:continue
            possible=np.flatnonzero((centers[:,0]>x)&(centers[:,0]<x+w)&(centers[:,1]>y)&(centers[:,1]<y+height)&(height>=.8*diameters)&(height<=3*diameters))
            if not len(possible):continue
            if cv2.contourArea(contour)<.6*w*height:continue
            dark_area=int(dark[y+height,x+w]-dark[y,x+w]-dark[y+height,x]+dark[y,x])
            if dark_area<.35*w*height:continue
            for i in possible:
                e=emitters[int(i)]
                if cv2.pointPolygonTest(contour,tuple(e['e']['center']),False)<0:continue
                box=[x-1,y-1,x+w+1,y+height+1];check=round_emitter(image,box,e['name'])
                if not check['valid'] or np.linalg.norm(np.asarray(check['center'])-e['e']['center'])>2.:continue
                e['matches'].append(dict(box=box,threshold=low))
    from arm_scene_a04 import box_iou
    rows=[]
    for e in emitters:
        matches=e['matches'];stable=[q for q in matches if len({r['threshold'] for r in matches if box_iou(q['box'],r['box'])>=.8})==3]
        if not stable or any(box_iou(stable[0]['box'],r['box'])<.6 for r in stable):continue
        box=max((r['box'] for r in stable),key=lambda b:(b[2]-b[0])*(b[3]-b[1]))
        rows.append(dict(class_id=e['class_id'],name=e['name'],confidence=min(1.,e['e']['circularity'],e['e']['disk_fill']),box=box,emitter_seed_box=e['seed'],
            proposal_source='current_emitter_and_global_closed_housing_edges',confidence_is_geometric_quality_not_probability=True,
            housing_refinement=dict(source='current_closed_housing_contours_at_three_global_edge_thresholds',thresholds=[30,50,70],original_emitter=e['e'],measurement_only=True,confidence_not_increased=True)))
    return merge_lights(rows,.5,30),dict(overloaded=False,component_candidates=len(candidates),source='current_emitter_and_global_closed_housing_edges',contours=contour_count)


def current_emitter_candidates(image):
    """Cheap coarse search, followed by original-resolution component checks.

    Subsampling proposes search regions only. All returned seeds are measured
    again from unchanged current pixels; they still need housing and both CNNs.
    """
    top=image[:round(image.shape[0]*.6)];step=2 if image.shape[1]>=960 else 1
    sample=top[::step,::step];h,s,v=cv2.split(cv2.cvtColor(sample,cv2.COLOR_BGR2HSV))
    base=(s>=80)&(v>=160)
    masks=[base&((h<=12)|(h>=165)),base&(h>=15)&(h<=34),base&(h>=35)&(h<=95)]
    candidates=[];seen=set();coarse_count=0
    for cls,mask in enumerate(masks):
        _,_,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
        if step==1:
            groups=[(0,0,round_component_stats(stats,image.shape[1]))]
        else:
            data=stats[1:];w,h,area=data[:,2],data[:,3],data[:,4]
            data=data[(area>=2)&(np.minimum(w,h)>=2)&(np.maximum(w,h)<=.04*image.shape[1]+2)&(w>=.4*h)&(w<=2.5*h)]
            coarse_count+=len(data)
            if coarse_count>256:return [],True
            groups=[]
            for x,y,w,height,_ in data:
                left,up=max(0,int(2*x-4)),max(0,int(2*y-4))
                right,down=min(top.shape[1],int(2*(x+w)+4)),min(top.shape[0],int(2*(y+height)+4))
                lh,ls,lv=cv2.split(cv2.cvtColor(top[up:down,left:right],cv2.COLOR_BGR2HSV))
                hue=[(lh<=12)|(lh>=165),(lh>=15)&(lh<=34),(lh>=35)&(lh<=95)][cls]
                _,_,local,_=cv2.connectedComponentsWithStats((hue&(ls>=80)&(lv>=160)).astype(np.uint8),8)
                groups.append((left,up,round_component_stats(local,image.shape[1])))
        for left,up,local in groups:
            for x,y,w,height,area in local:
                x,y,w,height=map(int,(x+left,y+up,w,height));key=(cls,x,y,w,height)
                if key in seen:continue
                seen.add(key)
                candidates.append((cls,[max(0,x-1),max(0,y-1),min(top.shape[1],x+w+1),min(top.shape[0],y+height+1)]))
                if len(candidates)>128:return [],True
    return candidates,False


def propose(image,executor=None):
    from current_lamp_housing import edge_housing
    top=image[:round(image.shape[0]*.6)];candidates,overloaded=current_emitter_candidates(image)
    if overloaded:return [],dict(overloaded=True,component_candidates=129,source='current_local_housing_edges')
    def measure(candidate):
        cls,box=candidate
        name=['red','yellow','green'][cls];e=round_emitter(image,box,name)
        if not e['valid']:return None
        a,b,c,d=box;px=max(2.,.3*(c-a));py=max(2.,.3*(d-b));context=[max(0,a-px),max(0,b-py),min(top.shape[1],c+px),min(top.shape[0],d+py)]
        a,b,c,d=context;w,h=c-a,d-b;region=(max(0,int(a-4*w)),max(0,int(b-h)),min(top.shape[1],int(c+4*w+1)),min(top.shape[0],int(d+h+1)))
        seed=dict(class_id=cls,name=name,confidence=min(1.,e['circularity'],e['disk_fill']),box=context,emitter_seed_box=box,proposal_source='current_emitter_geometry',confidence_is_geometric_quality_not_probability=True)
        # This only proposes a measured crop. The color CNN, type CNN, current
        # filled emitter, frontal group and three fresh green images still gate GO.
        row=edge_housing(image,seed,e,region,thresholds=(50,))
        return row if row.get('housing_refinement') else None
    measured=map(measure,candidates) if executor is None else executor.map(measure,candidates)
    rows=[r for r in measured if r is not None]
    return merge_lights(rows,.5,30),dict(overloaded=False,component_candidates=len(candidates),source='current_emitter_and_closed_housing_edges')
