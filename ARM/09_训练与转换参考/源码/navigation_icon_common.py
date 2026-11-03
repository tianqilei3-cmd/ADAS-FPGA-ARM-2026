"""NumPy/OpenCV-only icon preprocessing, matching frozen first-CNN semantics."""
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
NAMES=['left','straight','right','other']
SIZE=96


def canonical_icon(image):
    if image is None or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
        raise ValueError('Expected uint8 BGR icon crop')
    h,w=image.shape[:2]
    if min(h,w)<5:return None,'icon_too_small'
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    lo,hi=np.percentile(gray,[2,98])
    if hi-lo<20:return None,'insufficient_contrast'
    gray=np.clip((gray.astype(np.float32)-lo)*255/(hi-lo),0,255).astype(np.uint8)
    border=np.concatenate([gray[0],gray[-1],gray[:,0],gray[:,-1]])
    if np.median(border)<128:gray=255-gray
    _,ink=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    n,labels,stats,_=cv2.connectedComponentsWithStats(ink,8)
    if n<=1:return None,'no_foreground'
    index=1+int(np.argmax(stats[1:,4]))
    x,y,cw,ch,area=map(int,stats[index])
    if area<8 or min(cw,ch)<2:return None,'insufficient_foreground'
    selected=gray[y:y+ch,x:x+cw].copy()
    selected[labels[y:y+ch,x:x+cw]!=index]=255
    scale=76/max(cw,ch)
    rw,rh=max(1,round(cw*scale)),max(1,round(ch*scale))
    resized=cv2.resize(selected,(rw,rh),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
    canvas=np.full((SIZE,SIZE),255,np.uint8)
    a,b=(SIZE-rw)//2,(SIZE-rh)//2
    canvas[b:b+rh,a:a+rw]=resized
    return cv2.cvtColor(canvas,cv2.COLOR_GRAY2BGR),'ok'


def classify_probabilities(probabilities,*,threshold=.80,margin=.25):
    p=np.asarray(probabilities,dtype=float)
    if p.shape!=(4,) or not np.isfinite(p).all() or (p<0).any() or not np.isclose(p.sum(),1,atol=1e-4):
        raise ValueError('Expected normalized four-class probabilities')
    order=np.argsort(-p);best,second=map(int,order[:2])
    accepted=best!=3 and p[best]>=threshold and p[best]-p[second]>=margin
    return dict(direction=NAMES[best] if accepted else 'unknown',direction_valid=bool(accepted),
                raw_class=NAMES[best],confidence=float(p[best]),margin=float(p[best]-p[second]),
                probabilities=p.tolist(),immediate_turn_authorized=False,
                scope='Provided primary navigation icon only; route intent, not current maneuver permission.')
