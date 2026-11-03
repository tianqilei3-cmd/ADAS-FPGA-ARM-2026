"""Tiny learned navigation glyph classifier. Runtime needs only NumPy/OpenCV."""
from pathlib import Path
import cv2
import numpy as np

from navigation_icon_common import NAMES,canonical_icon,classify_probabilities


class ShapeHOG:
    """8x8 cells,2x2-cell L2-Hys blocks,9 unsigned orientation bins.

    Implemented here because this installed OpenCV5 build omits HOGDescriptor.
    Sobel gradients plus NumPy histograms keep the ARM runtime dependency small.
    """
    def compute(self,gray):
        if gray.shape!=(96,96) or gray.dtype!=np.uint8:
            raise ValueError('HOG expects canonical uint8 96x96 image')
        values=np.sqrt(gray.astype(np.float32)/255.)
        gx=cv2.Sobel(values,cv2.CV_32F,1,0,ksize=1)
        gy=cv2.Sobel(values,cv2.CV_32F,0,1,ksize=1)
        magnitude=np.hypot(gx,gy)
        angle=np.mod(np.rad2deg(np.arctan2(gy,gx)),180.)/20.
        low=np.floor(angle).astype(np.int32)%9
        fraction=angle-np.floor(angle)
        yy,xx=np.indices(gray.shape)
        cells=(yy//12)*8+xx//12
        hist=np.bincount((cells*9+low).ravel(),weights=(magnitude*(1-fraction)).ravel(),minlength=64*9)
        hist+=np.bincount((cells*9+(low+1)%9).ravel(),weights=(magnitude*fraction).ravel(),minlength=64*9)
        hist=hist.reshape(8,8,9)
        blocks=np.stack([hist[y:y+2,x:x+2].ravel() for y in range(7) for x in range(7)])
        blocks/=np.sqrt((blocks*blocks).sum(1,keepdims=True)+1e-6)
        blocks=np.minimum(blocks,.2)
        blocks/=np.sqrt((blocks*blocks).sum(1,keepdims=True)+1e-6)
        return blocks.astype(np.float32).ravel()


def descriptor():
    return ShapeHOG()


def feature_from_canonical(image,hog=None):
    return (hog or descriptor()).compute(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)).reshape(-1).astype(np.float32)


class NavigationHOGClassifier:
    def __init__(self,weights):
        with np.load(Path(weights),allow_pickle=False) as data:
            if data['names'].tolist()!=NAMES:raise ValueError('Unexpected navigation class order')
            self.weights=data['weights'].copy();self.bias=data['bias'].copy()
        self.hog=descriptor()
        if self.weights.shape!=(4,1764) or self.bias.shape!=(4,):
            raise ValueError('Unexpected HOG classifier shape')

    def predict(self,crops):
        rows=[]
        for crop in crops:
            normal,reason=canonical_icon(crop)
            if normal is None:
                rows.append(dict(direction='unknown',direction_valid=False,reason=reason,
                                 immediate_turn_authorized=False,probabilities=None))
                continue
            feature=feature_from_canonical(normal,self.hog)
            logits=self.weights@feature+self.bias
            p=np.exp(logits-logits.max());p/=p.sum()
            rows.append(dict(**classify_probabilities(p),reason='learned_hog_on_provided_icon'))
        return rows
