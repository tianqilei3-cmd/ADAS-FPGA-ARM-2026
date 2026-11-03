"""Three-class stable softmax with the board OpenCV vectorized exponential."""
import cv2,numpy as np


def softmax3(logits):
    logits=np.asarray(logits,dtype=np.float32)
    if logits.ndim!=3 or logits.shape[0]!=3 or not np.isfinite(logits).all():
        raise ValueError('Expected finite three-class logits')
    maximum=np.maximum(np.maximum(logits[0],logits[1]),logits[2])
    shifted=np.ascontiguousarray(logits-maximum)
    p=cv2.exp(shifted.reshape(-1,logits.shape[2])).reshape(logits.shape)
    p/=p[0]+p[1]+p[2]
    return p
