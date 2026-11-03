"""Small bounded salt-noise filter; invalid input cannot establish clearance."""
import cv2
import numpy as np


def prepare(image):
    if not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
        raise ValueError('Expected current uint8 BGR capture')
    filtered=cv2.medianBlur(image,3)
    sample=filtered[::8,::8].reshape(-1,3)
    lo,hi=np.percentile(sample,[10,90],axis=0)
    spread=float((hi-lo).max())
    # Isolated chroma errors must not make a uniform no-signal picture valid.
    diff=np.abs(image[::4,::4].astype(np.int16)-filtered[::4,::4].astype(np.int16))
    noise=float((diff.max(axis=2)>40).mean())
    valid=spread>=12. and noise<.30
    return filtered,dict(valid=valid,reason='spatial_content_present' if valid else 'severe_impulse_noise' if noise>=.30 else 'uniform_or_no_signal_input',
        preprocessing='median3',robust_channel_range=spread,
        salt_noise_fraction=noise,maximum_impulse_fraction=.30)


def invalidate_context(context,quality):
    if not quality['valid']:
        context=dict(context,valid_flags=context['valid_flags']&~(1|8),boundary=0,arm_action=1)
    return context


def inspect_native_patches(image):
    """Quality-only native median neighborhoods; model pixels stay untouched."""
    if not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
        raise ValueError('Expected current uint8 BGR capture')
    h,w=image.shape[:2]
    if min(h,w)<96:return prepare(image)[1]
    noise=[];colors=[]
    for yf in (.1,.5,.9):
        for xf in (.1,.3666667,.6333333,.9):
            left=max(0,min(w-64,round(xf*w)-32));top=max(0,min(h-64,round(yf*h)-32))
            patch=image[top:top+64,left:left+64];filtered=cv2.medianBlur(patch,3)
            # Interior centers have the exact same native3x3 neighborhood as a
            # full-image median, without filtering pixels the model never uses.
            a=patch[2:-2:4,2:-2:4].astype(np.int16);b=filtered[2:-2:4,2:-2:4].astype(np.int16)
            noise.append((np.abs(a-b).max(axis=2)>40).ravel());colors.append(filtered[2:-2:8,2:-2:8].reshape(-1,3))
    lo,hi=np.percentile(np.concatenate(colors),[10,90],axis=0);spread=float((hi-lo).max());fraction=float(np.concatenate(noise).mean())
    valid=spread>=12 and fraction<.30
    return dict(valid=valid,reason='spatial_content_present' if valid else 'severe_impulse_noise' if fraction>=.30 else 'uniform_or_no_signal_input',
        preprocessing='12_native_patches_median3_quality_only_model_pixels_unchanged',robust_channel_range=spread,salt_noise_fraction=fraction,maximum_impulse_fraction=.30,native_patch_count=12)
