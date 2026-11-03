"""Reuse a SHA only for an array backed by genuinely immutable Python bytes.

Merely setting writeable=False is insufficient: another view, shared memory or
camera process can still change its backing storage. Those always get rehashed.
"""
from collections import OrderedDict
import hashlib,threading,weakref
import numpy as np

_cache=OrderedDict()
_lock=threading.Lock()


def pixel_digest(image):
    base=image
    while isinstance(base,np.ndarray) and base.base is not None:
        base=base.base
    immutable=isinstance(base,bytes) and image.flags.c_contiguous
    key=id(image);tag=(image.shape,image.strides,image.dtype.str)
    if immutable:
        with _lock:
            hit=_cache.get(key)
            if hit is not None and hit[0]() is image and hit[1]==tag:
                _cache.move_to_end(key);return hit[2]
    digest=hashlib.sha256(memoryview(np.ascontiguousarray(image)))
    digest.update(str(image.shape).encode('ascii'));result=digest.hexdigest()
    if immutable:
        with _lock:
            _cache[key]=(weakref.ref(image),tag,result);_cache.move_to_end(key)
            while len(_cache)>8:_cache.popitem(last=False)
    return result
