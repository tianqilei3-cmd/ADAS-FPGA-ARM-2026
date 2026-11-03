"""Bounded image-token CPU geometry job overlapping the road NPU pass."""
import multiprocessing as mp
import queue,time
import numpy as np
from lamp_cpu_worker import CpuLampWorker


def geometry_worker(requests,replies,ready,pixels):
    try:
        import cv2
        from geometry_lamp_proposals import propose
        from arm_scene_a04 import frame_identity
        cv2.setNumThreads(1)
        ready.set()
        while True:
            request=requests.get()
            if request is None:break
            token,shape=request;image=np.frombuffer(pixels,dtype=np.uint8)[:int(np.prod(shape))].reshape(shape);image.setflags(write=False);tick=time.monotonic()
            if frame_identity(image,'worker',0,0,0)['image_identity']!=token[1]:raise RuntimeError('Shared geometry image does not match source identity')
            rows,meta=propose(image);meta.update(geometry_workers=1,queue_capacity=1,shared_image_identity_verified=True)
            replies.put((token,(rows,meta),(time.monotonic()-tick)*1000,None))
    except BaseException as exc:
        ready.set()
        try:replies.put_nowait((None,None,None,repr(exc)))
        except queue.Full:pass


class GeometryWorker(CpuLampWorker):
    def __init__(self):
        context=mp.get_context('spawn');self.requests=context.Queue(maxsize=1);self.replies=context.Queue(maxsize=1)
        self.ready=context.Event();self.pending=None;self.closed=False
        self.pixels=context.RawArray('B',1080*1920*3)
        self.process=context.Process(target=geometry_worker,args=(self.requests,self.replies,self.ready,self.pixels),daemon=True);self.process.start()
        if not self.ready.wait(15):self.close();raise TimeoutError('Geometry worker did not initialize')
        if not self.process.is_alive():self.close();raise RuntimeError('Geometry worker initialization failed')
    def submit(self,image,token):
        if self.closed or self.pending is not None:raise RuntimeError('Only one current geometry image allowed')
        if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3 or image.size>1080*1920*3:raise ValueError('Bounded current BGR image required')
        self.pending=token;np.frombuffer(self.pixels,dtype=np.uint8)[:image.size]=image.reshape(-1)
        self.requests.put_nowait((token,image.shape))
        from lamp_cpu_worker import Pending
        return Pending(self,token)
