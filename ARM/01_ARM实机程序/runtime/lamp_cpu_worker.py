"""One in-flight current-image CPU classifier job overlapping road NPU work."""
import multiprocessing as mp
import queue,time
import numpy as np


def worker(root,requests,replies,ready):
    try:
        import cv2,onnxruntime as ort
        from pathlib import Path
        from fast_types import FastTypes
        from lamp_attrs_portable import PortableAttributes
        from signal_engine import SignalEngine
        cv2.setNumThreads(1);root=Path(root)
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        options.add_session_config_entry('session.intra_op.allow_spinning','0')
        engine=object.__new__(SignalEngine)
        engine.guard=ort.InferenceSession(str(root/'models/color_guard.onnx'),options,providers=['CPUExecutionProvider'])
        engine.fast_types=FastTypes(root/'models/lamp_types_whole.onnx',2)
        class LazyDirection:
            model=None
            def predict(self,*args,**kwargs):
                if self.model is None:self.model=PortableAttributes(root/'models/lamp_attributes.onnx',threads=2)
                return self.model.predict(*args,**kwargs)
        engine.attributes=LazyDirection();ready.set()
        while True:
            request=requests.get()
            if request is None:break
            token,image,proposals,allow_direction=request
            rows,elapsed=engine.classify(image,proposals,allow_direction)
            replies.put((token,rows,elapsed,None))
    except BaseException as exc:
        ready.set()
        try:replies.put_nowait((None,None,None,repr(exc)))
        except queue.Full:pass


class Pending:
    def __init__(self,parent,token):self.parent,self.token=parent,token
    def result(self,timeout=5.):return self.parent.take(self.token,timeout)


class CpuLampWorker:
    def __init__(self,root):
        context=mp.get_context('spawn');self.requests=context.Queue(maxsize=1);self.replies=context.Queue(maxsize=1)
        self.ready=context.Event();self.pending=None;self.closed=False
        self.process=context.Process(target=worker,args=(str(root),self.requests,self.replies,self.ready),daemon=True);self.process.start()
        if not self.ready.wait(20):self.close();raise TimeoutError('CPU lamp models did not initialize')
        if not self.process.is_alive():self.close();raise RuntimeError('CPU lamp worker initialization failed')
    def submit(self,image,proposals,allow_direction,token):
        if self.closed or self.pending is not None:raise RuntimeError('Only one current CPU lamp job allowed')
        if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3 or image.size>1080*1920*3:raise ValueError('Bounded current BGR image required')
        self.pending=token;self.requests.put_nowait((token,image,proposals,allow_direction));return Pending(self,token)
    def take(self,token,timeout):
        if token!=self.pending:raise ValueError('CPU lamp token does not match in-flight image')
        try:actual,rows,elapsed,error=self.replies.get(timeout=timeout)
        except queue.Empty:raise TimeoutError('CPU lamp job did not finish')
        self.pending=None
        if error or actual!=token:raise RuntimeError('CPU lamp image mismatch/failure: '+str(error))
        return rows,elapsed
    def close(self):
        if self.closed:return
        self.closed=True
        try:self.requests.put_nowait(None)
        except queue.Full:pass
        self.process.join(2)
        if self.process.is_alive():self.process.terminate();self.process.join(2)
        for q in (self.requests,self.replies):q.cancel_join_thread();q.close()
