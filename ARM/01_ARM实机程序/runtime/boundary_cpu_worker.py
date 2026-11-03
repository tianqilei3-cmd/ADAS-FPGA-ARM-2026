"""One fixed shared-buffer CPU job, overlapping the serial road NPU call."""
import multiprocessing as mp
import numpy as np


def classify(logits):
    from boundary_softmax import softmax3
    p=softmax3(logits)
    one=(p[1]>p[0])&(p[1]>=p[2]);two=(p[2]>p[0])&(p[2]>p[1])
    classes=np.zeros(p.shape[1:],np.uint8)
    classes[one&(p[1]>=.5)]=1;classes[two&(p[2]>=.5)]=2
    return classes


def worker(pipe,logits,classes):
    import cv2,time
    cv2.setNumThreads(1)
    try:
        shared=np.frombuffer(logits,dtype=np.float32).reshape(3,288,512)
        output=np.frombuffer(classes,dtype=np.uint8).reshape(288,512)
        pipe.send(('ready',))
        while True:
            command,token=pipe.recv()
            if command=='stop':break
            if command!='classify':raise ValueError('Invalid CPU request')
            started=time.monotonic();output[:]=classify(shared)
            pipe.send(('done',token,(time.monotonic()-started)*1000))
    except BaseException as exc:
        try:pipe.send(('error',repr(exc)))
        except (OSError,EOFError):pass
    finally:pipe.close()


class BoundaryCPU:
    def __init__(self):
        self.context=mp.get_context('spawn')
        self.logits=self.context.RawArray('f',3*288*512)
        self.classes=self.context.RawArray('B',288*512)
        self.pipe,child=self.context.Pipe()
        self.process=self.context.Process(target=worker,args=(child,self.logits,self.classes),daemon=True)
        self.pending=None;self.closed=False;self.process.start();child.close()
        if not self.pipe.poll(8) or self.pipe.recv()!=('ready',):
            self.close();raise RuntimeError('CPU worker failed to start')
    def submit(self,logits,token):
        if self.closed or self.pending is not None:raise RuntimeError('CPU buffer already occupied or closed')
        if logits.shape!=(3,288,512) or logits.dtype!=np.float32:raise ValueError('Wrong logit buffer contract')
        np.frombuffer(self.logits,dtype=np.float32)[:]=logits.reshape(-1)
        self.pending=token;self.pipe.send(('classify',token))
    def take(self,token):
        if self.closed or token!=self.pending:raise ValueError('Current image identity required')
        if not self.pipe.poll(2):raise TimeoutError('Current CPU job did not complete')
        answer=self.pipe.recv()
        if answer[:2]!=('done',token):raise RuntimeError('CPU response identity/error: '+repr(answer))
        self.pending=None
        return np.frombuffer(self.classes,dtype=np.uint8).reshape(288,512).copy(),answer[2]
    def close(self):
        if self.closed:return
        self.closed=True
        try:self.pipe.send(('stop',None))
        except (OSError,EOFError):pass
        self.process.join(timeout=1)
        if self.process.is_alive():self.process.terminate();self.process.join(timeout=1)
        self.pipe.close()
