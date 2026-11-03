"""Static batch3 FP adapters. Padding rows never become scene observations."""
import numpy as np
from npu_signal_attributes import load_model,NpuColorGuard,NpuWholeTypes
from lamp_attrs_preprocess import preprocess,direction_input


def grouped(items,batch_size=3):
    for start in range(0,len(items),batch_size):
        rows=list(items[start:start+batch_size]);n=len(rows)
        array=np.stack(rows+[np.zeros_like(rows[0])]*(batch_size-n))
        yield array,n


class BatchedColorGuard(NpuColorGuard):
    def __init__(self,path,batch_size=3):
        self.batch_size=batch_size
        from rknnlite.api import RKNNLite
        self.model=RKNNLite(verbose=False)
        try:
            assert self.model.load_rknn(str(path))==0;assert self.model.init_runtime()==0
            self.model.inference(inputs=[np.zeros((batch_size,128,128,3),np.uint8)],data_format=['nhwc'])
        except BaseException:self.model.release();raise
    def run(self,outputs,values):
        array=np.asarray(values['images'],np.float32)
        if array.ndim!=4 or array.shape[1:]!=(3,128,128):raise ValueError('Expected RGB NCHW128')
        images=np.ascontiguousarray(np.rint(array.transpose(0,2,3,1)*255).clip(0,255).astype(np.uint8));rows=[]
        for rgb,n in grouped(images,self.batch_size):
            out=self.model.inference(inputs=[rgb],data_format=['nhwc'])
            rows.append(np.asarray(out[0],np.float32).reshape(self.batch_size,6)[:n])
        return [np.concatenate(rows) if rows else np.empty((0,6),np.float32)]


class BatchedWholeTypes(NpuWholeTypes):
    def __init__(self,path,batch_size=3):
        self.batch_size=batch_size
        from rknnlite.api import RKNNLite
        self.model=RKNNLite(verbose=False)
        try:
            assert self.model.load_rknn(str(path))==0;assert self.model.init_runtime()==0
            self.model.inference(inputs=[np.zeros((batch_size,128,128,3),np.uint8)],data_format=['nhwc'])
        except BaseException:self.model.release();raise
    def logits(self,images):
        rgb=[preprocess(im) for im in images];types=[];old=[]
        for batch,n in grouped(rgb,self.batch_size):
            values=self.model.inference(inputs=[batch],data_format=['nhwc'])
            types.append(np.asarray(values[0],np.float32).reshape(self.batch_size,8)[:n]);old.append(np.asarray(values[1],np.float32).reshape(self.batch_size,6)[:n])
        return (np.concatenate(types) if types else np.empty((0,8),np.float32),
            np.concatenate(old) if old else np.empty((0,6),np.float32),[direction_input(im)[1] for im in images])
