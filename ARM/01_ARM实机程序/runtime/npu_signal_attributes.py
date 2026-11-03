"""FP NPU adapters for the frozen color and grayscale whole-type graphs."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from lamp_attrs_preprocess import preprocess,direction_input
from lamp_attrs_portable import guarded_decode


def load_model(path):
    from rknnlite.api import RKNNLite
    model=RKNNLite(verbose=False)
    try:
        assert model.load_rknn(str(path))==0
        assert model.init_runtime()==0
        model.inference(inputs=[np.zeros((1,128,128,3),np.uint8)],data_format=['nhwc'])
    except BaseException:
        model.release();raise
    return model


class NpuColorGuard:
    on_npu=True
    def __init__(self,model):self.model=load_model(model)
    def get_inputs(self):return [SimpleNamespace(name='images')]
    def run(self,outputs,values):
        array=np.asarray(values['images'],np.float32)
        if array.ndim!=4 or array.shape[1:]!=(3,128,128):raise ValueError('Color input must be RGB NCHW 128')
        rows=[]
        for item in array:
            rgb=np.ascontiguousarray(np.rint(item.transpose(1,2,0)*255).clip(0,255).astype(np.uint8))
            result=self.model.inference(inputs=[rgb[None]],data_format=['nhwc'])
            rows.append(np.asarray(result[0],np.float32).reshape(1,6))
        return [np.concatenate(rows) if rows else np.empty((0,6),np.float32)]
    def close(self):self.model.release()


class NpuWholeTypes:
    def __init__(self,model):self.model=load_model(model)
    def logits(self,images):
        types=[];original=[];quality=[]
        for image in images:
            rgb=preprocess(image)
            values=self.model.inference(inputs=[rgb[None]],data_format=['nhwc'])
            types.append(np.asarray(values[0],np.float32).reshape(1,8))
            original.append(np.asarray(values[1],np.float32).reshape(1,6))
            quality.append(direction_input(image)[1])
        return (np.concatenate(types) if types else np.empty((0,8),np.float32),
                np.concatenate(original) if original else np.empty((0,6),np.float32),quality)
    def predict(self,images,batch=4):
        types,original,quality=self.logits(images)
        zero=np.zeros((len(images),4),np.float32)
        rows=guarded_decode(types,original,zero,quality,zero)
        for row in rows:row['type_source']='FP_NPU_whole_only_type_graph'
        return rows
    def close(self):self.model.release()
