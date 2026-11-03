"""Numerically equivalent whole-crop type heads; no direction authorization."""
from pathlib import Path
import numpy as np
from lamp_attrs_preprocess import preprocess,direction_input
from lamp_attrs_portable import guarded_decode


class FastTypes:
    def __init__(self,model,threads=2):
        import onnxruntime as ort
        options=ort.SessionOptions();options.intra_op_num_threads=threads;options.inter_op_num_threads=1
        options.add_session_config_entry('session.intra_op.allow_spinning','0')
        self.session=ort.InferenceSession(str(model),options,providers=['CPUExecutionProvider'])

    def predict(self,images,batch=4):
        rows=[]
        for start in range(0,len(images),batch):
            crops=images[start:start+batch]
            x=np.stack([preprocess(im) for im in crops]).transpose(0,3,1,2).astype(np.float32)/255
            types,original=self.session.run(None,{'whole':x})
            q=[direction_input(im)[1] for im in crops]
            zero=np.zeros((len(crops),4),np.float32)
            result=guarded_decode(types,original,zero,q,zero)
            for row in result:row['type_source']='equivalent_whole_only_type_graph'
            rows.extend(result)
        return rows
