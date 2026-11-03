"""Frozen typed-boundary network and its existing conservative geometry policy."""
import time
from pathlib import Path
import cv2,numpy as np
import lane_boundary_wide_recovery as recovery
import lane_boundary_segments_v3 as typed_segments
from boundary_policy import requested_boundaries
from row_boundary_post import process,process_classes
from boundary_softmax import softmax3

class BoundaryRuntime:
    def __init__(self,path):
        from rknnlite.api import RKNNLite
        self.model=RKNNLite(verbose=False)
        try:
            assert self.model.load_rknn(str(path))==0;assert self.model.init_runtime()==0
            self.model.inference(inputs=[np.zeros((1,288,512,3),np.uint8)],data_format=['nhwc'])
        except BaseException:self.model.release();raise
    def infer(self,image):
        started=time.monotonic();rgb=np.ascontiguousarray(cv2.cvtColor(cv2.resize(image,(512,288)),cv2.COLOR_BGR2RGB))
        raw=self.model.inference(inputs=[rgb[None]],data_format=['nhwc'])
        logits=np.asarray(raw[0],np.float32).reshape(3,288,512).copy()
        return logits,(time.monotonic()-started)*1000
    @staticmethod
    def probabilities(logits):
        started=time.monotonic();p=softmax3(logits)
        return p,(time.monotonic()-started)*1000
    def finish(self,p,image,frame,objects,model_ms):
        started=time.monotonic()
        extraction,boundaries=process(p,image,objects,frame)
        return dict(boundaries=boundaries,
            paths=extraction['instances'],model_ms=model_ms,postprocess_ms=(time.monotonic()-started)*1000)
    def predict(self,image,frame,objects):
        logits,infer_ms=self.infer(image);p,normalize_ms=self.probabilities(logits)
        return self.finish(p,image,frame,objects,infer_ms+normalize_ms)
    def finish_classes(self,classes,image,frame,objects,model_ms):
        started=time.monotonic();extraction,boundaries=process_classes(classes,image,objects,frame)
        return dict(boundaries=boundaries,paths=extraction['instances'],model_ms=model_ms,
            postprocess_ms=(time.monotonic()-started)*1000)
    def close(self):self.model.release()
