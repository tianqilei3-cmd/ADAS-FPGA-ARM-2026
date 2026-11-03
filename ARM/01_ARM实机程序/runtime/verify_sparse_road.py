"""Rectangular stride and threshold equivalence for the road raw-head path."""
import json,unittest
import numpy as np
from vision_common import decoded_output,decode


def tensors():
    values=[]
    for h,w in [(48,80),(24,40),(12,20)]:
        values.extend([np.zeros((1,8,h,w),np.float32),np.zeros((1,4,h,w),np.float32)])
    return values


class Checks(unittest.TestCase):
    def test_known_rectangular_stride(self):
        out=tensors();out[1][0,1,20,10]=.9
        g=dict(gain=1.,left=0,top=0,width=640,height=384)
        rows=decode(decoded_output(out,[384,640],4,.25),g,['light','vehicle','bike','person'],.25,.45,100)
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['class_id'],1)
        np.testing.assert_allclose(rows[0]['box'],[80,160,88,168],atol=1e-5)
    def test_sparse_dense_same_detections(self):
        out=tensors();rng=np.random.default_rng(28)
        for i in range(0,6,2):
            out[i][:]=rng.normal(size=out[i].shape)
            out[i+1][:]=rng.uniform(0,.26,size=out[i+1].shape)
        g=dict(gain=1.,left=0,top=0,width=640,height=384)
        a=decode(decoded_output(out,[384,640],4),g,['light','vehicle','bike','person'],.25,.45,100)
        b=decode(decoded_output(out,[384,640],4,.25),g,['light','vehicle','bike','person'],.25,.45,100)
        self.assertEqual(a,b)
    def test_nonfinite_not_hidden_by_confidence(self):
        out=tensors();out[0][0,0,0,0]=np.nan
        with self.assertRaises(ValueError):decoded_output(out,[384,640],4,.25)


if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=0).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    print(json.dumps(dict(tests=r.testsRun,passed=r.wasSuccessful(),errors=len(r.errors),failures=len(r.failures))))
    raise SystemExit(not r.wasSuccessful())
