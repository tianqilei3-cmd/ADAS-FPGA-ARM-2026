import json,sys,unittest
import numpy as np
import cv2
from boundary_softmax import softmax3
from boundary_runtime import BoundaryRuntime


class Checks(unittest.TestCase):
    def test_probabilities_and_threshold_decisions_match(self):
        rng=np.random.default_rng(193)
        for scale in [0.,1.,5.,40.]:
            logits=(rng.normal(size=(3,288,512))*scale).astype(np.float32)
            old=np.exp(logits-logits.max(0,keepdims=True));old/=old.sum(0,keepdims=True)
            new=softmax3(logits)
            np.testing.assert_allclose(new,old,atol=3e-7,rtol=2e-6)
            a=old.argmax(0);a[old.max(0)<.5]=0
            b=new.argmax(0);b[new.max(0)<.5]=0
            self.assertTrue(np.array_equal(a,b))
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):softmax3(np.full((3,32,32),np.nan,np.float32))
    def test_split_output_owns_buffer_and_keeps_current_identity(self):
        class FakeModel:
            def __init__(self):self.buffer=np.zeros((1,3,288,512),np.float32)
            def inference(self,**kwargs):
                self.buffer.fill(-5);self.buffer[:,0]=0
                label=2 if kwargs['inputs'][0].max()==0 else 1
                cv2.line(self.buffer[0,label],(120,180),(185,267),10,4)
                return [self.buffer]
        model=BoundaryRuntime.__new__(BoundaryRuntime);model.model=FakeModel()
        image=np.zeros((288,512,3),np.uint8)
        logits,elapsed=model.infer(image)
        model.infer(np.full_like(image,255))
        p,normalization=model.probabilities(logits)
        frame=dict(camera_id='current-test-camera',frame_id=17,captured_ms=20.,image_identity='d'*64)
        result=model.finish(p,image,frame,[],elapsed+normalization)
        self.assertEqual(result['boundaries']['left']['line_type'],'dashed')
        self.assertEqual(result['boundaries']['image_identity'],frame['image_identity'])
        self.assertEqual(result['boundaries']['frame_id'],17)


if __name__=='__main__':
    r=unittest.TextTestRunner(stream=sys.stderr).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    print(json.dumps(dict(tests=r.testsRun,passed=r.wasSuccessful(),errors=len(r.errors),failures=len(r.failures))))
    sys.exit(0 if r.wasSuccessful() else 1)
