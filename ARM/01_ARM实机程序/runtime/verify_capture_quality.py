import json,sys,unittest
import numpy as np
from capture_quality import prepare,invalidate_context


class Checks(unittest.TestCase):
    def test_noisy_uniform_input_stays_invalid(self):
        rng=np.random.default_rng(51);image=np.zeros((360,640,3),np.uint8);image[:,:,2]=250
        image[rng.random((360,640))<.03]=rng.integers(0,256,3,dtype=np.uint8)
        _,q=prepare(image)
        self.assertFalse(q['valid'])
        context=invalidate_context(dict(valid_flags=127,boundary=1,arm_action=2),q)
        self.assertEqual(context['valid_flags']&9,0);self.assertEqual(context['arm_action'],1)
    def test_textured_input_preserves_content(self):
        a=(np.indices((360,640)).sum(0)//20%2*200+20).astype(np.uint8)
        image=np.repeat(a[:,:,None],3,axis=2)
        result,q=prepare(image)
        self.assertTrue(q['valid']);self.assertEqual(result.shape,image.shape)
        self.assertTrue(np.array_equal(image[30:40,30:40],result[30:40,30:40]))
    def test_input_bytes_are_preserved_for_evidence(self):
        image=np.full((20,20,3),90,np.uint8);image[10,10]=255;copy=image.copy()
        result,_=prepare(image)
        self.assertTrue(np.array_equal(image,copy));self.assertEqual(int(result[10,10,0]),90)


if __name__=='__main__':
    r=unittest.TextTestRunner(stream=sys.stderr).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    print(json.dumps(dict(tests=r.testsRun,passed=r.wasSuccessful(),errors=len(r.errors),failures=len(r.failures))))
    sys.exit(0 if r.wasSuccessful() else 1)
