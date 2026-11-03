import json,sys,unittest
import numpy as np
from boundary_cpu_worker import BoundaryCPU,classify
from compact_boundary_post import current_masks,filter_classes


class Checks(unittest.TestCase):
    def test_single_buffer_identity_and_results(self):
        rng=np.random.default_rng(13);logits=rng.normal(size=(3,288,512)).astype(np.float32)
        w=BoundaryCPU()
        try:
            token=(17,'a'*64);w.submit(logits,token)
            with self.assertRaises(RuntimeError):w.submit(logits,(18,'b'*64))
            with self.assertRaises(ValueError):w.take((18,'b'*64))
            result,_=w.take(token)
            self.assertTrue(np.array_equal(result,classify(logits)))
            from boundary_softmax import softmax3
            objects=[dict(name='vehicle',confidence=.8,box=[200,200,400,350])]
            old,_=current_masks(softmax3(logits),(360,640,3),objects)
            fast,_=filter_classes(result,(360,640,3),objects)
            self.assertTrue(np.array_equal(old,fast))
        finally:w.close()
        self.assertFalse(w.process.is_alive())


if __name__=='__main__':
    r=unittest.TextTestRunner(stream=sys.stderr).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    print(json.dumps(dict(tests=r.testsRun,passed=r.wasSuccessful(),errors=len(r.errors),failures=len(r.failures))))
    sys.exit(0 if r.wasSuccessful() else 1)
