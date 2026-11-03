import json,unittest,time
import numpy as np,cv2
from geometry_cpu_worker import GeometryWorker
from arm_scene_a04 import frame_identity

class Checks(unittest.TestCase):
    def image(self):
        im=np.full((240,480,3),210,np.uint8)
        for x in (100,260):
            cv2.rectangle(im,(x-45,55),(x+10,75),(35,35,35),-1);cv2.circle(im,(x,65),5,(0,255,0),-1)
        return im
    def test_shared_current_image_round_trip(self):
        worker=GeometryWorker();im=self.image();token=(3,frame_identity(im,'test',0,3,time.monotonic())['image_identity'])
        try:
            (rows,meta),elapsed=worker.submit(im,token).result(timeout=5)
            self.assertEqual(len(rows),2);self.assertTrue(meta['shared_image_identity_verified']);self.assertEqual(meta['queue_capacity'],1)
        finally:worker.close()
    def test_mismatched_image_token_never_returns_geometry(self):
        worker=GeometryWorker()
        try:
            with self.assertRaises(RuntimeError):worker.submit(self.image(),(4,'0'*64)).result(timeout=5)
        finally:worker.close()

if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=0).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    print(json.dumps(dict(tests=r.testsRun,passed=r.wasSuccessful(),errors=len(r.errors),failures=len(r.failures))))
    raise SystemExit(not r.wasSuccessful())
