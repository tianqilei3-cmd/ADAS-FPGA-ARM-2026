import json,unittest
import cv2,numpy as np
from current_lamp_housing import refine

class Checks(unittest.TestCase):
    def scene(self):
        im=np.full((100,160,3),210,np.uint8);cv2.rectangle(im,(45,45),(100,65),(35,35,35),-1);cv2.circle(im,(90,55),5,(0,255,0),-1)
        return im,dict(name='green',confidence=.8,box=[82,45,100,65])
    def test_measured_housing_keeps_confidence(self):
        im,p=self.scene();r=refine(im,p);self.assertIn('housing_refinement',r)
        self.assertEqual(r['confidence'],p['confidence']);self.assertEqual(r['detector_box'],p['box'])
        self.assertLess(r['box'][0],50);self.assertGreater(r['box'][2]-r['box'][0],45)
    def test_no_dark_housing_unchanged(self):
        im,p=self.scene();im[:]=210;cv2.circle(im,(90,55),5,(0,255,0),-1)
        self.assertEqual(refine(im,p),p)
    def test_hollow_digit_not_a_filled_emitter(self):
        im,p=self.scene();cv2.circle(im,(90,55),5,(35,35,35),-1);cv2.circle(im,(90,55),5,(0,255,0),1)
        self.assertEqual(refine(im,p),p)
    def test_large_background_component_rejected(self):
        im,p=self.scene();cv2.rectangle(im,(45,65),(100,99),(35,35,35),-1)
        self.assertEqual(refine(im,p),p)

if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=0).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    print(json.dumps(dict(tests=r.testsRun,passed=r.wasSuccessful(),errors=len(r.errors),failures=len(r.failures))))
    raise SystemExit(not r.wasSuccessful())
