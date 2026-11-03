import json,unittest
import cv2,numpy as np
from geometry_lamp_proposals import propose

class Checks(unittest.TestCase):
    def test_two_measured_housings(self):
        im=np.full((240,480,3),210,np.uint8)
        for x in (100,260):
            cv2.rectangle(im,(x-45,55),(x+10,75),(35,35,35),-1)
            cv2.circle(im,(x,65),5,(0,255,0),-1)
        rows,meta=propose(im);self.assertFalse(meta['overloaded']);self.assertEqual(len(rows),2)
        for r in rows:
            self.assertTrue(r['confidence_is_geometric_quality_not_probability']);self.assertEqual(r['name'],'green')
            self.assertTrue(r['housing_refinement']['measurement_only'])
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as executor:
            parallel,parallel_meta=propose(im,executor)
        self.assertEqual(parallel,rows);self.assertEqual(parallel_meta,meta)
    def test_color_alone_does_not_supply_housing(self):
        im=np.full((240,480,3),210,np.uint8)
        for x in (100,260):cv2.circle(im,(x,65),5,(0,255,0),-1)
        rows,meta=propose(im);self.assertFalse(meta['overloaded']);self.assertEqual(rows,[])
    def test_coarse_search_measures_original_pixels_at_odd_offsets(self):
        from geometry_lamp_proposals import current_emitter_candidates
        from signal_s3_policy import round_emitter
        for offset in range(2):
            im=np.full((720,1280,3),210,np.uint8)
            for x in (301+offset,701+offset):
                cv2.rectangle(im,(x-45,95+offset),(x+10,115+offset),(35,35,35),-1)
                cv2.circle(im,(x,105+offset),5,(0,255,0),-1)
            before=im.copy();seeds,overloaded=current_emitter_candidates(im)
            self.assertFalse(overloaded);self.assertEqual(len(seeds),2)
            for cls,box in seeds:
                evidence=round_emitter(im,box,'green')
                self.assertTrue(evidence['valid']);self.assertAlmostEqual(evidence['center'][1],105+offset)
            rows,meta=propose(im);self.assertEqual(len(rows),2)
            np.testing.assert_array_equal(im,before)
    def test_crowded_input_rejected_instead_of_silent_topk(self):
        im=np.full((600,1600,3),210,np.uint8)
        for row in range(15):
            for col in range(16):cv2.circle(im,(20+col*50,20+row*20),4,(0,255,0),-1)
        rows,meta=propose(im);self.assertTrue(meta['overloaded']);self.assertEqual(rows,[])

if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=0).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    print(json.dumps(dict(tests=r.testsRun,passed=r.wasSuccessful(),errors=len(r.errors),failures=len(r.failures))))
    raise SystemExit(not r.wasSuccessful())
