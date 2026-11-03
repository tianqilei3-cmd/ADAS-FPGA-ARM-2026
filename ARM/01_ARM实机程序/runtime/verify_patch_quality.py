import json,unittest
import numpy as np
from capture_quality import inspect_native_patches

class Checks(unittest.TestCase):
    def test_uniform_is_not_road(self):
        for color in (0,100,255):self.assertFalse(inspect_native_patches(np.full((360,640,3),color,np.uint8))['valid'])
    def test_strong_impulse_noise_is_invalid(self):
        im=np.random.default_rng(31).integers(0,256,(360,640,3),dtype=np.uint8)
        q=inspect_native_patches(im);self.assertFalse(q['valid']);self.assertEqual(q['reason'],'severe_impulse_noise')
    def test_no_model_pixel_is_changed(self):
        im=np.tile(np.linspace(0,255,640,dtype=np.uint8)[None,:,None],(360,1,3));before=im.copy()
        self.assertTrue(inspect_native_patches(im)['valid']);np.testing.assert_array_equal(im,before)

if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=0).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    print(json.dumps(dict(tests=r.testsRun,passed=r.wasSuccessful(),errors=len(r.errors),failures=len(r.failures))))
    raise SystemExit(not r.wasSuccessful())
