import json,sys,unittest
import cv2,numpy as np
from compact_boundary_post import from_mask,current_masks
from lane_boundary_wide_recovery import prepare_evidence
class Checks(unittest.TestCase):
    def test_current_mask_parity(self):
        rng=np.random.default_rng(173)
        for seed in range(5):
            labels=rng.integers(0,3,(36,64),dtype=np.uint8)
            labels=cv2.resize(labels,(512,288),interpolation=cv2.INTER_NEAREST)
            p=np.eye(3,dtype=np.float32)[labels].transpose(2,0,1)*.8+.2/3
            if seed==0:p[:,20:30,20:30]=1/3
            objects=[dict(name='vehicle',confidence=.9,box=[250,170,350,250]),dict(name='person',confidence=.4,box=[-12,300,70,370])]
            old,classes,_=prepare_evidence(p,(360,640,3),objects)
            fast,current=current_masks(p,(360,640,3),objects)
            self.assertTrue(np.array_equal(old,fast));self.assertTrue(np.array_equal(classes,current))
    def fixture(self,label=2):
        image=np.zeros((288,512,3),np.uint8);mask=np.zeros((288,512),np.uint8)
        cv2.line(mask,(120,180),(185,267),label,4)
        return image,mask
    def test_typed_dashed_component(self):
        image,mask=self.fixture();_,b=from_mask(mask,mask,image,{})
        self.assertEqual(b['left']['line_type'],'dashed')
        self.assertFalse(b['physical_lane_association_verified'])
    def test_solid_never_becomes_dashed_from_gaps(self):
        image,mask=self.fixture(1);mask[205:208]=0
        e,_=from_mask(mask,mask,image,{})
        self.assertFalse(any(x['line_type']=='dashed' for x in e['instances']))
    def test_mixed_style_unknown(self):
        image,mask=self.fixture();mask[(mask>0)&(np.indices(mask.shape)[0]>220)]=1
        _,b=from_mask(mask,mask,image,{})
        self.assertFalse(b['left']['state_valid'])
    def test_rejected_nearer_semantics_block(self):
        image,mask=self.fixture();classes=mask.copy();classes[232:240,230:235]=1
        _,b=from_mask(mask,classes,image,{})
        self.assertFalse(b['left']['state_valid'])
    def test_crosswalk_not_a_parallel_boundary(self):
        image=np.zeros((288,512,3),np.uint8);mask=np.zeros((288,512),np.uint8);mask[232:237,100:240]=1
        _,b=from_mask(mask,mask,image,{})
        self.assertFalse(b['left']['state_valid'])
    def test_overload_unknown(self):
        image=np.zeros((288,512,3),np.uint8);mask=np.zeros((288,512),np.uint8)
        for y in range(10,280,20):
            for x in range(10,500,20):mask[y:y+2,x:x+2]=1
        _,b=from_mask(mask,mask,image,{})
        self.assertFalse(b['left']['state_valid']);self.assertFalse(b['right']['state_valid'])
if __name__=='__main__':
    r=unittest.TextTestRunner(stream=sys.stderr).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    print(json.dumps(dict(tests=r.testsRun,passed=r.wasSuccessful(),errors=len(r.errors),failures=len(r.failures))))
    sys.exit(0 if r.wasSuccessful() else 1)
