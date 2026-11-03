import json,sys,unittest
import cv2,numpy as np
from row_boundary_post import from_mask,trace
import lane_boundary_instances as base
from lane_boundary_instances_v2 import trace_without_stale_fragments


class Checks(unittest.TestCase):
    def fixture(self,label=2):
        image=np.zeros((288,512,3),np.uint8);mask=np.zeros((288,512),np.uint8)
        cv2.line(mask,(120,180),(185,267),label,4)
        return image,mask
    def test_observed_typed_path(self):
        for label,name in [(1,'solid'),(2,'dashed')]:
            image,mask=self.fixture(label);_,b=from_mask(mask,image,[],{})
            self.assertEqual(b['left']['line_type'],name)
            self.assertFalse(b['physical_lane_association_verified'])
    def test_conflict_stays_unknown(self):
        image,mask=self.fixture();mask[(mask>0)&(np.indices(mask.shape)[0]>220)]=1
        _,b=from_mask(mask,image,[],{});self.assertFalse(b['left']['state_valid'])
    def test_no_observed_reference_no_extrapolation(self):
        image,mask=self.fixture();mask[230:]=0
        _,b=from_mask(mask,image,[],{});self.assertFalse(b['left']['state_valid'])
    def test_occluder_never_bridged(self):
        image,mask=self.fixture();objects=[dict(name='vehicle',confidence=.9,box=[0,220,250,252])]
        _,b=from_mask(mask,image,objects,{});self.assertFalse(b['left']['state_valid'])
    def test_solid_gap_does_not_invent_dashed(self):
        image,mask=self.fixture(1);mask[206:209]=0
        e,_=from_mask(mask,image,[],{});self.assertFalse(any(p['line_type']=='dashed' for p in e['instances']))
    def test_trace_parity_with_retained_v2(self):
        for seed in range(6):
            image,mask=self.fixture(1+seed%2);mask[210:210+seed]=0
            cv2.line(mask,(400,180),(350,268),1,4)
            if seed%2:mask[230:250,200:220]=2
            p=base.InstancePolicy();blocked=np.zeros_like(mask,dtype=bool)
            old=trace_without_stale_fragments(mask,blocked,p);fast=trace(mask,blocked,p)
            key=lambda t:(t['points'],t['counts'],t['widths'],t['ambiguous'])
            self.assertEqual(sorted(map(key,old)),sorted(map(key,fast)))


if __name__=='__main__':
    r=unittest.TextTestRunner(stream=sys.stderr).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    print(json.dumps(dict(tests=r.testsRun,passed=r.wasSuccessful(),errors=len(r.errors),failures=len(r.failures))))
    sys.exit(0 if r.wasSuccessful() else 1)
