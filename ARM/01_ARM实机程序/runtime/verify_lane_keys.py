"""Physical intent parsing and fail-closed lane decisions; no hardware verdict."""
import json,sys,time,unittest
import numpy as np
from basic70_wire import ReplyStream
from physical_keys import decode_keys,KeyTracker
from arm_scene_a04 import frame_identity
from competition_scene_logic import SceneController
from run_lane_demo import decide,make_context

def keys(state,sequence=1):
    p=bytes([0x55,3,2,state,sequence]);return p+bytes([sum(p)&255])

class Checks(unittest.TestCase):
    def setup_case(self,state=0x15,line='dashed'):
        controls=decode_keys(keys(state))['controls']
        frame=frame_identity(np.zeros((20,40,3),np.uint8),'fixture',0,1,time.monotonic())
        value=dict(line_type=line,state_valid=line!='unknown',simulation_association_valid=line!='unknown')
        boundaries=dict(frame,left=value,right=value)
        risk=dict(action='GO',decision_valid=True,alarm=False,reason='clear',active_tracks=[])
        return controls,frame,boundaries,risk
    def test_six_byte_fragment_and_legacy_parser(self):
        parser=ReplyStream();a=keys(0x15,3);legacy=bytes([0x55,3,1,0,0x59])
        self.assertEqual(parser.feed(a[:4]),[])
        self.assertEqual(parser.feed(a[4:]+legacy+bytes.fromhex('55 13 68')),[a,legacy,bytes.fromhex('55 13 68')])
    def test_bad_checksum_and_reserved_values(self):
        for raw in [keys(0x43),keys(0x23),keys(0x15)[:-1]+b'\x00',bytes.fromhex('55 03 01 00 59')]:
            with self.assertRaises(ValueError):decode_keys(raw)
    def test_startup_is_stop(self):
        self.assertEqual(decode_keys(keys(0,0))['controls'].requested_motion,'stop')
    def test_remote_driver_is_explicitly_labelled(self):
        snapshot=decode_keys(keys(0x55))
        self.assertEqual(snapshot['source'],'remote_test_driver')
        self.assertEqual(snapshot['controls'].requested_turn,'left')
    def test_force_without_drive_rejected(self):
        with self.assertRaises(ValueError):decode_keys(keys(0x25))
    def test_right_and_straight_mapping(self):
        self.assertEqual(decode_keys(keys(0x16))['controls'].requested_turn,'right')
        self.assertEqual(decode_keys(keys(0x14))['controls'].requested_turn,'none')
    def test_sequence_wrap_and_hold_are_idempotent(self):
        t=KeyTracker();t.update(decode_keys(keys(0x15,255)))
        self.assertTrue(t.update(decode_keys(keys(0x16,0)))['changed'])
        self.assertFalse(t.update(decode_keys(keys(0x16,0)))['changed'])
    def test_state_without_sequence_is_rejected(self):
        t=KeyTracker();t.update(decode_keys(keys(0x15,3)))
        with self.assertRaises(ValueError):t.update(decode_keys(keys(0x16,3)))
    def test_dashed_left_allows_current_virtual_change(self):
        c,f,b,r=self.setup_case();d=decide(f,c,b,r,SceneController(max_age_ms=500),f['captured_ms']+100)
        self.assertEqual(d['action'],'CHANGE_LEFT');self.assertFalse(d['physical_lane_association_verified'])
    def test_solid_blocks_and_forced_alarms(self):
        for state,alarm in [(0x15,False),(0x35,True)]:
            c,f,b,r=self.setup_case(state,'solid');d=decide(f,c,b,r,SceneController(max_age_ms=500),f['captured_ms']+100)
            self.assertEqual(d['action'],'KEEP_LANE');self.assertEqual(d['alarm'],alarm)
    def test_unknown_or_different_image_never_allows_change(self):
        for line,mismatch in [('unknown',False),('dashed',True)]:
            c,f,b,r=self.setup_case(line=line)
            if mismatch:b['frame_id']=2
            d=decide(f,c,b,r,SceneController(max_age_ms=500),f['captured_ms']+100)
            self.assertEqual(d['action'],'STOP');self.assertFalse(d['decision_valid'])
    def test_physical_stop_is_prioritized(self):
        c,f,b,r=self.setup_case(0x05);d=decide(f,c,b,r,SceneController(max_age_ms=500),f['captured_ms']+100)
        self.assertEqual(d['action'],'STOP')
    def test_risk_and_unknown_clearance_block_change(self):
        for risk in ['STOP','HOLD']:
            c,f,b,r=self.setup_case();r.update(action=risk,alarm=risk=='STOP')
            d=decide(f,c,b,r,SceneController(max_age_ms=500),f['captured_ms']+100);self.assertEqual(d['action'],'STOP')
    def test_changed_key_cannot_authorize_old_image(self):
        c,f,b,r=self.setup_case();d=dict(action='CHANGE_LEFT',decision_valid=True)
        context=make_context(c,b,d,dict(objects=[]),r,5,time.monotonic(),False)
        self.assertEqual(context['arm_action'],1);self.assertEqual(context['valid_flags']&40,0)
    def test_forced_solid_context_preserves_alarm_request(self):
        c,f,b,r=self.setup_case(0x35,'solid');d=decide(f,c,b,r,SceneController(max_age_ms=500),f['captured_ms']+100)
        context=make_context(c,b,d,dict(objects=[]),r,5,time.monotonic(),True)
        self.assertEqual(context['boundary'],2);self.assertTrue(context['control_flags']&16);self.assertEqual(context['arm_action'],2)

if __name__=='__main__':
    r=unittest.TextTestRunner(stream=sys.stderr).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    print(json.dumps(dict(tests=r.testsRun,passed=r.wasSuccessful(),errors=len(r.errors),failures=len(r.failures))))
    sys.exit(0 if r.wasSuccessful() else 1)
