import json,sys,time,unittest,queue
from dataclasses import replace
import numpy as np
from competition_scene_logic import Controls,SceneController
from unified_policy import choose,context,legal_driver_state
from signal_engine import unknown_signal
from latest_live_capture_20261027 import LatestCaptureReader
from content_guard import ContentGuard
from capture_quality import prepare


class Checks(unittest.TestCase):
    def test_two_repeated_corrupted_frames_are_not_fresh_content(self):
        g=ContentGuard();a=np.zeros((64,64,3),np.uint8);b=a.copy();b[0,0]=255
        self.assertFalse(g.observe(a,10.)['content_freeze_suspected'])
        self.assertFalse(g.observe(b,10.1)['content_freeze_suspected'])
        self.assertTrue(g.observe(a,12.2)['content_freeze_suspected'])
        c=a.copy();c[31,37]=200;self.assertFalse(g.observe(c,12.3)['content_freeze_suspected'])
    def test_high_noise_is_not_valid_road(self):
        image=np.random.default_rng(15).integers(0,256,(360,640,3),dtype=np.uint8)
        _,q=prepare(image);self.assertFalse(q['valid']);self.assertEqual(q['reason'],'severe_impulse_noise')
    def test_looping_video_is_not_a_frozen_frame(self):
        g=ContentGuard();image=np.zeros((64,64,3),np.uint8)
        for i in range(180):
            image[0,0]=i%30
            self.assertFalse(g.observe(image,i/30)['content_freeze_suspected'])
    def fixture(self,state):
        now=time.monotonic()*1000
        frame=dict(camera_id='current',frame_id=3,captured_ms=now-20,image_identity='a'*64)
        controls=Controls(requested_motion='go');key=dict(supported=True,scene_index=0,sequence=19)
        risk=dict(action='GO',alarm=False,decision_valid=True,reason='clear',active_tracks=[])
        selected=dict(frame,selected_valid=state!='unknown',selected_state=state,simulation_association_valid=state!='unknown',lane_id=controls.lane_id,movement='straight',head_id='current-head')
        return now,frame,controls,key,risk,selected
    def test_red_stop_and_forced_alarm(self):
        now,frame,c,key,risk,selected=self.fixture('red')
        for forced in [False,True]:
            d=choose(frame,replace(c,forced=forced),key,None,risk,selected,SceneController(max_age_ms=500.),now)
            self.assertEqual(d['action'],'STOP');self.assertEqual(d['alarm'],forced)
    def test_green_requires_three_fresh_images(self):
        now,frame,c,key,risk,s=self.fixture('green');controller=SceneController(max_age_ms=500.)
        for i in range(3):
            f=dict(frame,frame_id=3+i,captured_ms=now+10*i);sel=dict(s,**f)
            d=choose(f,c,key,None,risk,sel,controller,now+10*i+20)
            self.assertEqual(d['action'],'GO' if i==2 else 'STOP')
    def test_unknown_or_hazard_no_go(self):
        now,frame,c,key,risk,s=self.fixture('unknown')
        d=choose(frame,c,key,None,risk,s,SceneController(max_age_ms=500.),now);self.assertEqual(d['action'],'STOP')
        risk.update(action='STOP',alarm=True)
        d=choose(frame,c,key,None,risk,s,SceneController(max_age_ms=500.),now);self.assertEqual(d['action'],'STOP');self.assertTrue(d['alarm'])
    def test_new_key_or_invalid_input_cannot_permit(self):
        now,frame,c,key,risk,s=self.fixture('green');d=dict(action='GO',decision_valid=True)
        for unchanged,valid in [(False,True),(True,False)]:
            tx=context(c,key,None,s,d,dict(objects=[]),risk,time.monotonic(),unchanged,dict(valid=valid))
            self.assertEqual(tx['arm_action'],1)
    def test_driver_state_has_no_lamp_answer(self):
        self.assertTrue(legal_driver_state(0x10));self.assertTrue(legal_driver_state(0x30));self.assertTrue(legal_driver_state(0x15))
        for value in [0x13,0x18,0x20,0x50,-1,None]:self.assertFalse(legal_driver_state(value))
    def test_stale_capture_is_skipped_not_retimed(self):
        reader=LatestCaptureReader.__new__(LatestCaptureReader);reader.closed=False;reader.read_timeout=.2;reader.previous=0;reader.info=[];reader.roi=None;reader.source='/dev/video0';reader.uniform_rejected=0
        class Process:
            def is_alive(self):return True
        class Stop:
            def wait(self,_):pass
        class Mailbox:
            def __init__(self):self.calls=0;self.captured=time.monotonic()
            def take(self,*_):
                self.calls+=1
                if self.calls==1:raise TimeoutError('one stale frame')
                image=np.tile(np.arange(64,dtype=np.uint8)[None,:,None],(64,1,3))*3
                return 2,self.captured-.01,self.captured,image
        reader.process=Process();reader.stop=Stop();reader.notices=queue.Queue();reader.mailbox=Mailbox()
        frame=reader.read();self.assertEqual(frame.decoded_at,reader.mailbox.captured);self.assertEqual(frame.source_frame,2)


if __name__=='__main__':
    r=unittest.TextTestRunner(stream=sys.stderr).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    print(json.dumps(dict(tests=r.testsRun,passed=r.wasSuccessful(),errors=len(r.errors),failures=len(r.failures))))
    sys.exit(0 if r.wasSuccessful() else 1)
