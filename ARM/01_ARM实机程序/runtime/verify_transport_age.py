import json,unittest
from basic70_wire import context_packet
from transport_age_guard import refresh_context

class Checks(unittest.TestCase):
    def wire(self):return context_packet(10,dict(camera_id=0,scene=1,valid_flags=99,epoch=3,age_units=60,arm_action=2))
    def test_age_advanced_at_write(self):
        raw,meta=refresh_context(self.wire(),100.5,100.401)
        self.assertEqual(raw[17],83);self.assertEqual(raw[18],2)
        self.assertEqual(sum(raw[:-1])&255,raw[-1])
    def test_old_grant_becomes_invalid_stop(self):
        raw,meta=refresh_context(self.wire(),100.5,100.462)
        self.assertEqual(raw[17],95);self.assertEqual(raw[18],1);self.assertFalse(raw[8]&64)
        self.assertTrue(meta['grant_suppressed_for_transport_budget'])
    def test_newer_age_never_reduced(self):
        raw=bytearray(self.wire());raw[17]=98;raw[-1]=sum(raw[:-1])&255
        new,_=refresh_context(bytes(raw),100.5,100.3);self.assertEqual(new[17],98)

if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=0).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    print(json.dumps(dict(tests=r.testsRun,passed=r.wasSuccessful(),errors=len(r.errors),failures=len(r.failures))))
    raise SystemExit(not r.wasSuccessful())
