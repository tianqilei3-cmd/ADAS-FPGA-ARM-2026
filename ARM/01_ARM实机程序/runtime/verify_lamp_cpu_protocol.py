"""A mismatched old result or a second in-flight image must not be consumed."""
import json,queue,unittest
import numpy as np
from lamp_cpu_worker import CpuLampWorker

class Checks(unittest.TestCase):
    def make(self):
        obj=object.__new__(CpuLampWorker);obj.pending=(20,'image-b');obj.closed=False;obj.replies=queue.Queue(1)
        return obj
    def test_old_reply_rejected(self):
        obj=self.make();obj.replies.put(((19,'image-a'),[{'name':'green'}],12.,None))
        with self.assertRaises(RuntimeError):obj.take((20,'image-b'),.01)
    def test_second_image_not_queued(self):
        obj=self.make()
        with self.assertRaises(RuntimeError):obj.submit(np.zeros((2,2,3),np.uint8),[],False,(21,'image-c'))

if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=0).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    print(json.dumps(dict(tests=r.testsRun,passed=r.wasSuccessful(),errors=len(r.errors),failures=len(r.failures))))
    raise SystemExit(not r.wasSuccessful())
