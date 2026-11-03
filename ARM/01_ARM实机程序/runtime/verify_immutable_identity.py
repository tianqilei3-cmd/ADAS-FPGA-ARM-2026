import hashlib,json,unittest
from unittest.mock import patch
import numpy as np
import immutable_identity as identity
from latest_live_capture_20261027 import LatestMailbox


class Checks(unittest.TestCase):
    def test_identical_original_sha_contract_and_reuse(self):
        a=np.arange(90,dtype=np.uint8).reshape(5,6,3)
        frozen=np.frombuffer(a.tobytes(),np.uint8).reshape(a.shape)
        h=hashlib.sha256(a.tobytes());h.update(str(a.shape).encode('ascii'))
        with patch.object(identity.hashlib,'sha256',wraps=hashlib.sha256) as sha:
            self.assertEqual(identity.pixel_digest(frozen),h.hexdigest())
            self.assertEqual(identity.pixel_digest(frozen),h.hexdigest());self.assertEqual(sha.call_count,1)
        with self.assertRaises(ValueError):frozen.setflags(write=True)

    def test_readonly_view_of_mutable_storage_must_be_rehashed(self):
        base=np.zeros((8,8,3),np.uint8);view=base.view();view.setflags(write=False)
        first=identity.pixel_digest(view);base[3,3]=255
        self.assertNotEqual(first,identity.pixel_digest(view))

    def test_shape_change_must_change_identity(self):
        a=np.frombuffer(bytes(96),np.uint8).reshape(4,8,3);first=identity.pixel_digest(a)
        a.shape=(8,4,3);self.assertNotEqual(first,identity.pixel_digest(a))

    def test_mailbox_copy_remains_bound_to_original_frame(self):
        box=LatestMailbox();a=np.zeros((8,8,3),np.uint8);box.publish(a,1.,1.1)
        _,_,_,first=box.take(0,1.2,.5);digest=identity.pixel_digest(first)
        box.publish(np.full_like(a,99),1.3,1.4)
        self.assertEqual(identity.pixel_digest(first),digest);self.assertFalse(first.any())
        with self.assertRaises(ValueError):first[0,0]=255
        _,_,_,second=box.take(1,1.45,.5);self.assertNotEqual(identity.pixel_digest(second),digest)

    def test_cache_does_not_retain_image_buffers_and_is_bounded(self):
        import gc,weakref
        for _ in range(20):
            a=np.frombuffer(bytes(96),np.uint8).reshape(4,8,3);identity.pixel_digest(a)
        ref=weakref.ref(a);del a;gc.collect();self.assertIsNone(ref());self.assertLessEqual(len(identity._cache),8)


if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=0).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    print(json.dumps(dict(tests=r.testsRun,passed=r.wasSuccessful(),errors=len(r.errors),failures=len(r.failures))))
    raise SystemExit(not r.wasSuccessful())
