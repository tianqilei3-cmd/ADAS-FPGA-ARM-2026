"""Bounded repeated-content suspicion, not an exposure timestamp guarantee."""
from collections import deque
import hashlib
import numpy as np


class ContentGuard:
    def __init__(self,maximum_static_seconds=2.,history=8):
        self.maximum_static_seconds=maximum_static_seconds;self.recent=deque(maxlen=history);self.last_novel=None
    def observe(self,image,decoded):
        digest=hashlib.sha256(memoryview(np.ascontiguousarray(image))).digest()
        if digest not in self.recent:
            self.recent.append(digest);self.last_novel=decoded
        age=max(0.,decoded-self.last_novel)
        return dict(content_change_age_ms=age*1000,content_freeze_suspected=age>self.maximum_static_seconds,
            scope='Repeated pixels only; sensor/FPGA source identity remains unverified')
