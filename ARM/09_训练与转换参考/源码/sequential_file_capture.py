"""Exact-frame file reading: decode short forward gaps instead of random seek.

No background buffering, time relabeling, or live-camera support. Position errors
and decode failures raise, so a previous image can never masquerade as a new frame.
"""
import math
import time

import cv2


class SequentialFileCapture:
    def __init__(self, capture, max_forward_skip=32):
        if type(max_forward_skip) is not int or max_forward_skip < 0:
            raise ValueError('Invalid forward skip limit')
        self.capture = capture
        self.max_forward_skip = max_forward_skip
        self.next_index = 0
        self.target = 0
        self.closed = False
        self.failed = False
        self.records = []
        self.random_seeks = 0
        self.grabbed_frames = 0

    def isOpened(self):
        return not self.closed and not self.failed and self.capture.isOpened()

    def get(self, prop):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self.target)
        return self.capture.get(prop)

    def set(self, prop, value):
        if not self.isOpened():
            raise RuntimeError('Capture is closed or failed')
        if prop != cv2.CAP_PROP_POS_FRAMES:
            raise ValueError('This exact-frame reader only allows frame-position changes')
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or value != int(value):
            raise ValueError('Frame index must be a nonnegative integer')
        self.target = int(value)
        return True

    def read(self):
        if not self.isOpened():
            raise RuntimeError('Capture is closed or failed')
        started = time.perf_counter()
        target = self.target
        gap = target - self.next_index
        skipped = 0
        mode = 'sequential'
        try:
            if gap < 0 or gap > self.max_forward_skip:
                mode = 'random_seek'
                if not self.capture.set(cv2.CAP_PROP_POS_FRAMES, target):
                    raise RuntimeError('Backend rejected requested seek')
                self.random_seeks += 1
                actual = self.capture.get(cv2.CAP_PROP_POS_FRAMES)
                if not math.isfinite(actual) or abs(actual - target) > .01:
                    raise RuntimeError('Backend seek position mismatch')
            else:
                for _ in range(gap):
                    if not self.capture.grab():
                        raise RuntimeError('Cannot decode skipped frame')
                    skipped += 1
                self.grabbed_frames += skipped
            ok, image = self.capture.read()
            if not ok or image is None:
                raise RuntimeError('Cannot decode requested frame')
            actual = self.capture.get(cv2.CAP_PROP_POS_FRAMES)
            if not math.isfinite(actual) or abs(actual - target - 1) > .01:
                raise RuntimeError('Backend decoded frame position mismatch')
            self.next_index = self.target = target + 1
            self.records.append(dict(frame_id=target, mode=mode, grabbed=skipped,
                                     decode_ms=(time.perf_counter() - started) * 1000))
            return True, image
        except Exception:
            self.failed = True
            raise

    def release(self):
        if not self.closed:
            self.closed = True
            self.capture.release()

