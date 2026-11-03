"""Bounded latest-frame capture with one shared image and cancellable reader.

The timestamp is userspace decode completion, not a sensor or FPGA timestamp.
No file fallback and no reuse of an old frame after timeout/input failure.
"""
from dataclasses import dataclass
from contextlib import contextmanager
import multiprocessing as mp
import queue
import time
from content_guard import ContentGuard

import numpy as np

MAX_SHAPE = (1080, 1920, 3)


@dataclass
class LiveFrame:
    image: object
    source: str
    source_frame: int
    source_visit: int
    source_switched: bool
    read_started: float
    decoded_at: float
    standby_wait_ms: float = 0.
    source_open_ms: float = 0.
    request_started: float = 0.
    delivered_at: float = 0.
    prefetched_first_frame: bool = False
    dropped_before_delivery: int = 0
    freshness_clock: str = 'userspace_decode_completion_not_sensor_timestamp'
    fpga_frame_identity_verified: bool = False
    source_parent_size: object = None
    capture_roi_pixels: object = None




def decode_yuyv_roi(raw,roi):
    import cv2
    if raw.dtype!=np.uint8 or raw.ndim!=3 or raw.shape[2]!=2:
        raise ValueError('Expected physical packed YUYV H W 2 capture')
    h,w=raw.shape[:2]
    if len(roi)!=4 or not 0<=roi[0]<roi[2]<=1 or not 0<=roi[1]<roi[3]<=1:
        raise ValueError('Invalid raw capture ROI')
    x0,y0,x1,y1=[round(v*s) for v,s in zip(roi,[w,h,w,h])]
    if x0%2 or x1%2:raise ValueError('YUYV ROI boundaries must preserve chroma pairs')
    pixels=np.ascontiguousarray(raw[y0:y1,x0:x1])
    image=cv2.cvtColor(pixels,cv2.COLOR_YUV2BGR_YUY2)
    return image,[w,h],[x0,y0,x1,y1]


class LatestMailbox:
    def __init__(self, context=None):
        context = context or mp.get_context('spawn')
        self.pixels = context.RawArray('B', int(np.prod(MAX_SHAPE)))
        # seq, read-start, decoded, h,w,c, delivered-seq, terminal-failure
        self.meta = context.RawArray('d', 8)
        self.lock = context.Lock()

    @contextmanager
    def _locked(self):
        if not self.lock.acquire(timeout=.2):
            raise TimeoutError('Capture shared buffer lock timed out')
        try:
            yield
        finally:
            self.lock.release()

    def publish(self, image, started, decoded):
        if (image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3
                or not 0 < image.shape[0] <= MAX_SHAPE[0]
                or not 0 < image.shape[1] <= MAX_SHAPE[1]
                or not np.isfinite([started, decoded]).all() or not 0 < started <= decoded):
            raise ValueError('Invalid capture image or timestamp')
        with self._locked():
            if self.meta[7]:
                raise RuntimeError('Capture has failed')
            np.frombuffer(self.pixels, dtype=np.uint8)[:image.size] = image.reshape(-1)
            self.meta[0] += 1
            self.meta[1:6] = [started, decoded, *image.shape]

    def fail(self):
        with self._locked():
            self.meta[7] = 1

    def take(self, previous, now, maximum_age):
        with self._locked():
            if self.meta[7]:
                raise RuntimeError('Capture stopped or input failed')
            seq = int(self.meta[0])
            if seq <= previous:
                return None
            started, decoded = self.meta[1:3]
            if not 0 <= now - decoded <= maximum_age:
                raise TimeoutError('Newest capture frame is stale or has a future timestamp')
            shape = tuple(map(int, self.meta[3:6]))
            # One snapshot copy, backed by immutable bytes rather than a mutable
            # ndarray owner. Identity checks may safely reuse this frame's SHA.
            snapshot=np.frombuffer(self.pixels,dtype=np.uint8)[:int(np.prod(shape))].tobytes()
            image=np.frombuffer(snapshot,dtype=np.uint8).reshape(shape)
            self.meta[6] = seq
            return seq, started, decoded, image


def _capture_worker(mailbox, notices, stop, source, width, height, fps, fourcc, roi):
    import cv2
    cap = None
    try:
        cv2.setNumThreads(1)
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError('Cannot open capture device: ' + str(source))
        settings = [(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*fourcc)),
                    (cv2.CAP_PROP_FRAME_WIDTH,width),(cv2.CAP_PROP_FRAME_HEIGHT,height),
                    (cv2.CAP_PROP_FPS,fps),(cv2.CAP_PROP_BUFFERSIZE,1)]
        accepted = [bool(cap.set(key,value)) for key,value in settings]
        raw_roi=roi is not None
        if raw_roi:
            if fourcc!='YUYV' or not cap.set(cv2.CAP_PROP_CONVERT_RGB,0):
                raise RuntimeError('Physical raw YUYV capture is not supported')
        actual_size=[int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        if abs(cap.get(cv2.CAP_PROP_FPS)-fps)>.5:raise RuntimeError('Capture FPS differs from requested hardware mode')
        notices.put(dict(event='opened', source=source,requested=dict(width=width,height=height,fps=fps,fourcc=fourcc),
                         actual=dict(width=cap.get(cv2.CAP_PROP_FRAME_WIDTH),height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                                     fps=cap.get(cv2.CAP_PROP_FPS),fourcc=int(cap.get(cv2.CAP_PROP_FOURCC))),
                         set_property_accepted=accepted,driver_buffer_bound_not_proven=True,raw_YUYV_ROI=raw_roi,capture_roi=roi,
                         timestamp_clock='userspace_decode_completion_not_sensor_timestamp'))
        # Drain initialization frames before publishing. No old inference is cached.
        warmed = 0
        content_guard=ContentGuard()
        initial_empty = 0
        while not stop.is_set():
            started = time.monotonic()
            ok, image = cap.read()
            if not ok or image is None:
                if warmed==0 and initial_empty<10:
                    initial_empty+=1;stop.wait(.02);continue
                raise RuntimeError('Video input read failed')
            if raw_roi:
                if image.ndim==2 and image.shape==(1,actual_size[0]*actual_size[1]*2):
                    image=image.reshape(actual_size[1],actual_size[0],2)
                image,parent_size,roi_pixels=decode_yuyv_roi(image,roi)
                if parent_size!=actual_size:raise RuntimeError('Raw capture dimensions differ from negotiated physical mode')
            decoded=time.monotonic()
            warmed += 1
            if warmed <= 4:
                continue
            # Inspect every captured frame, not sparse inference samples of a loop.
            if content_guard.observe(image,decoded)['content_freeze_suspected']:
                continue
            mailbox.publish(image,started,decoded)
    except BaseException as error:
        mailbox.fail()
        try:
            notices.put_nowait(dict(event='failed',error=repr(error)))
        except queue.Full:
            pass
    finally:
        if cap is not None:
            cap.release()


class LatestCaptureReader:
    def __init__(self, source='/dev/video0', *, width=1280,height=720,fps=30,fourcc='YUYV',read_timeout=2.,roi=None):
        if (not str(source).startswith('/dev/video') or not 0 < width <= 1920
                or not 0 < height <= 1080 or not 1 <= fps <= 60
                or fourcc not in ['YUYV','MJPG'] or not .1 <= read_timeout <= 10):
            raise ValueError('Invalid physical-capture configuration')
        if roi is not None and (fourcc!='YUYV' or len(roi)!=4 or not 0<=roi[0]<roi[2]<=1 or not 0<=roi[1]<roi[3]<=1):
            raise ValueError('Invalid raw YUYV capture ROI')
        self.roi=roi
        self.context=mp.get_context('spawn')
        self.mailbox=LatestMailbox(self.context)
        self.notices=self.context.Queue(maxsize=4)
        self.stop=self.context.Event()
        self.process=self.context.Process(target=_capture_worker,
            args=(self.mailbox,self.notices,self.stop,str(source),width,height,fps,fourcc,roi),daemon=True)
        self.source=str(source);self.read_timeout=read_timeout;self.previous=0;self.closed=False;self.info=[]
        self.uniform_rejected=0
        self.process.start()

    def read(self,next_step=1):
        if self.closed:
            raise RuntimeError('Capture reader is closed')
        requested=time.monotonic()
        while time.monotonic()-requested < self.read_timeout:
            try:
                while True:self.info.append(self.notices.get_nowait())
            except queue.Empty:
                pass
            if not self.process.is_alive():
                raise RuntimeError('Capture process exited: '+repr(self.info))
            try:item=self.mailbox.take(self.previous,time.monotonic(),.5)
            except TimeoutError:
                self.stop.wait(.005);continue
            if item is not None:
                seq,started,decoded,image=item
                dropped=max(0,seq-self.previous-1);self.previous=seq
                # Uniform black/solid-color frames cannot establish a road observation.
                spatial=float(image[::16,::16].reshape(-1,3).std(axis=0).max())
                if spatial < .5:
                    self.uniform_rejected+=1
                    self.stop.wait(.005)
                    continue
                parent_size=roi_pixels=None
                if self.roi is not None:
                    mode=next((x['actual'] for x in self.info if x.get('event')=='opened'),None)
                    if mode is None:raise RuntimeError('No negotiated capture geometry')
                    parent_size=[int(mode['width']),int(mode['height'])]
                    roi_pixels=[round(v*s) for v,s in zip(self.roi,parent_size*2)]
                    if image.shape[:2]!=(roi_pixels[3]-roi_pixels[1],roi_pixels[2]-roi_pixels[0]):raise RuntimeError('Captured ROI dimensions mismatch')
                return LiveFrame(image,self.source,seq,0,False,started,decoded,
                    source_parent_size=parent_size,capture_roi_pixels=roi_pixels,
                    request_started=requested,delivered_at=time.monotonic(),dropped_before_delivery=dropped)
            self.stop.wait(.005)
        raise TimeoutError('No new input frame within capture timeout')

    def grab(self):
        # Producer drains continuously; consumer skips queued frames by construction.
        return not self.closed

    def close(self):
        if self.closed:return
        self.closed=True;self.stop.set();self.process.join(timeout=1.)
        if self.process.is_alive():
            self.process.terminate();self.process.join(timeout=1.)
        self.notices.cancel_join_thread();self.notices.close()
