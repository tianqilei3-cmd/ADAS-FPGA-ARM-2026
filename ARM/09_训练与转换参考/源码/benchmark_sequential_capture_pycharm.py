"""Compare exact BGR hashes against the original per-frame seek on eight videos."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import cv2

from official_light_config import ROOT, VIDEOS
from sequential_file_capture import SequentialFileCapture

OUT = ROOT / 'runs/sequential_decode_20261022'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(image):
    h = hashlib.sha256()
    h.update(str(image.shape).encode())
    h.update(image.tobytes())
    return h.hexdigest()


def main():
    cv2.setNumThreads(1)
    folder = OUT / ('benchmark_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(parents=True)
    records = []
    for source in sorted(VIDEOS.glob('video_20260929_*.mp4')):
        reference = cv2.VideoCapture(str(source))
        optimized = SequentialFileCapture(cv2.VideoCapture(str(source)))
        if not reference.isOpened() or not optimized.isOpened():
            raise RuntimeError('Cannot open ' + str(source))
        total = int(reference.get(cv2.CAP_PROP_FRAME_COUNT))
        base = min(600, max(0, total // 2 - 84))
        offsets = [0, 1, 4, 7, 37, 38, 36, 39, 80, 83]
        positions = [base + i for i in offsets if base + i < total]
        paired = []
        try:
            for index, frame_id in enumerate(positions):
                images = {}
                durations = {}
                order = ['reference', 'optimized']
                if index % 2:
                    order.reverse()
                for name in order:
                    cap = reference if name == 'reference' else optimized
                    started = time.perf_counter()
                    if not cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id):
                        raise RuntimeError('Cannot select comparison frame')
                    ok, image = cap.read()
                    durations[name] = (time.perf_counter() - started) * 1000
                    if not ok:
                        raise RuntimeError('Cannot read comparison frame')
                    images[name] = identity(image)
                paired.append(dict(frame_id=frame_id, durations_ms=durations,
                                   bgr_hashes=images, identical=images['reference'] == images['optimized'],
                                   optimized_read=optimized.records[-1]))
                if not paired[-1]['identical']:
                    raise RuntimeError('BGR pixels differ for ' + source.name + ' frame ' + str(frame_id))
        finally:
            reference.release()
            optimized.release()
        records.append(dict(source=str(source), size_bytes=source.stat().st_size,
                            frame_count=total, pairs=paired, random_seeks=optimized.random_seeks,
                            original_seek_calls=len(positions)))
        print('DECODE_PAIRED', source.name, len(paired), 'identical; seeks', optimized.random_seeks, flush=True)
    # Short forward steps are the intended optimization; long/backward seeks are
    # retained and reported separately rather than hiding their cost.
    forward = [row for record in records for row in record['pairs'][1:]
               if row['optimized_read']['mode'] == 'sequential']
    means = {name: sum(r['durations_ms'][name] for r in forward) / len(forward)
             for name in ('reference', 'optimized')}
    result = dict(records=records, compared_frames=sum(len(r['pairs']) for r in records),
                  all_bgr_identical=True, short_forward_mean_ms=means,
                  accepted=len(records) >= 8 and means['optimized'] < means['reference'],
                  source_hashes={n: digest(ROOT / n) for n in (
                      'sequential_file_capture.py', 'benchmark_sequential_capture_pycharm.py')},
                  hardware_accessed=False, inference_accuracy_measured=False)
    path = folder / 'results.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'latest_benchmark.json').write_text(json.dumps(dict(path=str(path), sha256=digest(path)), indent=2), encoding='utf-8')
    print('DECODE_RESULT', result['compared_frames'], means, result['accepted'])


if __name__ == '__main__':
    main()
