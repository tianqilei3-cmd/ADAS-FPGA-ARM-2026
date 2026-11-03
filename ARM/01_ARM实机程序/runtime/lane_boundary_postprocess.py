"""Current-frame segmentation regions only; no ego-lane assignment or control."""
import cv2
import numpy as np


def filter_regions(probabilities, source_shape, occluders, minimum_probability=.50):
    p = np.asarray(probabilities)
    if p.ndim != 3 or p.shape[0] != 3 or not np.isfinite(p).all():
        raise ValueError('Expected finite3xHxW boundary probabilities')
    roundoff = 4 * np.finfo(np.float32).eps
    if p.min() < -roundoff or p.max() > 1 + roundoff or not np.allclose(p.sum(0), 1., atol=.001):
        raise ValueError('Expected normalized softmax probabilities')
    p = np.clip(p, 0., 1.)
    if not 0 <= minimum_probability <= 1:
        raise ValueError('Invalid segmentation threshold')
    source_h, source_w = source_shape[:2]
    h, w = p.shape[1:]
    classes = p.argmax(0).astype(np.uint8)
    classes[p.max(0) < minimum_probability] = 0
    # Generic image exclusion, not a claimed calibrated physical lane ROI.
    classes[:round(h * .40)] = 0
    classes[round(h * .94):] = 0
    for obj in occluders:
        if obj.get('name') not in ('vehicle', 'person', 'two_wheeler') or obj.get('confidence', 0) < .35:
            continue
        a, b, c, d = obj['box']
        if not np.isfinite([a, b, c, d]).all() or c <= a or d <= b:
            raise ValueError('Invalid current occluder')
        if c <= 0 or d <= 0 or a >= source_w or b >= source_h:
            continue
        a, b = max(0, int(a * w / source_w) - 1), max(0, int(b * h / source_h) - 1)
        c, d = min(w, round(c * w / source_w) + 2), min(h, round(d * h / source_h) + 2)
        classes[b:d, a:c] = 0
    accepted = np.zeros((h, w), np.uint8)
    regions = []
    for label in (1, 2):
        count, components, stats, _ = cv2.connectedComponentsWithStats((classes == label).astype(np.uint8), 8)
        for index in range(1, count):
            x, y, cw, ch, area = stats[index]
            if area < 20 or ch < 12 or area / ch > 24:
                continue
            chosen = components == index
            accepted[chosen] = label
            contours, _ = cv2.findContours(chosen.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
            if len(contour) > 200:
                contour = cv2.approxPolyDP(contour, 1., True).reshape(-1, 2)
            regions.append(dict(style='solid' if label == 1 else 'dashed',
                                mean_probability=float(p[label][chosen].mean()),
                                box=[float(x * source_w / w), float(y * source_h / h),
                                     float((x + cw) * source_w / w), float((y + ch) * source_h / h)],
                                contour=(contour * [source_w / w, source_h / h]).tolist(),
                                work_pixels=int(area), ego_lane_association_verified=False))
    return accepted, regions
