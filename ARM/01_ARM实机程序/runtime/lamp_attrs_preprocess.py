"""Exact exported preprocessing; see metadata source hashes."""
import cv2
import numpy as np

def input_rgb(image, size=128):
    height, width = image.shape[:2]
    gain = size / max(height, width)
    w, h = (max(1, round(width * gain)), max(1, round(height * gain)))
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    left, top = ((size - w) // 2, (size - h) // 2)
    image = cv2.copyMakeBorder(image, top, size - h - top, left, size - w - left, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def color_evidence(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    eligible = (s >= 80) & (v >= 80)
    masks = [eligible & ((h <= 12) | (h >= 165)), eligible & (h >= 15) & (h <= 34), eligible & (h >= 35) & (h <= 95)]
    counts = np.asarray([int(x.sum()) for x in masks])
    order = np.argsort(-counts)
    first, second = (int(order[0]), int(order[1]))
    strong = counts[first] >= max(4, image.shape[0] * image.shape[1] * 0.003)
    strong = strong and counts[first] >= 3 * max(1, counts[second])
    return {'counts': counts.tolist(), 'state': first if strong else None, 'scope': 'Uncalibrated development evidence, not a verified traffic signal decision'}

def preprocess(image, size=128):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return input_rgb(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), size)

def direction_input(image, size=128):
    """Normalize luminous shape brightness separately from the full housing."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    state = color_evidence(image)['state']
    if state is None:
        return (preprocess(image, size), False)
    hue = [(h <= 12) | (h >= 165), (h >= 15) & (h <= 34), (h >= 35) & (h <= 95)][state]
    mask = (hue & (s >= 80) & (v >= 80)).astype(np.uint8)
    joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    if n <= 1:
        return (preprocess(image, size), False)
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, height, area = map(int, stats[component])
    if area < 4:
        return (preprocess(image, size), False)
    pad = max(2, round(max(w, height) * 0.2))
    left, top = (max(0, x - pad), max(0, y - pad))
    right, bottom = (min(image.shape[1], x + w + pad), min(image.shape[0], y + height + pad))
    shape = (v * (labels == component).astype(np.uint8))[top:bottom, left:right]
    return (input_rgb(cv2.cvtColor(shape, cv2.COLOR_GRAY2BGR), size), True)

def native_descriptors(image):
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, s, v = cv2.split(hsv)
    state = color_evidence(image)['state']
    select = [(hue <= 12) | (hue >= 165), (hue >= 15) & (hue <= 34), (hue >= 35) & (hue <= 95)]
    mask = ((s >= 80) & (v >= 80) & (select[state] if state is not None else False)).astype(np.uint8)
    joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    bw, bh, extent = (0, 0, 0.0)
    if n > 1:
        index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        _, _, bw, bh, _ = stats[index]
        extent = float((mask & (labels == index)).sum()) / max(1, int(bw * bh))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    filled = float(((joined > 0) & (mask == 0)).sum()) / max(1, int(mask.sum()))
    return np.asarray([np.log1p(w), np.log1p(h), np.log1p(mask.sum()), float(mask.mean()), filled, float(s.mean() / 255), float(v.mean() / 255), float(v.std() / 255), float(cv2.Laplacian(gray, cv2.CV_32F).var()), np.log1p(bw), np.log1p(bh), extent], dtype=np.float32)
