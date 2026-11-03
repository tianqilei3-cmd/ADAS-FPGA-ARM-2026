"""Pre/post processing for the four-class YOLOv8 decoded ONNX output.

Contract: RGB uint8 NHWC input to RKNN (normalization is in RKNN config),
one decoded output [1, 4 + nc, anchors], or six raw YOLOv8 head outputs.
"""
from pathlib import Path

import cv2
import numpy as np


def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


def preprocess(image, size=640):
    height, width = image.shape[:2]
    target_h,target_w=(size,size) if isinstance(size,int) else size
    gain = min(target_h / height, target_w / width)
    resized_w, resized_h = round(width * gain), round(height * gain)
    pad_x, pad_y = (target_w - resized_w) / 2, (target_h - resized_h) / 2
    left, top = round(pad_x - 0.1), round(pad_y - 0.1)
    right, bottom = round(pad_x + 0.1), round(pad_y + 0.1)
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                               cv2.BORDER_CONSTANT, value=(114, 114, 114))
    rgb = np.ascontiguousarray(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))
    return rgb, {"gain": gain, "left": left, "top": top,
                 "height": height, "width": width}


def box_iou(box, boxes):
    intersection = np.maximum(0, np.minimum(box[2:], boxes[:, 2:]) -
                              np.maximum(box[:2], boxes[:, :2])).prod(axis=1)
    a = np.maximum(0, box[2:] - box[:2]).prod()
    b = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    return intersection / np.maximum(a + b - intersection, 1e-7)


def decoded_output(outputs, size=640, class_count=4, score_threshold=None):
    """Accept decoded FP output or six raw head outputs (box, sigmoid per scale).

    Keeping score and regression outputs separate avoids quantizing small
    probabilities together with large pixel coordinates. DFL runs in NumPy.
    A threshold skips regression decoding for anchors whose class scores cannot
    pass the downstream confidence filter; retained detections stay identical.
    """
    if len(outputs) == 1:
        return np.asarray(outputs[0], dtype=np.float32)
    if len(outputs) != 6:
        raise ValueError(f"Expected one decoded output or six head outputs, got {len(outputs)}")
    if score_threshold is not None and not 0 <= score_threshold <= 1:
        raise ValueError("score_threshold must be between zero and one")
    scales = []
    for branch in range(3):
        regression = np.asarray(outputs[2 * branch], dtype=np.float32)
        scores = np.asarray(outputs[2 * branch + 1], dtype=np.float32)
        if regression.ndim != 4 or regression.shape[0] != 1 or regression.shape[1] < 4 or regression.shape[1] % 4:
            raise ValueError(f"Unexpected regression shape {regression.shape}")
        _, channels, height, width = regression.shape
        if scores.shape != (1, class_count, height, width):
            raise ValueError(f"Unexpected class score shape {scores.shape}")
        if not np.isfinite(regression).all() or not np.isfinite(scores).all():
            raise ValueError("Head output contains NaN or Inf")
        bins = channels // 4
        logits = regression.reshape(1, 4, bins, height * width)
        scores = scores.reshape(1, class_count, height * width)
        if score_threshold is None:
            indices = np.arange(height * width)
        else:
            indices = np.flatnonzero(scores.max(axis=1)[0] >= score_threshold)
            if not len(indices):
                continue
            logits = logits[..., indices]
            scores = scores[..., indices]
        probabilities = np.exp(logits - logits.max(axis=2, keepdims=True))
        probabilities /= probabilities.sum(axis=2, keepdims=True)
        distances = (probabilities * np.arange(bins, dtype=np.float32)[None, None, :, None]).sum(axis=2)
        gx = (indices % width).astype(np.float32) + 0.5
        gy = (indices // width).astype(np.float32) + 0.5
        center = np.stack((gx, gy), axis=0)[None]
        target_h,target_w=(size,size) if isinstance(size,int) else size
        stride = np.asarray([target_w / width, target_h / height], dtype=np.float32)[None, :, None]
        minimum = (center - distances[:, :2]) * stride
        maximum = (center + distances[:, 2:]) * stride
        xywh = np.concatenate(((minimum + maximum) / 2, maximum - minimum), axis=1)
        scales.append(np.concatenate((xywh, scores), axis=1))
    return np.concatenate(scales, axis=2) if scales else np.empty((1, 4 + class_count, 0), dtype=np.float32)


def decode(output, geometry, names, conf=0.25, iou=0.45, max_det=100):
    raw = np.asarray(output, dtype=np.float32)
    if raw.ndim == 3 and raw.shape[0] == 1:
        raw = raw[0]
    channels = 4 + len(names)
    if raw.ndim != 2:
        raise ValueError(f"Expected decoded YOLO output, got {raw.shape}")
    if raw.shape[0] == channels:
        rows = raw.T
    elif raw.shape[1] == channels:
        rows = raw
    else:
        raise ValueError(f"Expected {channels} channels, got {raw.shape}")
    if not np.isfinite(rows).all():
        raise ValueError("Model output contains NaN or Inf")
    classes = rows[:, 4:].argmax(axis=1)
    scores = rows[np.arange(len(rows)), classes + 4]
    selected = scores >= conf
    xywh, scores, classes = rows[selected, :4], scores[selected], classes[selected]
    if not len(scores):
        return []
    boxes = np.concatenate((xywh[:, :2] - xywh[:, 2:] / 2,
                            xywh[:, :2] + xywh[:, 2:] / 2), axis=1)
    order = np.argsort(-scores, kind="stable")[:30000]
    keep = []
    while len(order) and len(keep) < max_det:
        current = order[0]
        keep.append(current)
        rest = order[1:]
        overlap = box_iou(boxes[current], boxes[rest])
        order = rest[(classes[rest] != classes[current]) | (overlap <= iou)]
    boxes = boxes[keep].copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - geometry["left"]) / geometry["gain"]
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - geometry["top"]) / geometry["gain"]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, geometry["width"])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, geometry["height"])
    return [{"class_id": int(classes[k]), "name": names[int(classes[k])],
             "confidence": float(scores[k]), "box": [float(x) for x in box]}
            for k, box in zip(keep, boxes) if box[2] > box[0] and box[3] > box[1]]


def match_detections(reference, candidate, min_iou=0.5):
    used, overlaps, score_errors = set(), [], []
    for ref in sorted(reference, key=lambda x: -x["confidence"]):
        choices = [(i, item) for i, item in enumerate(candidate)
                   if i not in used and item["class_id"] == ref["class_id"]]
        if not choices:
            continue
        ious = box_iou(np.asarray(ref["box"]), np.asarray([x[1]["box"] for x in choices]))
        best = int(ious.argmax())
        if ious[best] >= min_iou:
            index, item = choices[best]
            used.add(index)
            overlaps.append(float(ious[best]))
            score_errors.append(abs(ref["confidence"] - item["confidence"]))
    return {"reference_count": len(reference), "candidate_count": len(candidate),
            "matched": len(used), "reference_only": len(reference) - len(used),
            "candidate_only": len(candidate) - len(used),
            "matched_iou": overlaps, "matched_confidence_error": score_errors}


def draw(image, detections):
    canvas = image.copy()
    colors = [(0, 220, 220), (0, 200, 0), (220, 120, 0), (0, 120, 255)]
    for item in detections:
        x1, y1, x2, y2 = [round(x) for x in item["box"]]
        color = colors[item["class_id"] % len(colors)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, f'{item["name"]} {item["confidence"]:.2f}',
                    (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return canvas
