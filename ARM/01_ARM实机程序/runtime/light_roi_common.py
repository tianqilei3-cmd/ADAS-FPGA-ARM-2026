"""Fixed color-classification preprocessing; class IDs match the lamp detector."""
from pathlib import Path
import cv2
import numpy as np

NAMES = ['red', 'yellow', 'green', 'off']
ROOT = Path(__file__).resolve().parent


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot decode {path}')
    return image


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f'Cannot encode {path}')
    data.tofile(str(path))


def crop_head(image, box, padding=.08):
    x1, y1, x2, y2 = map(float, box)
    dx, dy = (x2-x1)*padding, (y2-y1)*padding
    left, top = max(0, int(np.floor(x1-dx))), max(0, int(np.floor(y1-dy)))
    right, bottom = min(image.shape[1], int(np.ceil(x2+dx))), min(image.shape[0], int(np.ceil(y2+dy)))
    if right-left < 2 or bottom-top < 2:
        raise ValueError(f'Too small or invalid lamp box: {box}')
    return image[top:bottom, left:right].copy()


def input_rgb(image, size=128):
    height, width = image.shape[:2]
    gain = size/max(height, width)
    w, h = max(1, round(width*gain)), max(1, round(height*gain))
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    left, top = (size-w)//2, (size-h)//2
    image = cv2.copyMakeBorder(image, top, size-h-top, left, size-w-left,
                              cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def make_model():
    from ultralytics import YOLO
    from ultralytics.nn.tasks import ClassificationModel
    model = YOLO(str(ROOT/'yolov8n-cls.pt')).model.float()
    ClassificationModel.reshape_outputs(model, len(NAMES))
    model.names = dict(enumerate(NAMES))
    model.yaml['nc'] = len(NAMES)
    model.requires_grad_(True)
    return model


def logits_of(output):
    return output[-1] if isinstance(output, (list, tuple)) else output


def color_evidence(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    eligible = (s >= 80) & (v >= 80)
    masks = [eligible & ((h <= 12) | (h >= 165)),
             eligible & (h >= 15) & (h <= 34), eligible & (h >= 35) & (h <= 95)]
    counts = np.asarray([int(x.sum()) for x in masks])
    order = np.argsort(-counts)
    first, second = int(order[0]), int(order[1])
    # Initial development rule, retained unchanged across the first training experiment.
    strong = counts[first] >= max(4, image.shape[0]*image.shape[1]*.003)
    strong = strong and counts[first] >= 3*max(1, counts[second])
    return {'counts': counts.tolist(), 'state': first if strong else None,
            'scope': 'Uncalibrated development evidence, not a verified traffic signal decision'}


def classify_policy(probabilities, image, threshold=.70, margin=.20, use_hsv=True):
    order = np.argsort(-np.asarray(probabilities))
    first, second = int(order[0]), int(order[1])
    if probabilities[first] < threshold or probabilities[first]-probabilities[second] < margin:
        return 4, 'low_confidence'
    evidence = color_evidence(image)
    if use_hsv and evidence['state'] is not None and first != evidence['state']:
        return 4, 'color_conflict'
    if use_hsv and first == 2 and evidence['state'] != 2:
        return 4, 'green_without_color_support'
    return first, 'accepted'
