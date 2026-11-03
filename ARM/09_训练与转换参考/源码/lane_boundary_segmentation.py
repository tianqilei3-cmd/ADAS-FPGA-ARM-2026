"""Typed boundary-corridor segmentation, not segmentation of painted dashes.

Targets follow original BDD parallel-lane style. Dashed gaps belong to the same
boundary corridor; mask continuity must NEVER be used to infer solid/dashed.
Old-format cubic paths follow BDD's Matplotlib Path conversion convention.
"""
import copy
from pathlib import Path

import cv2
from matplotlib.path import Path as MplPath
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

NAMES = ['background', 'solid_boundary', 'dashed_boundary']
ALLOWED = {'lane/single white', 'lane/double white', 'lane/single yellow', 'lane/double yellow'}


def decode_old_poly2d(vertices):
    """Flatten old [x,y,L/C] paths exactly through the official Path convention."""
    if len(vertices) < 2 or any(len(v) != 3 or v[2] not in ('L', 'C') for v in vertices):
        raise ValueError('Unsupported old BDD path')
    points = np.asarray([v[:2] for v in vertices], dtype=float)
    if not np.isfinite(points).all():
        raise ValueError('Nonfinite BDD vertices')
    codes = [MplPath.LINETO if v[2] == 'L' else MplPath.CURVE4 for v in vertices]
    codes[0] = MplPath.MOVETO
    path = MplPath(points, codes)
    flattened = []
    for xy, code in path.iter_segments(curves=False, simplify=False, remove_nans=False):
        if code not in (MplPath.MOVETO, MplPath.LINETO) or len(xy) != 2:
            raise ValueError('Unsupported open path segment')
        flattened.append(xy)
    if len(flattened) < 2:
        raise ValueError('Empty BDD path')
    return np.asarray(flattened, dtype=np.float32)


def lane_class(obj):
    attrs = obj.get('attributes', {})
    if obj.get('category') in ALLOWED and attrs.get('direction') == 'parallel':
        return {'solid': 1, 'dashed': 2}.get(attrs.get('style'))
    return None


def render_target(objects, source_shape, size=(512, 288), width=5):
    """Return uint8 {0,1,2,255}; excluded lanes/occluders are ignored, not gaps."""
    source_h, source_w = source_shape[:2]
    w, h = size
    positive = [np.zeros((h, w), np.uint8) for _ in range(2)]
    ignored = np.zeros((h, w), np.uint8)
    counts = dict(solid=0, dashed=0, excluded_lanes=0, invalid_paths=0, cubic_paths=0)
    for obj in objects:
        if not obj.get('category', '').startswith('lane/') or not obj.get('poly2d'):
            continue
        try:
            points = decode_old_poly2d(obj['poly2d'])
        except ValueError:
            counts['invalid_paths'] += 1
            continue
        if (np.abs(points) > 4 * max(source_w, source_h)).any():
            counts['invalid_paths'] += 1
            continue
        xy = np.round(points * [w / source_w, h / source_h]).astype(np.int32)
        label = lane_class(obj)
        if label is None:
            cv2.polylines(ignored, [xy], False, 1, width + 4, cv2.LINE_8)
            counts['excluded_lanes'] += 1
        else:
            cv2.polylines(positive[label - 1], [xy], False, 1, width, cv2.LINE_8)
            counts['solid' if label == 1 else 'dashed'] += 1
            counts['cubic_paths'] += any(v[2] == 'C' for v in obj['poly2d'])
    for obj in objects:
        if obj.get('category') not in ('car', 'truck', 'bus', 'person', 'rider', 'bike', 'motor', 'train') or not obj.get('box2d'):
            continue
        box = obj['box2d']
        a, b, c, d = [float(box[k]) for k in ('x1', 'y1', 'x2', 'y2')]
        if not np.isfinite([a, b, c, d]).all() or c <= a or d <= b:
            raise ValueError('Invalid annotated occluder box')
        a, b, c, d = round(a * w / source_w), round(b * h / source_h), round(c * w / source_w), round(d * h / source_h)
        cv2.rectangle(ignored, (a, b), (c, d), 1, -1)
    target = np.zeros((h, w), np.uint8)
    target[positive[0] > 0] = 1
    target[positive[1] > 0] = 2
    target[(ignored > 0) | ((positive[0] > 0) & (positive[1] > 0))] = 255
    return target, counts


class BoundarySegmenter(nn.Module):
    """Local pretrained YOLOv8n encoder + small semantic decoder."""
    def __init__(self, initialization, freeze_encoder=True):
        super().__init__()
        from ultralytics import YOLO
        source = YOLO(str(initialization)).model.model
        if any(layer.f != -1 for layer in source[:10]):
            raise ValueError('Unexpected YOLO encoder graph')
        self.encoder = nn.ModuleList([copy.deepcopy(layer).float() for layer in source[:10]])
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            self.encoder.requires_grad_(False)
        self.lateral = nn.ModuleList([nn.Conv2d(c, 32, 1) for c in (32, 64, 128, 256)])
        self.refine = nn.ModuleList([nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()) for _ in range(3)])
        self.head = nn.Sequential(nn.Conv2d(32, 24, 3, padding=1), nn.ReLU(), nn.Conv2d(24, 3, 1))

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward(self, images):
        features = []
        x = images
        for index, layer in enumerate(self.encoder):
            x = layer(x)
            if index in (2, 4, 6, 9):
                features.append(x)
        x = self.lateral[3](features[3])
        for slot, level in enumerate((2, 1, 0)):
            x = F.interpolate(x, size=features[level].shape[-2:], mode='bilinear', align_corners=False)
            x = self.refine[slot](x + self.lateral[level](features[level]))
        return F.interpolate(self.head(x), size=images.shape[-2:], mode='bilinear', align_corners=False)


def boundary_loss(logits, target):
    valid = target != 255
    if not valid.any():
        return logits.sum() * 0
    cross_entropy = F.cross_entropy(logits, target, weight=logits.new_tensor([.15, 1., 1.]), ignore_index=255)
    probability = logits.softmax(1)
    dice = []
    for label in (1, 2):
        actual = (target == label).float()
        prediction = probability[:, label] * valid
        dice.append(1 - (2 * (prediction * actual).sum() + 1) / (prediction.sum() + actual.sum() + 1))
    return cross_entropy + sum(dice) / 2


def image_tensor(images):
    array = np.stack([image[:, :, ::-1] for image in images]).copy()
    return torch.from_numpy(array).permute(0, 3, 1, 2).float() / 255


class BoundaryPredictor:
    def __init__(self, weights):
        checkpoint = torch.load(weights, map_location='cpu', weights_only=False)
        if checkpoint['names'] != NAMES or checkpoint['purpose'] != 'typed_boundary_corridors':
            raise ValueError('Not a boundary-corridor model')
        self.size = tuple(checkpoint['plan']['size'])
        self.model = BoundarySegmenter(checkpoint['plan']['initialization']).eval()
        self.model.load_state_dict(checkpoint['model'])

    @torch.inference_mode()
    def predict(self, image):
        resized = cv2.resize(image, self.size)
        return self.model(image_tensor([resized])).softmax(1)[0].numpy()
