"""Color-independent trained approach-role scores, not physical lane permission."""
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from light_roi_common import crop_head, input_rgb, logits_of
from review_navigation_temporal_pycharm import digest
from signal_emission_guard import EmissionGuardClassifier

FORMAT = "signal_gray_approach_relevance_v1"
DIMENSIONS = 1280 * 3 + 10


def validate_image_box(image, box):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("uint8 BGR scene required")
    if len(box) != 4 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in box):
        raise ValueError("Finite native xyxy box required")
    a, b, c, d = box
    if not 0 <= a < c <= image.shape[1] or not 0 <= b < d <= image.shape[0]:
        raise ValueError("Positive in-scene box required")


def inputs(image, box):
    validate_image_box(image, box)
    a, b, c, d = box
    h, w = image.shape[:2]
    cx, cy = (a + c) / 2, (b + d) / 2
    wide_w, wide_h = max(96., 6 * (c - a)), max(96., 6 * (d - b))
    left, top = max(0, math.floor(cx - wide_w / 2)), max(0, math.floor(cy - wide_h / 2))
    right, bottom = min(w, math.ceil(cx + wide_w / 2)), min(h, math.ceil(cy + wide_h / 2))
    crops = [crop_head(image, box), image[top:bottom, left:right], image]
    gray = [input_rgb(cv2.cvtColor(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR), 128) for crop in crops]
    geometry = [a / w, b / h, c / w, d / h, cx / w, cy / h, (c - a) / w, (d - b) / h,
                math.log((c - a) / (d - b)), (c - a) * (d - b) / (w * h)]
    return gray, np.asarray(geometry, np.float32)


def make_head():
    return torch.nn.Sequential(torch.nn.Linear(DIMENSIONS, 32), torch.nn.ReLU(),
                               torch.nn.Dropout(.2), torch.nn.Linear(32, 2))


class ApproachFeatures:
    def __init__(self, parent):
        self.trunk = EmissionGuardClassifier(parent).model.eval().requires_grad_(False)
        self.trunk.model[-1].linear = torch.nn.Identity()

    def extract(self, examples, batch=24):
        if not examples:
            return torch.empty((0, DIMENSIONS))
        outputs = []
        for first in range(0, len(examples), batch):
            prepared = [inputs(image, box) for image, box in examples[first:first + batch]]
            n = len(prepared)
            arrays = [row[0][view] for view in range(3) for row in prepared]
            tensor = torch.from_numpy(np.stack(arrays).transpose(0, 3, 1, 2).copy()).float() / 255
            with torch.inference_mode():
                features = logits_of(self.trunk(tensor)).clone()
            geometry = torch.from_numpy(np.stack([row[1] for row in prepared]))
            outputs.append(torch.cat([features[:n], features[n:2 * n], features[2 * n:], geometry], dim=1))
        return torch.cat(outputs)


class ApproachRelevance:
    def __init__(self, weights):
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        if checkpoint.get("format") != FORMAT:
            raise ValueError("Unexpected approach checkpoint")
        parent = Path(checkpoint["parent"])
        if digest(parent) != checkpoint["parent_sha256"]:
            raise ValueError("Frozen feature parent changed")
        self.features = ApproachFeatures(parent)
        self.head = make_head().eval().requires_grad_(False)
        self.head.load_state_dict(checkpoint["head_state"])
        self.mean = checkpoint["mean"].clone()
        self.std = checkpoint["std"].clone()
        self.threshold = float(checkpoint["threshold"])
        self.checkpoint = checkpoint

    def predict(self, image, boxes):
        if not boxes:
            return []
        x = self.features.extract([(image, box) for box in boxes])
        with torch.inference_mode():
            probabilities = self.head((x - self.mean) / self.std).softmax(1)[:, 1].tolist()
        return [dict(approach_probability=p, predicted_relevant=p >= self.threshold,
                     physical_lane_association_verified=False, color_changed=False) for p in probabilities]
