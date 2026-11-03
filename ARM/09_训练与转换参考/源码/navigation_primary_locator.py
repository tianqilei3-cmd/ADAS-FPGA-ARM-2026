"""Experimental full-frame primary navigation locator; NumPy/OpenCV runtime.

Connected-component proposals are class agnostic. A learned context classifier
separates primary instructions from secondary prompts, lane rows, map markers,
and digits. Glyph direction uses the retained, independently trained HOG head.
This emits route intent only, never an immediate-turn/control command.
"""
from pathlib import Path
import cv2
import numpy as np

from navigation_hog_classifier import NavigationHOGClassifier, ShapeHOG

CONTEXT_NAMES = ['primary', 'secondary', 'lane_row', 'map_marker', 'digits']
CONTEXT_SIZE = (160, 96)
CONTEXT_THRESHOLD = .90
CONTEXT_MARGIN = .25


def overlap(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2-x1)*max(0, y2-y1)
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union > 0 else 0.


def extract_context(image, box):
    x1, y1, x2, y2 = map(float, box)
    unit = max(x2-x1, y2-y1)
    # Coordinate frame follows the candidate, not an assumed screen location.
    sx, sy = x1-.40*unit, y1-.40*unit
    ex, ey = x1+5.0*unit, y1+2.84*unit
    transform = np.array([[(ex-sx)/160, 0, sx], [0, (ey-sy)/96, sy]], np.float32)
    patch = cv2.warpAffine(image, transform, CONTEXT_SIZE,
                           flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_REPLICATE)
    return patch


def context_features(patch):
    if patch.shape != (96, 160, 3) or patch.dtype != np.uint8:
        raise ValueError('Expected uint8 96x160 BGR context')
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    lo, hi = np.percentile(gray, [5, 95])
    normal = np.clip((gray.astype(np.float32)-lo)*255/max(20., hi-lo), 0, 255).astype(np.uint8)
    # Standardize day/night polarity from the immediate arrow background.
    surrounding = np.concatenate([normal[3:10, 3:45].ravel(), normal[40:46, 3:45].ravel()])
    if np.median(surrounding) < 127:
        normal = 255-normal
    hog = ShapeHOG().compute(cv2.resize(normal, (96, 96)))
    spatial = cv2.resize((255-normal).astype(np.float32)/255., (20, 12), interpolation=cv2.INTER_AREA).ravel()
    return np.concatenate([hog, spatial]).astype(np.float32)


def rule_context_score(patch):
    """Frozen baseline: several dark components to the right, quieter below.

    No trained context weights, absolute screen positions, or video timestamps.
    """
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    background = float(np.median(gray[3:12, 4:130]))
    mask = (gray < background-22).astype(np.uint8)
    right = mask[9:39, 41:132]
    n, _, stats, _ = cv2.connectedComponentsWithStats(right, 8)
    count = sum(1 for x, y, w, h, area in stats[1:] if h >= 8 and w >= 2 and area >= 12)
    ink_right = float(right.mean())
    return float(min(count/3., 1.) * min(ink_right/.12, 1.))


class PrimaryNavigationLocator:
    def __init__(self, glyph_weights, context_weights=None):
        self.glyph = NavigationHOGClassifier(glyph_weights)
        self.weights = self.bias = None
        if context_weights is not None:
            with np.load(Path(context_weights), allow_pickle=False) as data:
                if data['names'].tolist() != CONTEXT_NAMES:
                    raise ValueError('Unexpected context class order')
                self.weights, self.bias = data['weights'].copy(), data['bias'].copy()
            if self.weights.shape != (5, 2004) or self.bias.shape != (5,):
                raise ValueError('Invalid context weights')

    def proposals(self, image):
        h, w = image.shape[:2]
        scale = min(1., 1600/max(h, w))
        small = cv2.resize(image, (round(w*scale), round(h*scale))) if scale < 1 else image
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        boxes = []
        for inverted in [False, True]:
            source = 255-gray if inverted else gray
            for block, bias in [(61, 9), (31, 5)]:
                mask = cv2.adaptiveThreshold(source, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, block, bias)
                n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
                for x, y, cw, ch, area in stats[1:]:
                    if not (10 <= ch <= 150 and 7 <= cw <= 150 and .18 <= cw/ch <= 1.65):
                        continue
                    if not (.10 <= area/(cw*ch) <= .78) or area < 24:
                        continue
                    box = [max(0, int((x-3)/scale)), max(0, int((y-3)/scale)),
                           min(w, int((x+cw+3)/scale)), min(h, int((y+ch+3)/scale))]
                    if any(overlap(box, prior) > .70 for prior in boxes):
                        continue
                    boxes.append(box)
        # Keep bounded execution; overflow refuses output rather than silently
        # selecting from a frame-coordinate biased truncated proposal list.
        if len(boxes) > 1600:
            return [], dict(proposal_overflow=True, raw_proposals=len(boxes))
        crops = [image[b[1]:b[3], b[0]:b[2]] for b in boxes]
        glyphs = self.glyph.predict(crops)
        rows = []
        for box, glyph in zip(boxes, glyphs):
            if glyph.get('confidence', 0) < .80 or glyph.get('margin', 0) < .25:
                continue
            # Keep unsupported glyphs for truthful unknown-primary diagnostics.
            patch = extract_context(image, box)
            row = dict(box=box, glyph=glyph, rule_score=rule_context_score(patch))
            if self.weights is not None:
                logits = self.weights @ context_features(patch) + self.bias
                p = np.exp(logits-logits.max()); p /= p.sum()
                row['context_probabilities'] = p.tolist()
                row['context_score'] = float(p[0])
                row['context_margin'] = float(p[0]-max(p[1:]))
            rows.append(row)
        return rows, dict(proposal_overflow=False, raw_proposals=len(boxes))

    @staticmethod
    def select(rows, method='learned'):
        if method not in ['learned', 'rule']:
            raise ValueError('method must be learned or rule')
        key = 'context_score' if method == 'learned' else 'rule_score'
        threshold = CONTEXT_THRESHOLD if method == 'learned' else .80
        candidates = [r for r in rows if r.get(key, 0) >= threshold and
                      (method != 'learned' or r.get('context_margin', 0) >= CONTEXT_MARGIN)]
        candidates.sort(key=lambda r: r[key], reverse=True)
        # Overlapping component variants count as the same candidate.
        distinct = []
        for row in candidates:
            if not any(overlap(row['box'], old['box']) > .35 for old in distinct):
                distinct.append(row)
        if not distinct:
            return dict(direction='unknown', direction_valid=False, reason='no_primary_candidate',
                        primary_box=None, immediate_turn_authorized=False)
        best = distinct[0]
        # Multiple convincing primary cards in an arbitrary image are ambiguous.
        if len(distinct) > 1 and distinct[1][key] >= best[key]-.03:
            return dict(direction='unknown', direction_valid=False, reason='ambiguous_primary_candidates',
                        primary_box=None, immediate_turn_authorized=False)
        glyph = best['glyph']
        return dict(direction=glyph['direction'], direction_valid=glyph['direction_valid'],
                    reason='experimental_primary_context', primary_box=best['box'],
                    context_score=best[key], glyph_confidence=glyph.get('confidence', 0),
                    immediate_turn_authorized=False)

    def predict(self, image, method='learned'):
        if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Expected uint8 BGR full frame')
        if min(image.shape[:2]) < 64:
            raise ValueError('Full frame must be at least 64 pixels per side')
        if method == 'learned' and self.weights is None:
            raise ValueError('Learned method requires context weights')
        rows, metadata = self.proposals(image)
        result = self.select(rows, method)
        return dict(**result, **metadata, candidates=rows, automatic_roi_attempted=True,
                    distance_reading=False, selected_for_control=False)
