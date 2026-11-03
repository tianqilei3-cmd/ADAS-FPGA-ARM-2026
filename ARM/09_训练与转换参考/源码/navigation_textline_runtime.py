"""Experimental distance-line detector + existing explicit-unit OCR.

Inference takes pixels and a predicted navigation card, never reference labels.
This module does not authorize motion and does not replace the default reader.
"""
import math

import cv2

from navigation_distance_ocr import (
    MobileTextRecognizer, _distance_literal, numeric_hint, parse_distance,
)


def unknown(reason, box=None):
    return dict(distance_m=None, distance_valid=False, unit=None,
                distance_box=box, numeric_value_hint=None, numeric_hint_unit=None,
                numeric_hint_for_control=False, immediate_turn_authorized=False,
                reason=reason)


def resolve_readings(readings, box):
    """Same explicit-unit/confidence policy as the frozen OCR baseline."""
    result = unknown('unreadable_or_missing_explicit_distance_unit', box)
    result['ocr_views'] = readings
    hints = {v for r in readings if (v := numeric_hint(r)) is not None}
    if len(hints) == 1:
        result['numeric_value_hint'] = next(iter(hints))
    accepted, literals = [], []
    for reading in readings:
        if reading.get('confidence', 0) < .85 or reading.get('minimum_character_confidence', 0) < .50:
            continue
        literal = _distance_literal(reading.get('text'))
        parsed = parse_distance(reading.get('text'))
        if literal is not None:
            literals.append(literal['distance_m'])
        if parsed is not None:
            accepted.append((parsed, reading))
    if len(set(literals)) > 1:
        result['reason'] = 'conflicting_ocr_views'
        return result
    if not accepted:
        return result
    parsed, best = max(accepted, key=lambda pair: pair[1]['confidence'])
    for reading in readings:
        hint = numeric_hint(reading)
        if parse_distance(reading.get('text')) is None and hint is not None and hint != parsed['observed_value']:
            result['reason'] = 'conflicting_numeric_views'
            return result
    result.update(parsed, ocr=best, distance_valid=True, reason='explicit_predicted_line_distance_read')
    return result


def iou(a, b):
    overlap = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    area = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1])
    return overlap/max(1e-9, area-overlap)


def choose_line(rows, width, height, confidence=.50):
    valid = []
    for row in rows:
        box, score = row.get('box'), row.get('confidence', 0)
        if (not box or len(box) != 4 or not math.isfinite(score) or score < confidence
                or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box)):
            continue
        a,b,c,d = box
        if not (0 <= a < c <= width and 0 <= b < d <= height and c-a >= 10 and d-b >= 6):
            continue
        valid.append(row)
    valid.sort(key=lambda r:r['confidence'], reverse=True)
    if not valid:
        return None, 'no_confident_primary_text_line'
    if any(iou(valid[0]['box'], r['box']) < .30 for r in valid[1:]):
        return None, 'ambiguous_primary_text_lines'
    return valid[0], 'primary_text_line_detected'


def fuse_distances(baseline, localized):
    """Any disagreement between confident readings causes rejection."""
    old, new = baseline.get('distance_valid', False), localized.get('distance_valid', False)
    if old and new and baseline['distance_m'] != localized['distance_m']:
        return dict(unknown('baseline_textline_distance_conflict'), distance_source='conflict')
    chosen = baseline if old else localized
    other = localized if old else baseline
    if chosen.get('distance_valid') and other.get('numeric_value_hint') is not None:
        if other['numeric_value_hint'] != chosen.get('observed_value'):
            return dict(unknown('baseline_textline_numeric_conflict'), distance_source='conflict')
    return dict(chosen, distance_source='baseline' if old else 'experimental_textline',
                immediate_turn_authorized=False)


class NavigationTextlineReader:
    def __init__(self, weights, ocr_weights, detector=None, ocr=None):
        if detector is None:
            from ultralytics import YOLO
            detector = YOLO(str(weights))
        self.detector = detector
        self.ocr = ocr if ocr is not None else MobileTextRecognizer(ocr_weights)

    def read(self, image, nav):
        box = nav.get('card_box')
        if (not box or len(box) != 4 or not all(isinstance(v, (int,float)) and math.isfinite(v) for v in box)):
            return unknown('no_predicted_navigation_card')
        a,b,c,d = map(int, box)
        if not (0 <= a < c <= image.shape[1] and 0 <= b < d <= image.shape[0]):
            return unknown('invalid_predicted_navigation_card')
        card = image[b:d, a:c]
        prediction = self.detector.predict(card, imgsz=320, conf=.50, iou=.50, max_det=5,
                                           device='cpu', verbose=False)[0]
        rows = [dict(box=row[:4].tolist(), confidence=float(row[4]))
                for row in prediction.boxes.data.cpu().numpy() if int(row[5]) == 0]
        selected, reason = choose_line(rows, c-a, d-b)
        if selected is None:
            return dict(unknown(reason), line_detections=rows)
        local = selected['box']
        x1,y1,x2,y2 = [max(a, math.floor(local[0]+a)), max(b, math.floor(local[1]+b)),
                       min(c, math.ceil(local[2]+a)), min(d, math.ceil(local[3]+b))]
        crop = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.createCLAHE(clipLimit=2., tileGridSize=(4,2)).apply(gray)
        readings = [self.ocr.recognize(crop), self.ocr.recognize(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR))]
        return dict(resolve_readings(readings, [x1,y1,x2,y2]),
                    line_confidence=selected['confidence'], line_detections=rows)
