"""Primary-distance OCR adapter; reused mobile PP-OCRv4, no fine-tuning.

Preprocess/CTC conventions follow pinned Apache-2.0 RapidOCR sources recorded in
external_models/navigation_ocr/rapidocr_v4_mobile/provenance.json. Adapted here
to a navigation-card ROI, strict units and explicit uncertainty.
"""
import math
from pathlib import Path
import re
import sys
import unicodedata
import cv2
import numpy as np

from navigation_icon_common import ROOT


def _distance_literal(text):
    if not isinstance(text, str): return None
    normalized = unicodedata.normalize('NFKC', text).strip()
    if any(token in normalized for token in ['/', ':', '每秒', '每小时', '时速', '分钟']): return None
    # Optional trailing Chinese road name is tolerated only after a clear unit.
    # No digits alone, speed/time strings, signs or fuzzy O/0 substitutions.
    match = re.fullmatch(r'([0-9]{1,4}(?:\.[0-9]{1,2})?)\s*(公里|千米|km|KM|米|m|M)([\u4e00-\u9fff]{0,12})', normalized)
    if not match: return None
    value = float(match.group(1)); unit = match.group(2)
    metres = value*(1000 if unit in ['公里', '千米', 'km', 'KM'] else 1)
    if not math.isfinite(metres) or metres < 0: return None
    return dict(distance_m=metres, unit='m', observed_value=value, observed_unit=unit,
                normalized_text=normalized)


def parse_distance(text):
    parsed = _distance_literal(text)
    return parsed if parsed is not None and parsed['distance_m'] <= 10000 else None


def numeric_hint(reading):
    """Display-only numeric prefix; never gives a unit or valid distance."""
    text = unicodedata.normalize('NFKC', reading.get('text', '')).strip()
    match = re.match(r'^([0-9]{1,4}(?:\.[0-9]{1,2})?)(?=$|[^0-9])', text)
    if not match: return None
    count = len(match.group(1)); probabilities = reading.get('character_confidences', [])[:count]
    if len(probabilities) != count or min(probabilities) < .60 or np.mean(probabilities) < .85: return None
    return float(match.group(1))


def distance_roi(image, nav):
    if not nav.get('card_box') or not nav.get('glyph_box'): return None
    if len(nav['card_box']) != 4 or len(nav['glyph_box']) != 4: return None
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in [*nav['card_box'], *nav['glyph_box']]): return None
    x1, y1, x2, y2 = map(float, nav['card_box'])
    gx1, gy1, gx2, gy2 = map(float, nav['glyph_box'])
    if x2 <= x1 or y2 <= y1 or gx2 <= gx1 or gy2 <= gy1: return None
    if not (x1 <= (gx1+gx2)/2 <= x2 and y1 <= (gy1+gy2)/2 <= y2): return None
    unit = gy2-gy1
    if unit < 8: return None
    left = max(x1, gx2+.03*unit)
    right = min(x2, x1+(x2-x1)*.64, left+4.2*unit)
    if nav.get('card_class') == 1:
        top, bottom = gy1-.25*unit, gy2+.25*unit
    else:
        top, bottom = gy1-.15*unit, gy1+.65*unit
    box = [max(0, int(left)), max(0, int(top)), min(image.shape[1], int(right)), min(image.shape[0], int(bottom))]
    if box[2]-box[0] < 12 or box[3]-box[1] < 6: return None
    return box


def crop_text_line(crop):
    """Trim blank margins of a dark navigation text line, retaining small units.

    The recognizer expects a text-line crop. Large blank margins shrink the font
    during height normalization and can generate spurious trailing characters.
    This only changes image framing; it never replaces or guesses characters.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 5)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    components = [s for s in stats[1:] if s[4] >= 5 and s[3] >= 4
                  and s[2] >= 2 and s[3] < gray.shape[0]*.95]
    if not components:
        return crop
    # All qualifying components contribute, including a unit smaller than digits.
    left = max(0, min(int(s[0]) for s in components)-2)
    top = max(0, min(int(s[1]) for s in components)-2)
    right = min(crop.shape[1], max(int(s[0]+s[2]) for s in components)+2)
    bottom = min(crop.shape[0], max(int(s[1]+s[3]) for s in components)+2)
    return crop[top:bottom, left:right]


class MobileTextRecognizer:
    def __init__(self, model):
        try:
            import onnxruntime as ort
        except ImportError:
            isolated = ROOT/'external_models/lane_runtime'
            if not (isolated/'onnxruntime').exists():
                raise RuntimeError('ONNX Runtime unavailable; preserve base environment and configure an isolated runtime')
            sys.path.append(str(isolated))
            import onnxruntime as ort
        options = ort.SessionOptions(); options.intra_op_num_threads = 4; options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(Path(model)), sess_options=options, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        character = self.session.get_modelmeta().custom_metadata_map.get('character')
        if character is None: raise RuntimeError('Pinned OCR model has no embedded dictionary')
        self.characters = ['blank']+character.split('\n')+[' ']
        if self.session.get_outputs()[0].shape[-1] != len(self.characters):
            raise RuntimeError('OCR dictionary/output mismatch')

    def recognize(self, crop):
        crop = crop_text_line(crop)
        height, width = crop.shape[:2]
        resized_width = min(640, max(1, math.ceil(48*width/height)))
        # BGR convention matches the pinned upstream implementation.
        image = cv2.resize(crop, (resized_width, 48)).astype(np.float32).transpose(2, 0, 1)/127.5-1
        padded = np.zeros((1, 3, 48, max(320, resized_width)), np.float32)
        padded[0, :, :, :resized_width] = image
        probabilities = self.session.run(None, {self.input_name:padded})[0][0]
        indices = probabilities.argmax(1); values = probabilities.max(1)
        text, confidence = [], []
        previous = -1
        for index, value in zip(indices, values):
            if index != 0 and index != previous:
                text.append(self.characters[int(index)]); confidence.append(float(value))
            previous = int(index)
        return dict(text=''.join(text), confidence=float(np.mean(confidence)) if confidence else 0.,
                    minimum_character_confidence=min(confidence) if confidence else 0.,
                    character_confidences=confidence)


class NavigationDistanceReader:
    def __init__(self, weights): self.ocr = MobileTextRecognizer(weights)

    def read(self, image, nav):
        box = distance_roi(image, nav)
        result = dict(distance_m=None, distance_valid=False, distance_box=box, unit=None,
                      numeric_value_hint=None, numeric_hint_unit=None, numeric_hint_for_control=False,
                      immediate_turn_authorized=False, reason='no_primary_distance_region')
        if box is None: return result
        x1, y1, x2, y2 = box; crop = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        components = [tuple(map(int, s)) for s in stats[1:] if s[4] >= 5 and s[3] >= 4 and s[2] >= 2 and s[3] < gray.shape[0]*.95]
        if components:
            maximum = max(s[3] for s in components)
            main = [s for s in components if s[3] >= maximum*.65 and s[2] <= maximum*4]
            if main:
                top = max(0, min(s[1] for s in main)-2)
                bottom = min(crop.shape[0], max(s[1]+s[3] for s in main)+2)
                crop = crop[top:bottom]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        contrast = cv2.createCLAHE(clipLimit=2., tileGridSize=(4, 2)).apply(gray)
        variants = [crop, cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR)]
        readings = [self.ocr.recognize(view) for view in variants]
        result.update(ocr=readings[0], ocr_views=readings)
        hints = {value for raw in readings if (value := numeric_hint(raw)) is not None}
        if len(hints) == 1: result['numeric_value_hint'] = hints.pop()
        valid = []
        confident_literals = []
        for raw in readings:
            parsed = parse_distance(raw['text'])
            literal = _distance_literal(raw['text'])
            if literal is not None and raw['confidence'] >= .85 and raw['minimum_character_confidence'] >= .50:
                confident_literals.append(literal['distance_m'])
            if parsed is not None and raw['confidence'] >= .85 and raw['minimum_character_confidence'] >= .50:
                valid.append((parsed, raw))
        if len(set(confident_literals)) > 1:
            result['reason'] = 'conflicting_ocr_views'; return result
        if not valid:
            result['reason'] = 'unreadable_or_missing_explicit_distance_unit'; return result
        if len({p['distance_m'] for p, _ in valid}) > 1:
            result['reason'] = 'conflicting_ocr_views'; return result
        parsed, raw = max(valid, key=lambda pair:pair[1]['confidence'])
        for alternative in readings:
            alternative_parse = parse_distance(alternative['text'])
            hint = numeric_hint(alternative)
            if alternative_parse is None and hint is not None and hint != parsed['observed_value']:
                result['reason'] = 'conflicting_numeric_views'; return result
        result.update(parsed, ocr=raw, distance_valid=True, reason='explicit_primary_distance_read')
        return result
