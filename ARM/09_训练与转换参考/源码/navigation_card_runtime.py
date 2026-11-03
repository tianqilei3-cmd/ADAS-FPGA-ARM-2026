"""Card detection -> native-resolution glyph extraction -> route intent.

PC implementation. No OCR, junction timing, physical steering, or FPGA output.
The card detector can later be replaced by an RKNN adapter; that is not done here.
"""
from pathlib import Path
import cv2
import numpy as np

from navigation_hog_classifier import NavigationHOGClassifier
from navigation_primary_locator import overlap

CARD_CONFIDENCE = .25


class CardGlyphReader:
    def __init__(self, weights):
        self.classifier = NavigationHOGClassifier(weights)

    def read(self, image, card_box, card_class):
        height, width = image.shape[:2]
        x1, y1, x2, y2 = map(int, card_box)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        if x2-x1 < 40 or y2-y1 < 18:
            return dict(direction='unknown', direction_valid=False, reason='card_too_small',
                        glyph_box=None, immediate_turn_authorized=False, candidates=[])
        card = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
        ch, cw = gray.shape
        rows = []
        for inverted in [False, True]:
            source = 255-gray if inverted else gray
            for block, bias in [(31, 5), (61, 9)]:
                mask = cv2.adaptiveThreshold(source, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, block, bias)
                n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
                for xx, yy, ww, hh, area in stats[1:]:
                    # Crop edges/card outlines are not independent glyphs.
                    if xx < 2 or yy < 2 or xx+ww >= cw-2 or yy+hh >= ch-2:
                        continue
                    center_x, center_y = (xx+ww/2)/cw, (yy+hh/2)/ch
                    min_h = max(8, ch*(.14 if card_class == 0 else .22))
                    max_h = ch*(.65 if card_class == 0 else .94)
                    if not (min_h <= hh <= max_h and 5 <= ww <= cw*.28 and .18 <= ww/hh <= 1.8):
                        continue
                    if not (.08 <= area/(ww*hh) <= .85) or area < 14:
                        continue
                    if xx/cw > .20 or center_x > .32:
                        continue
                    if not (.03 <= center_y <= (.52 if card_class == 0 else .92)):
                        continue
                    box = [max(x1, x1+xx-2), max(y1, y1+yy-2),
                           min(x2, x1+xx+ww+2), min(y2, y1+yy+hh+2)]
                    box = list(map(int, box))
                    if any(overlap(box, row['box']) > .72 for row in rows): continue
                    crop = image[box[1]:box[3], box[0]:box[2]]
                    prediction = self.classifier.predict([crop])[0]
                    rows.append(dict(box=box, ink_area=int(area), prediction=prediction,
                                     center=[float(center_x), float(center_y)]))
        if not rows:
            return dict(direction='unknown', direction_valid=False, reason='no_primary_glyph',
                        glyph_box=None, candidates=[], immediate_turn_authorized=False)
        # Largest foreground in the left portion of the first instruction row.
        # Unsupported symbols remain unknown; never fall through to a lane arrow.
        rows.sort(key=lambda r:r['ink_area'], reverse=True)
        primary = rows[0]; prediction = primary['prediction']
        return dict(direction=prediction['direction'], direction_valid=prediction['direction_valid'],
                    reason='card_primary_glyph' if prediction['direction_valid'] else 'unsupported_or_uncertain_primary_glyph',
                    glyph_box=primary['box'], glyph_prediction=prediction, candidates=rows,
                    immediate_turn_authorized=False)


class NavigationCardPipeline:
    def __init__(self, detector_weights, glyph_weights):
        from ultralytics import YOLO
        self.detector = YOLO(str(Path(detector_weights)))
        self.reader = CardGlyphReader(glyph_weights)

    def predict(self, image):
        if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Expected uint8 BGR frame')
        prediction = self.detector.predict(image, imgsz=512, conf=CARD_CONFIDENCE,
                                           iou=.5, max_det=10, device='cpu', verbose=False)[0]
        cards = [dict(box=box.tolist(), confidence=float(confidence), class_id=int(label))
                 for box, confidence, label in zip(prediction.boxes.xyxy.cpu().numpy(),
                     prediction.boxes.conf.cpu().numpy(), prediction.boxes.cls.cpu().numpy())]
        cards.sort(key=lambda r:r['confidence'], reverse=True)
        base = dict(cards=cards, card_box=None, glyph_box=None, direction='unknown', direction_valid=False,
                    distance_reading=False, immediate_turn_authorized=False, selected_for_control=False,
                    hardware_access_performed=False)
        if not cards: return dict(**base, reason='no_navigation_card')
        if len(cards) > 1 and cards[1]['confidence'] >= cards[0]['confidence']-.05 and overlap(cards[0]['box'], cards[1]['box']) < .35:
            return dict(**base, reason='ambiguous_navigation_cards')
        card = cards[0]
        result = self.reader.read(image, card['box'], card['class_id'])
        base.update(result, card_box=card['box'], card_class=card['class_id'], card_confidence=card['confidence'])
        return base
