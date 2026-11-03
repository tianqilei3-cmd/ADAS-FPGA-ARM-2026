"""Read current image color at an explicitly selected head for PC simulation.

Selection is supplied per image, with its pixel digest and frame identity. It
does not locate lights, track a head across frames, or verify physical lanes.
The existing type/color models must still accept the selected crop.
"""
import hashlib
import math
import cv2
import numpy as np

from light_roi_common import NAMES, classify_policy, crop_head


def image_identity(image):
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 2):
        raise ValueError('A uint8 BGR image is required')
    payload = str(image.shape).encode('ascii') + np.ascontiguousarray(image).tobytes()
    return hashlib.sha256(payload).hexdigest()


def make_selection(image, box, *, camera_id, frame_id, captured_ms, lane_id, movement):
    """Bind a manual head selection to exactly one source frame, never a color."""
    return dict(box=list(box), image_identity=image_identity(image), camera_id=camera_id,
                frame_id=frame_id, captured_ms=captured_ms, lane_id=lane_id, movement=movement,
                selection_origin='manual_current_frame_head', association_scope='manual_head_to_virtual_lane')


def selection_valid(image, selection, *, camera_id, frame_id, captured_ms):
    if not isinstance(selection, dict):
        return False
    if selection.get('selection_origin') != 'manual_current_frame_head':
        return False
    if selection.get('association_scope') != 'manual_head_to_virtual_lane':
        return False
    if not isinstance(camera_id, str) or not camera_id:
        return False
    if type(frame_id) is not int or frame_id < 0:
        return False
    if isinstance(captured_ms, bool) or not isinstance(captured_ms, (int, float)) or not math.isfinite(captured_ms) or captured_ms < 0:
        return False
    if (selection.get('camera_id') != camera_id or selection.get('frame_id') != frame_id
            or selection.get('captured_ms') != captured_ms):
        return False
    if type(selection.get('frame_id')) is not int or isinstance(selection.get('captured_ms'), bool):
        return False
    if not isinstance(selection.get('lane_id'), str) or not selection['lane_id']:
        return False
    if selection.get('movement') not in ['left', 'straight', 'right', 'all']:
        return False
    box = selection.get('box')
    if not isinstance(box, (tuple, list)) or len(box) != 4:
        return False
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in box):
        return False
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= image.shape[1] and 0 <= y1 < y2 <= image.shape[0]
            and x2-x1 >= 4 and y2-y1 >= 4):
        return False
    return selection.get('image_identity') == image_identity(image)


def interior_lamp_evidence(crop):
    """Separate an enclosed luminous component from colored crop-edge scenery.

    This is local corroboration of a highly confident red classification, not
    a whole-image light detector. Comparable interior colors remain ambiguous.
    """
    hue, saturation, value = cv2.split(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV))
    # Current daylight development threshold; dim/night inputs remain unknown.
    eligible = (saturation >= 80) & (value >= 160)
    masks = [eligible & ((hue <= 12) | (hue >= 165)),
             eligible & (hue >= 15) & (hue <= 34), eligible & (hue >= 35) & (hue <= 95)]
    scores = []
    for mask in masks:
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        components = [s for s in stats[1:] if s[0] > 0 and s[1] > 0
                      and s[0]+s[2] < crop.shape[1]-1 and s[1]+s[3] < crop.shape[0]-1
                      and min(s[2],s[3]) >= 2 and .2 <= s[2]/s[3] <= 5
                      and s[4] >= max(4, crop.shape[0]*crop.shape[1]*.003)]
        scores.append(max((int(s[4]) for s in components), default=0))
    order = np.argsort(-np.asarray(scores));first,second = map(int,order[:2])
    state = first if scores[first] >= 4 and scores[first] >= 3*max(1,scores[second]) else None
    return dict(interior_component_areas=scores,state=state,minimum_value=160)


class SelectedSignalReader:
    def __init__(self, color_weights, attribute_weights):
        from signal_emission_guard import EmissionGuardClassifier
        from signal_attribute_consensus import load_candidate
        self.color = EmissionGuardClassifier(color_weights)
        self.attributes = load_candidate(attribute_weights)

    def read(self, image, selection, *, camera_id, frame_id, captured_ms):
        image_identity(image)
        result = dict(origin='verified_perception', selected_state='unknown', selected_valid=False,
            association_verified=False, association_scope='manual_head_to_virtual_lane',
            physical_lane_association_verified=False, selection_origin='manual_current_frame_head',
            automatic_head_location=False, camera_id=camera_id, frame_id=frame_id, captured_ms=captured_ms,
            lane_id=None, movement=None, hardware_verified=False, reason='no_current_manual_head_selection',
            immediate_turn_authorized=False)
        if not selection_valid(image, selection, camera_id=camera_id, frame_id=frame_id, captured_ms=captured_ms):
            return result
        box = selection['box']
        crop = crop_head(image, box)
        probabilities, emission = self.color.predict_with_guard([crop])
        attributes = self.attributes.predict([crop])[0]
        p = probabilities[0]
        if (p.shape != (4,) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
                or abs(float(p.sum())-1.) > .001 or not math.isfinite(float(emission[0]))
                or not 0 <= emission[0] <= 1):
            result['reason'] = 'invalid_model_probability'
            return result
        decision, color_reason = classify_policy(p, crop, threshold=.85, margin=.20)
        local = interior_lamp_evidence(crop)
        original_color_reason = color_reason
        if (decision == 4 and color_reason == 'color_conflict'
                and int(p.argmax()) == 0 and p[0] >= .95 and local['state'] == 0):
            decision, color_reason = 0, 'strong_red_with_interior_red_emission'
        result.update(association_verified=True, lane_id=selection['lane_id'], movement=selection['movement'],
            box=list(box), color_probabilities=p.tolist(), visible_emission_probability=float(emission[0]),
            attributes=attributes, color_policy_reason=color_reason,
            original_color_policy_reason=original_color_reason, interior_evidence=local,
            image_identity=selection['image_identity'], confidence=float(p.max()),
            confidence_scope='ROI classification probability, no detector score')
        if attributes.get('lamp_type') != 'vehicle_signal':
            result['reason'] = 'selected_head_not_confirmed_vehicle_signal'
            return result
        # An explicit conflicting arrow cannot be overridden by the manual lane key.
        arrow = attributes.get('control_direction')
        if arrow in ['left', 'straight', 'right'] and selection['movement'] not in [arrow]:
            result['reason'] = 'selected_movement_conflicts_with_visible_arrow'
            return result
        if decision == 4 or NAMES[decision] not in ['red', 'yellow', 'green']:
            result['reason'] = 'unknown_off_or_conflicting_color'
            return result
        if NAMES[decision] == 'green' and emission[0] < .50:
            result['reason'] = 'green_without_visible_emission'
            return result
        if NAMES[decision] == 'green' and local['state'] != 2:
            result['reason'] = 'green_without_bright_interior_emission'
            return result
        result.update(selected_state=NAMES[decision], selected_valid=True,
                      reason='current_image_color_for_manually_selected_head')
        return result
