"""Lightweight ARM scene adapter. No torch, label access or hardware actuation."""
import hashlib
import math
from pathlib import Path

import cv2
import numpy as np
from immutable_identity import pixel_digest


def box_iou(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    intersection = np.maximum(0., np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    union = np.maximum(0., a[2:] - a[:2]).prod() + np.maximum(0., b[2:] - b[:2]).prod() - intersection
    return float(intersection / union) if union > 0 else 0.


def frame_identity(image, source, visit, sequence, captured):
    return dict(camera_id=f'file:{Path(source).name}:visit:{visit}', frame_id=sequence,
                captured_ms=captured * 1000, image_identity=pixel_digest(image))


def scene_decision(frame, controls, hazard, now_ms):
    result = dict(action='STOP', reason='mode_perception_pending', alarm=False,
                  decision_valid=False, simulation_only=True, motor_command_sent=False,
                  own_lane_light_valid=False, physical_lane_association_verified=False,
                  source='current_NPU_road_detections_and_declared_virtual_corridor')
    age = now_ms - frame['captured_ms']
    if not math.isfinite(age) or not 0 <= age <= 500:
        return dict(result, reason='stale_or_future_frame', alarm=bool(hazard.get('active_tracks')))
    if controls.requested_motion == 'stop':
        return dict(result, reason='stop_key', decision_valid=True)
    if controls.mode != 'pedestrian':
        return dict(result, reason='lane_light_unknown' if controls.mode == 'signal' else 'boundary_not_on_ARM')
    action = hazard.get('action')
    if action == 'GO' and hazard.get('decision_valid'):
        return dict(result, action='RISK_CLEAR', reason=hazard['reason'], decision_valid=True,
                    clearance_scope='virtual_hazard_check_only_not_traffic_permission')
    return dict(result, reason=hazard.get('reason', 'hazard_unknown'),
                alarm=bool(hazard.get('alarm')), decision_valid=bool(hazard.get('decision_valid')))


def with_diagnostic_stop(wires, decision):
    """V2 bytes stay compatible. No lamp/nav/motion validity is invented.

    arm_action is diagnostic and ignored by the current FPGA. RISK_CLEAR is not
    vehicle permission, so never encode it as GO. Object flags retain ANY-person
    semantics until the teammate supplies an agreed risk-aware protocol.
    """
    status = bytearray(wires[0])
    if len(status) != 14 or status[:3] != bytes([0x55, 0x11, 10]):
        raise ValueError('Expected V2 status packet')
    status[12] = 1 if decision.get('action') == 'STOP' else 0
    status[-1] = sum(status[:-1]) & 255
    return [bytes(status), *wires[1:]]


def draw_scene(canvas, observation, decision, controls):
    image_w, image_h = observation['image_size']
    polygon = np.asarray(observation['polygon'], float) * [1280 / image_w, 720 / image_h]
    cv2.polylines(canvas, [np.rint(polygon).astype(np.int32)], True, (230, 180, 60), 2)
    cv2.rectangle(canvas, (0, 80), (1280, 154), (25, 30, 35), -1)
    color = (40, 210, 240) if decision['action'] == 'RISK_CLEAR' else (70, 100, 255)
    cv2.putText(canvas, f"ARM LOCAL {decision['action']} | {decision['reason']}",
                (12, 108), cv2.FONT_HERSHEY_SIMPLEX, .63, color, 2, cv2.LINE_AA)
    count = sum(item['in_corridor'] and item['confidence'] >= .35 for item in observation['objects'])
    cv2.putText(canvas, f'{controls.mode} | request {controls.requested_motion}/{controls.movement} | '
                f'force {int(controls.forced)} | intrusion {count} | alarm {int(decision["alarm"])} | VIRTUAL CORRIDOR',
                (12, 139), cv2.FONT_HERSHEY_SIMPLEX, .53, (240, 240, 240), 1, cv2.LINE_AA)
