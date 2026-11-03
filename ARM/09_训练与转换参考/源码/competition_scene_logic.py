"""PC replay decisions. All permissions concern an explicitly virtual scenario.

No UART, physical vehicle control, pixel presets, or annotated light selections.
Perception validity and physical lane verification are distinct fields.
"""
from dataclasses import asdict, dataclass, replace
import math


MODES = ("signal", "boundary", "pedestrian")
KEYS = set("123wasdcf567")


@dataclass(frozen=True)
class Controls:
    mode: str = "signal"
    lane_id: str = "straight_virtual_lane"
    movement: str = "straight"
    requested_motion: str = "stop"
    requested_turn: str = "none"
    forced: bool = False


def apply_key(controls, key):
    if key not in KEYS:
        raise ValueError(f"Unsupported scene key: {key!r}")
    if key in "123":
        direction = {"1": "left", "2": "straight", "3": "right"}[key]
        return replace(controls, lane_id=direction + "_virtual_lane", movement=direction,
                       forced=False)
    if key in "567":
        return replace(controls, mode={"5": "signal", "6": "boundary", "7": "pedestrian"}[key],
                       forced=False, requested_turn="none")
    if key == "w":
        return replace(controls, requested_motion="go")
    if key == "s":
        return replace(controls, requested_motion="stop", forced=False)
    if key in "ad":
        return replace(controls, requested_turn="left" if key == "a" else "right")
    if key == "c":
        return replace(controls, requested_turn="none", forced=False)
    if key == "f":
        return replace(controls, forced=not controls.forced)
    raise ValueError("Unhandled scene key")


def validate_events(value):
    """Events contain driver intent only; perception answers are forbidden."""
    if not isinstance(value, list):
        raise ValueError("Events must be a list of {frame_id, keys}")
    result = []
    previous = -1
    for row in value:
        if not isinstance(row, dict) or set(row) != {"frame_id", "keys"}:
            raise ValueError("Only frame_id and keys are allowed in an event")
        frame, keys = row["frame_id"], row["keys"]
        if type(frame) is not int or frame <= previous:
            raise ValueError("Event frame IDs must be nonnegative and strictly increasing")
        if not isinstance(keys, list) or not keys or any(not isinstance(k, str) or k not in KEYS for k in keys):
            raise ValueError("Event keys must be a nonempty list of supported key strings")
        result.append(dict(frame_id=frame, keys=list(keys)))
        previous = frame
    return result


def finite_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def same_frame(observation, frame):
    return (isinstance(observation, dict) and all(observation.get(k) == frame.get(k)
            for k in ("camera_id", "frame_id", "captured_ms", "image_identity")))


class SceneController:
    def __init__(self, max_age_ms=200., green_confirmations=3):
        if not finite_number(max_age_ms) or max_age_ms <= 0 or type(green_confirmations) is not int or green_confirmations < 1:
            raise ValueError("Invalid scene freshness/confirmation policy")
        self.max_age_ms = max_age_ms
        self.required_green = green_confirmations
        self.watermarks = {}
        self.context = None
        self.last_green = None
        self.green_count = 0
        self.last_now = None

    def reset(self):
        self.last_green = None
        self.green_count = 0

    def update(self, frame, controls, *, now_ms, signal=None, boundaries=None, pedestrian=None):
        if controls.mode not in MODES or controls.movement not in ("left", "straight", "right"):
            raise ValueError("Invalid scene controls")
        if controls.requested_motion not in ("stop", "go") or controls.requested_turn not in ("none", "left", "right") or type(controls.forced) is not bool:
            raise ValueError("Invalid motion controls")
        result = dict(simulation_only=True, hardware_command=None, hardware_command_sent=False,
                      physical_lane_association_verified=False, action="HOLD", alarm=False,
                      decision_valid=False, reason="no_current_evidence", green_confirmations=0,
                      scope="PC virtual scenario only; physical road association is unverified",
                      controls=asdict(controls))

        def finish(reason, action="HOLD", alarm=False, valid=False, reset=True):
            if reset:
                self.reset()
            return dict(result, reason=reason, action=action, alarm=alarm, decision_valid=valid,
                        green_confirmations=self.green_count)

        if not finite_number(now_ms) or now_ms < 0:
            raise ValueError("Invalid monotonic decision time")
        if self.last_now is not None and now_ms < self.last_now:
            self.last_now = now_ms
            return finish("clock_moved_backwards")
        self.last_now = now_ms
        frame_id, captured = frame.get("frame_id"), frame.get("captured_ms")
        camera = frame.get("camera_id")
        if (not isinstance(camera, str) or not camera or type(frame_id) is not int or frame_id < 0
                or not finite_number(captured) or captured < 0 or not isinstance(frame.get("image_identity"), str)
                or len(frame["image_identity"]) != 64):
            return finish("invalid_frame_metadata")
        previous = self.watermarks.get(camera)
        if previous and (frame_id <= previous[0] or captured <= previous[1]):
            return finish("replayed_or_out_of_order_frame")
        self.watermarks[camera] = (frame_id, captured)
        if not 0 <= now_ms - captured <= self.max_age_ms:
            return finish("stale_or_future_frame")
        context = (camera, controls.mode, controls.lane_id, controls.movement, controls.requested_motion)
        if context != self.context:
            self.context = context
            self.reset()
        if controls.requested_motion == "stop":
            return finish("stop_key", "STOP", valid=True)
        if controls.mode == "pedestrian" and pedestrian is not None:
            if not same_frame(pedestrian, frame):
                return finish("pedestrian_result_frame_mismatch")
            if pedestrian.get("intrusion") is True:
                return finish("person_in_virtual_path", "STOP", alarm=True, valid=True)
        if controls.mode == "pedestrian":
            if not same_frame(pedestrian, frame) or not pedestrian.get("observation_valid"):
                return finish("unknown_pedestrian_observation")
            return finish("no_detected_person_in_virtual_path", "GO", valid=True)
        if controls.mode == "boundary":
            self.reset()
            if controls.requested_turn == "none":
                return finish("no_lane_change_requested", "KEEP_LANE", valid=True)
            if not same_frame(boundaries, frame):
                return finish("boundary_result_frame_mismatch")
            side = boundaries.get(controls.requested_turn, {})
            if not side.get("simulation_association_valid") or not side.get("state_valid"):
                return finish(side.get("reason", "unknown_requested_side_boundary"))
            if side.get("line_type") == "solid":
                return finish("forced_solid_line_alarm" if controls.forced else "solid_line_keep_lane",
                              "KEEP_LANE", alarm=controls.forced, valid=True)
            if side.get("line_type") == "dashed":
                return finish("dashed_permission_by_marking_only", "CHANGE_" + controls.requested_turn.upper(), valid=True)
            return finish("unknown_requested_side_boundary")
        if not same_frame(signal, frame):
            return finish("signal_result_frame_mismatch")
        if (not signal.get("simulation_association_valid") or not signal.get("selected_valid")
                or signal.get("movement") != controls.movement or signal.get("lane_id") != controls.lane_id):
            return finish(signal.get("reason", "unknown_related_signal"))
        state = signal.get("selected_state")
        if state == "red":
            return finish("forced_red_light_alarm" if controls.forced else "red_light_stop",
                          "STOP", alarm=controls.forced, valid=True)
        if state == "yellow":
            return finish("yellow_light_stop", "STOP", valid=True)
        if state != "green":
            return finish("unknown_selected_color")
        head = signal.get("head_id")
        if not isinstance(head, str) or not head:
            return finish("missing_automatic_head_identity")
        current = (camera, head, captured)
        if (self.last_green is None or self.last_green[:2] != current[:2]
                or captured - self.last_green[2] > self.max_age_ms):
            self.green_count = 0
        self.green_count += 1
        self.last_green = current
        if self.green_count < self.required_green:
            return finish("confirming_current_green", reset=False)
        return finish("confirmed_green_in_virtual_direction_scenario", "GO", valid=True, reset=False)
