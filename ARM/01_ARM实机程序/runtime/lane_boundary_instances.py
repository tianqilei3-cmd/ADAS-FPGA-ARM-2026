"""Frame-local boundary paths and unverified image-center pair proposals.

Input is the current 3-class corridor mask, not exact paint. Geometry follows
observed rows only. Types are never inferred from corridor continuity. No
camera calibration, temporal persistence, physical ego-lane claim or control.
"""
from dataclasses import asdict, dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class InstancePolicy:
    row_merge_gap: int = 2
    max_gap_rows: int = 3
    max_run_width: int = 28
    min_span: int = 18
    min_rows: int = 14
    max_dx_dy: float = 2.25
    max_fit_error: float = 3.0
    min_type_fraction: float = .85
    conflict_row_fraction: float = .20
    min_conflict_rows: int = 4
    camera_center_fraction: float = .50

    def __post_init__(self):
        for name in ('row_merge_gap', 'max_gap_rows', 'max_run_width', 'min_span', 'min_rows', 'min_conflict_rows'):
            value = getattr(self, name)
            if type(value) is not int or value < 0 or (name not in ('row_merge_gap', 'max_gap_rows') and value == 0):
                raise ValueError('Invalid instance policy: ' + name)
        for name in ('max_dx_dy', 'max_fit_error'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError('Invalid geometry threshold')
        if not (.5 < self.min_type_fraction <= 1 and 0 < self.conflict_row_fraction < .5
                and 0 < self.camera_center_fraction < 1):
            raise ValueError('Invalid type or camera-center threshold')


def row_runs(row, merge_gap):
    """Foreground intervals. Joining a tiny inter-stripe gap is geometry only."""
    x = np.flatnonzero(row > 0)
    if not len(x):
        return []
    breaks = np.flatnonzero(np.diff(x) > merge_gap + 1)
    starts, ends = np.r_[0, breaks + 1], np.r_[breaks, len(x) - 1]
    return [(int(x[a]), int(x[b])) for a, b in zip(starts, ends)]


def transverse_yellow(image):
    """Observed yellow transverse components at mask resolution; not line type."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yellow = ((hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 40)
              & (hsv[:, :, 1] >= 80) & (hsv[:, :, 2] >= 105)).astype(np.uint8)
    h, w = yellow.shape
    yellow[:round(h * .4)] = 0
    yellow[round(h * .94):] = 0
    _, _, stats, _ = cv2.connectedComponentsWithStats(yellow, 8)
    return [list(map(int, s[:4])) for s in stats[1:]
            if s[2] >= max(8, w / 64) and s[3] >= 2 and s[2] / s[3] >= 2.5 and s[4] >= 10]


def _trace(mask, blocked, p, *, suppress_stale_fragments=False):
    tracks, complete = [], []
    for y, row in enumerate(mask):
        active = []
        for track in tracks:
            if y - track['points'][-1][1] <= p.max_gap_rows + 1:
                active.append(track)
            else:
                complete.append(track)
        tracks = active
        nodes = []
        for a, b in row_runs(row, p.row_merge_gap):
            if b - a + 1 > p.max_run_width:
                continue
            positions = np.flatnonzero(row[a:b + 1]) + a
            nodes.append(dict(x=float(np.median(positions)), y=y, a=a, b=b,
                              counts=[int((row[a:b + 1] == label).sum()) for label in (1, 2)]))
        links = []
        for ti, track in enumerate(tracks):
            x0, y0 = track['points'][-1]
            # V2: a vanished, unconfirmed speck must not steal an established path.
            if suppress_stale_fragments and y - y0 > 1 and len(track['points']) < 4:
                continue
            slope = 0.
            if len(track['points']) >= 3:
                xp, yp = track['points'][-3]
                slope = np.clip((x0 - xp) / (y0 - yp), -p.max_dx_dy, p.max_dx_dy)
            for ni, node in enumerate(nodes):
                dy = y - y0
                if abs(node['x'] - (x0 + slope * dy)) > 3 + p.max_dx_dy * dy:
                    continue
                # Never bridge a visible occluder by treating it as a dashed gap.
                crossed = False
                for yy in range(y0 + 1, y + 1):
                    xx = round(x0 + (node['x'] - x0) * (yy - y0) / dy)
                    if blocked[yy, max(0, xx - 1):min(mask.shape[1], xx + 2)].any():
                        crossed = True
                        break
                if not crossed:
                    links.append((ti, ni))
        attached = set()
        retired = set()
        # A fork/merge ends the ambiguous tracks; no forced nearest-neighbor join.
        for ti, track in enumerate(tracks):
            choices = [n for t, n in links if t == ti]
            if len(choices) == 1 and sum(n == choices[0] for _, n in links) == 1:
                node = nodes[choices[0]]
                track['points'].append((node['x'], y))
                track['counts'].append(node['counts'])
                track['widths'].append(node['b'] - node['a'] + 1)
                attached.add(choices[0])
            elif choices:
                track['ambiguous'] = True
                complete.append(track)
                retired.add(ti)
        tracks = [t for i, t in enumerate(tracks) if i not in retired]
        for ni, node in enumerate(nodes):
            if ni not in attached:
                tracks.append(dict(points=[(node['x'], y)], counts=[node['counts']],
                                   widths=[node['b'] - node['a'] + 1], ambiguous=any(n == ni for _, n in links)))
    return complete + tracks


def propose_pair(instances, shape, center_fraction):
    """A geometric image-center bracket only. Never marks association verified."""
    h, w = shape
    center = center_fraction * w
    candidates = []
    for left in instances:
        for right in instances:
            if left['id'] == right['id'] or left['ambiguous'] or right['ambiguous']:
                continue
            top = max(left['y_range'][0], right['y_range'][0])
            bottom = min(left['y_range'][1], right['y_range'][1])
            if bottom - top < 12 or bottom < .70 * h:
                continue
            ys = np.array([top, (top + bottom) / 2, bottom])
            lx = np.polyval(left['x_of_y'], ys)
            rx = np.polyval(right['x_of_y'], ys)
            width = rx - lx
            if not (np.all(lx < center - .01 * w) and np.all(rx > center + .01 * w)
                    and .10 * w <= width[-1] <= .8 * w and width[-1] >= width[0] * .90):
                continue
            candidates.append(dict(left_id=left['id'], right_id=right['id'],
                                   common_y=[int(top), int(bottom)], near_width=float(width[-1])))
    result = dict(state='unknown', left_id=None, right_id=None, camera_center_assumed=center_fraction,
                  camera_calibrated=False, ego_lane_association_verified=False,
                  selected_lane_valid=False, simulation_action='HOLD', hardware_command_sent=False)
    if not candidates:
        return dict(result, reason='no_supported_image_center_pair', alternatives=[])
    candidates.sort(key=lambda c: c['near_width'])
    if len(candidates) > 1:
        return dict(result, reason='multiple_image_center_pairs', alternatives=candidates)
    return dict(result, **{k: v for k, v in candidates[0].items() if k not in result},
                left_id=candidates[0]['left_id'], right_id=candidates[0]['right_id'],
                state='geometric_candidate_unverified', reason='single_image_center_bracket', alternatives=candidates)


def extract_instances(mask, image, occluders, policy=None, *, trace_function=None):
    """Return supported paths; type conflicts retain geometry but invalidate style."""
    p = policy or InstancePolicy()
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != np.uint8 or not np.isin(mask, [0, 1, 2]).all():
        raise ValueError('Expected uint8 corridor mask with labels0/1/2')
    if image is None or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('Expected current uint8 BGR source frame')
    h, w = mask.shape
    sh, sw = image.shape[:2]
    if h < 32 or w < 32:
        raise ValueError('Mask is too small for the fixed pixel policy')
    work = mask.copy()
    blocked = np.zeros_like(mask, dtype=bool)
    for obj in occluders:
        if obj.get('name') not in ('vehicle', 'two_wheeler', 'person') or obj.get('confidence', 0) < .35:
            continue
        x1, y1, x2, y2 = obj['box']
        if not np.isfinite([x1, y1, x2, y2]).all() or x2 <= x1 or y2 <= y1:
            raise ValueError('Invalid same-frame occluder')
        if x2 <= 0 or y2 <= 0 or x1 >= sw or y1 >= sh:
            continue
        a, b = max(0, int(x1 * w / sw) - 1), max(0, int(y1 * h / sh) - 1)
        c, d = min(w, round(x2 * w / sw) + 2), min(h, round(y2 * h / sh) + 2)
        blocked[b:d, a:c] = True
    work[blocked] = 0
    yellow = transverse_yellow(cv2.resize(image, (w, h)))
    accepted, rejected = [], []
    for track in (trace_function or _trace)(work, blocked, p):
        points = np.asarray(track['points'], dtype=float)
        span = points[-1, 1] - points[0, 1] + 1
        reason = None
        if len(points) < p.min_rows or span < p.min_span or len(points) / span < .70:
            reason = 'short_or_sparse_fragment'
        else:
            coefficients = np.polyfit(points[:, 1], points[:, 0], 2 if span >= 40 else 1)
            error = float(np.sqrt(np.mean((np.polyval(coefficients, points[:, 1]) - points[:, 0]) ** 2)))
            derivative = np.polyval(np.polyder(coefficients), points[:, 1])
            if abs(derivative).max() > p.max_dx_dy:
                reason = 'near_transverse_geometry'
            elif error > p.max_fit_error:
                reason = 'unstable_or_branched_geometry'
        if reason:
            rejected.append(dict(reason=reason, rows=len(points), y_range=points[[0, -1], 1].astype(int).tolist()))
            continue
        counts = np.asarray(track['counts'], dtype=float)
        fractions = counts / np.maximum(1., counts.sum(1, keepdims=True))
        shares = fractions.mean(0)
        significant = (fractions >= p.conflict_row_fraction).sum(0)
        conflict = bool((significant >= p.min_conflict_rows).all())
        bars = []
        for x, y, cw, ch in yellow:
            intersect = ((points[:, 1] >= y - 1) & (points[:, 1] < y + ch + 1)
                         & (points[:, 0] >= x - 2) & (points[:, 0] <= x + cw + 2)).any()
            if intersect:
                bars.append([x, y, cw, ch])
        label = int(shares.argmax()) + 1
        line_type = 'solid' if label == 1 else 'dashed'
        style_reason = 'consistent_semantic_support'
        if track['ambiguous']:
            line_type, style_reason = 'unknown', 'branch_or_merge_ambiguity'
        elif conflict or shares.max() < p.min_type_fraction:
            line_type, style_reason = 'unknown', 'conflicting_type_segments'
        elif len(bars) >= 2:
            line_type, style_reason = 'unknown', 'repeated_yellow_transverse_intersections'
        accepted.append(dict(id=len(accepted), line_type=line_type, state_valid=line_type != 'unknown',
             reason=style_reason, ambiguous=track['ambiguous'], x_of_y=coefficients.tolist(),
             y_range=points[[0, -1], 1].astype(int).tolist(), supported_rows=len(points),
             points_work=points.tolist(), points_native=(points * [sw / w, sh / h]).tolist(),
             fit_rms_work=error, mean_type_fraction=shares.tolist(), significant_type_rows=significant.tolist(),
             yellow_transverse_components=bars, ego_lane_association_verified=False))
    pair = propose_pair(accepted, (h, w), p.camera_center_fraction)
    return dict(instances=accepted, rejected=rejected, pair=pair, policy=asdict(p),
                source_size=[sw, sh], mask_size=[w, h], ego_lane_association_verified=False,
                selected_lane_valid=False, simulation_action='HOLD', hardware_command_sent=False)
