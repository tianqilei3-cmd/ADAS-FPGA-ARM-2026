"""Append geometry from three aligned visible paint fragments near dashed masks.

This does not infer a dashed type from gaps or fill the gaps with new pixels.
All added paths are unknown, with the previous paths and runtime pair intact.
"""
import copy
from dataclasses import asdict, dataclass
from itertools import combinations
import math

import cv2
import numpy as np

from lane_observed_yellow_geometry import overlaps


@dataclass(frozen=True)
class DashedGeometryPolicy:
    semantic_radius: int = 3
    minimum_components: int = 3
    minimum_component_area: int = 5
    minimum_component_rows: int = 3
    minimum_component_axis_ratio: float = 2.5
    minimum_component_axis_span: float = 7.
    minimum_group_span: int = 24
    minimum_paint_contrast: int = 20
    minimum_yellow_chroma: int = 20
    maximum_angle_difference_degrees: float = 15.
    maximum_center_distance: float = 100.
    maximum_perpendicular_error: float = 2.
    maximum_slope: float = 8.

    def __post_init__(self):
        if type(self.semantic_radius) is not int or not 0 <= self.semantic_radius <= 3:
            raise ValueError('Invalid semantic radius')
        for name in ['minimum_components', 'minimum_component_area', 'minimum_component_rows',
                     'minimum_group_span', 'minimum_paint_contrast', 'minimum_yellow_chroma']:
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError('Invalid policy ' + name)
        if self.minimum_components < 3:
            raise ValueError('At least three paint components are required')
        for name in ['minimum_component_axis_ratio', 'minimum_component_axis_span',
                     'maximum_angle_difference_degrees', 'maximum_center_distance',
                     'maximum_perpendicular_error', 'maximum_slope']:
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError('Invalid policy ' + name)
        if self.maximum_angle_difference_degrees >= 45:
            raise ValueError('Angle gate must remain narrow')


def blocked_mask(shape, source_shape, occluders):
    h, w = shape
    sh, sw = source_shape[:2]
    blocked = np.zeros(shape, bool)
    blocked[:round(.4*h)] = True
    blocked[round(.94*h):] = True
    for obj in occluders:
        if obj.get('name') not in ['vehicle', 'person', 'two_wheeler'] or obj.get('confidence', 0) < .35:
            continue
        x1, y1, x2, y2 = obj['box']
        if not np.isfinite([x1, y1, x2, y2]).all() or x2 <= x1 or y2 <= y1:
            raise ValueError('Invalid current-frame occluder')
        if x2 <= 0 or y2 <= 0 or x1 >= sw or y1 >= sh:
            continue
        a, b = max(0, int(x1*w/sw)-1), max(0, int(y1*h/sh)-1)
        c, d = min(w, round(x2*w/sw)+2), min(h, round(y2*h/sh)+2)
        blocked[b:d, a:c] = True
    return blocked


def observed_fragments(classes, image, occluders, policy):
    p = policy
    h, w = classes.shape
    blocked = blocked_mask(classes.shape, image.shape, occluders)
    support = (classes == 2).astype(np.uint8)
    support[blocked] = 0
    near = cv2.dilate(support, np.ones((2*p.semantic_radius+1, 2*p.semantic_radius+1), np.uint8))
    near[blocked] = 0
    small = cv2.resize(image, (w, h))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    channels = small.astype(np.int16)
    chroma = np.minimum(channels[:, :, 1], channels[:, :, 2]) - channels[:, :, 0]
    yellow = ((hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 40) & (hsv[:, :, 2] >= 105)
              & (chroma >= p.minimum_yellow_chroma))
    white = (hsv[:, :, 1] <= 90) & (hsv[:, :, 2] >= 160) & ~yellow
    contrast = cv2.morphologyEx(hsv[:, :, 2], cv2.MORPH_TOPHAT, np.ones((1, 31), np.uint8))
    _, regions = cv2.connectedComponents(near, 8)
    fragments = []
    for color, pixels in [('white', white), ('yellow', yellow)]:
        paint = (pixels & (contrast >= p.minimum_paint_contrast) & (near > 0)).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(paint, 8)
        for i in range(1, n):
            if stats[i, 4] < p.minimum_component_area or stats[i, 3] < p.minimum_component_rows:
                continue
            ys, xs = np.nonzero(labels == i)
            xy = np.column_stack([xs, ys]).astype(float)
            covariance = np.cov(xy, rowvar=False)
            values, vectors = np.linalg.eigh(covariance)
            ratio = math.sqrt(max(0., values[1]) / max(.01, values[0]))
            axis = vectors[:, 1]
            axis *= 1 if axis[1] >= 0 else -1
            projection = xy @ axis
            if ratio < p.minimum_component_axis_ratio or np.ptp(projection) < p.minimum_component_axis_span:
                continue
            points = np.array([[float(np.median(xs[ys == y])), float(y)] for y in np.unique(ys)])
            # Median of a split row may be a gap. Never invent such a paint point.
            nearest = np.round(points).astype(int)
            if not np.all(paint[nearest[:, 1], nearest[:, 0]]):
                continue
            fragments.append(dict(id=len(fragments), color=color, semantic_region=int(regions[ys[0], xs[0]]),
                points=points, center=points.mean(0), axis=axis, axis_ratio=ratio,
                y_range=[int(ys.min()), int(ys.max())], component_area=int(stats[i, 4])))
    return fragments, blocked


def compatible(a, b, blocked, p):
    if a['color'] != b['color'] or a['semantic_region'] != b['semantic_region']:
        return False
    upper, lower = sorted([a, b], key=lambda t: t['y_range'][0])
    if lower['y_range'][0] <= upper['y_range'][1] + 1:
        return False
    delta = b['center'] - a['center']
    distance = float(np.linalg.norm(delta))
    if distance < p.minimum_component_axis_span or distance > p.maximum_center_distance:
        return False
    axis = delta / distance
    cosine = math.cos(math.radians(p.maximum_angle_difference_degrees))
    if any(abs(np.dot(axis, part['axis'])) < cosine for part in [a, b]):
        return False
    points = np.vstack([a['points'], b['points']])
    normal = np.array([-axis[1], axis[0]])
    if np.abs((points - a['center']) @ normal).max() > p.maximum_perpendicular_error:
        return False
    # Explicitly reject occluders between fragments, even though no gap is drawn.
    start, end = upper['points'][-1], lower['points'][0]
    for y in range(int(start[1]), int(end[1])+1):
        x = round(start[0] + (end[0]-start[0]) * (y-start[1]) / (end[1]-start[1]))
        if blocked[y, max(0, x-1):min(blocked.shape[1], x+2)].any():
            return False
    return True


def augment(before, classes, image, occluders, policy=None):
    p = policy or DashedGeometryPolicy()
    classes = np.asarray(classes)
    if classes.dtype != np.uint8 or classes.ndim != 2 or not np.isin(classes, [0,1,2]).all():
        raise ValueError('Expected uint8 semantic labels0/1/2')
    if classes.shape != (288,512):
        raise ValueError('This fixed-pixel candidate is validated at512x288 only')
    if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected current BGR frame')
    fragments, blocked = observed_fragments(classes, image, occluders, p)
    graph = {i: set() for i in range(len(fragments))}
    for i, j in combinations(graph, 2):
        if compatible(fragments[i], fragments[j], blocked, p):
            graph[i].add(j)
            graph[j].add(i)
    groups, visited = [], set()
    for seed in graph:
        if seed in visited:
            continue
        pending, members = [seed], set()
        while pending:
            node = pending.pop()
            if node in members:
                continue
            members.add(node)
            pending.extend(graph[node] - members)
        visited.update(members)
        groups.append(sorted(members))
    result = copy.deepcopy(before)
    additions, audits = [], []
    next_id = max((a['id'] for a in before['instances']), default=-1) + 1
    h, w = classes.shape
    sh, sw = image.shape[:2]
    for indices in groups:
        audit = dict(fragment_ids=indices, accepted=False)
        if len(indices) < p.minimum_components:
            audit['reason'] = 'fewer_than_three_aligned_fragments'
            audits.append(audit)
            continue
        parts = sorted([fragments[i] for i in indices], key=lambda f: f['y_range'][0])
        points = np.vstack([f['points'] for f in parts])
        # Skip competing chains/overlapping depths instead of choosing a branch.
        if any(b['y_range'][0] <= a['y_range'][1]+1 for a,b in zip(parts,parts[1:])):
            audit['reason'] = 'overlapping_or_competing_fragments'
            audits.append(audit)
            continue
        span = points[-1,1] - points[0,1] + 1
        fit = np.polyfit(points[:,1], points[:,0], 1)
        residual = (np.polyval(fit, points[:,1]) - points[:,0]) / math.sqrt(1 + fit[0]**2)
        error = float(np.sqrt(np.mean(residual**2)))
        if span < p.minimum_group_span or abs(fit[0]) > p.maximum_slope or error > p.maximum_perpendicular_error:
            audit['reason'] = 'insufficient_span_or_unstable_alignment'
            audits.append(audit)
            continue
        candidate = dict(id=next_id, line_type='unknown', state_valid=False, ambiguous=False,
            reason='three_aligned_observed_paint_fragments_type_unconfirmed',
            type_scope='unknown_geometry_only', geometry_source='v6_observed_fragment_alignment',
            x_of_y=fit.tolist(), y_range=points[[0,-1],1].astype(int).tolist(), supported_rows=len(points),
            points_work=points.tolist(), points_native=(points*[sw/w,sh/h]).tolist(),
            fit_rms_work=error, mean_type_fraction=None, significant_type_rows=None,
            yellow_transverse_components=[], ego_lane_association_verified=False,
            observed_fragment_ranges=[f['y_range'] for f in parts], paint_color=parts[0]['color'],
            semantic_support_scope=f'within_{p.semantic_radius}_work_pixels_not_type_probability')
        if overlaps(candidate, before['instances']+additions):
            audit['reason'] = 'existing_path_covers_observations'
            audits.append(audit)
            continue
        additions.append(candidate)
        audit.update(accepted=True, reason='three_observed_aligned_fragments', added_path_id=next_id)
        audits.append(audit)
        next_id += 1
    result['instances'].extend(additions)
    result.update(algorithm_version='v6_observed_fragment_geometry_candidate',
        dashed_geometry_added_ids=[a['id'] for a in additions], dashed_geometry_audit=audits,
        dashed_geometry_policy=asdict(p), runtime_pair_preserved=True)
    return result
