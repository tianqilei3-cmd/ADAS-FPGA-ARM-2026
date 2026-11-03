"""Recover observed parallel solid paint paths from a rejected wide component.

The original filtered mask, V3 instances and runtime pair remain unchanged.
Recovery requires two distinct, paint-supported parallel solid paths inside
the same rejected semantic component. It does not restore the whole blob.
"""
import copy
from dataclasses import asdict, dataclass
import math

import cv2
import numpy as np

import lane_boundary_instances as base
import lane_boundary_instances_v2 as v2
import lane_boundary_segments_v3 as v3
from lane_boundary_postprocess import filter_regions


@dataclass(frozen=True)
class RecoveryPolicy:
    minimum_probability: float = .85
    minimum_paint_contrast: int = 20
    horizontal_background_width: int = 31
    maximum_paint_run_width: int = 12
    minimum_rows: int = 24
    minimum_span: int = 32
    minimum_density: float = .85
    maximum_fit_error: float = 2.
    minimum_common_rows: int = 24

    def __post_init__(self):
        for name in ('minimum_paint_contrast', 'maximum_paint_run_width', 'minimum_rows',
                     'minimum_span', 'minimum_common_rows'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError('Invalid recovery policy: ' + name)
        if (type(self.horizontal_background_width) is not int
                or self.horizontal_background_width < 3 or self.horizontal_background_width % 2 != 1):
            raise ValueError('Background width must be an odd integer')
        if not (.5 < self.minimum_probability <= 1 and 0 < self.minimum_density <= 1
                and math.isfinite(self.maximum_fit_error) and self.maximum_fit_error > 0):
            raise ValueError('Invalid recovery confidence or geometry policy')


def prepare_evidence(probabilities, source_shape, occluders, policy=None):
    """Retain exact old filtering; cache two small masks instead of full logits."""
    policy = policy or RecoveryPolicy()
    filtered, _ = filter_regions(probabilities, source_shape, occluders)
    p = np.clip(np.asarray(probabilities), 0., 1.)
    classes = p.argmax(0).astype(np.uint8)
    classes[p.max(0) < .50] = 0
    h, w = classes.shape
    sh, sw = source_shape[:2]
    classes[:round(h * .40)] = 0
    classes[round(h * .94):] = 0
    for obj in occluders:
        if obj.get('name') not in ('vehicle', 'person', 'two_wheeler') or obj.get('confidence', 0) < .35:
            continue
        x1, y1, x2, y2 = obj['box']  # Already validated by filter_regions.
        if x2 <= 0 or y2 <= 0 or x1 >= sw or y1 >= sh:
            continue
        a, b = max(0, int(x1 * w / sw) - 1), max(0, int(y1 * h / sh) - 1)
        c, d = min(w, round(x2 * w / sw) + 2), min(h, round(y2 * h / sh) + 2)
        classes[b:d, a:c] = 0
    strong_solid = ((classes == 1) & (p[1] >= policy.minimum_probability)).astype(np.uint8)
    return filtered, classes, strong_solid


def _paint_evidence(image, shape, policy):
    h, w = shape
    hsv = cv2.cvtColor(cv2.resize(image, (w, h)), cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] <= 90) & (hsv[:, :, 2] >= 160)
    yellow = ((hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 40)
              & (hsv[:, :, 1] >= 80) & (hsv[:, :, 2] >= 105))
    contrast = cv2.morphologyEx(hsv[:, :, 2], cv2.MORPH_TOPHAT,
                              np.ones((1, policy.horizontal_background_width), np.uint8))
    return (white | yellow) & (contrast >= policy.minimum_paint_contrast)


def _overlaps_old_path(candidate, original):
    points = np.asarray(candidate['points_work'])
    for old in original:
        common = points[(points[:, 1] >= old['y_range'][0]) & (points[:, 1] <= old['y_range'][1])]
        if len(common) < 14:
            continue
        distance = np.abs(common[:, 0] - np.polyval(old['x_of_y'], common[:, 1]))
        if np.mean(distance <= 3.) >= .75:
            return True
    return False


def extract_from_evidence(filtered, classes, strong_solid, image, occluders,
                          policy=None, recovery_policy=None):
    """Pure current-frame extraction; cached evidence must come from prepare_evidence."""
    p = policy or base.InstancePolicy()
    recovery = recovery_policy or RecoveryPolicy()
    result = v3.extract_instances(filtered, image, occluders, p)
    classes = np.asarray(classes)
    strong_solid = np.asarray(strong_solid)
    if (classes.dtype != np.uint8 or classes.shape != filtered.shape
            or not np.isin(classes, [0, 1, 2]).all()
            or strong_solid.shape != filtered.shape or not np.isin(strong_solid, [0, 1]).all()
            or np.any((strong_solid != 0) & (classes != 1))):
        raise ValueError('Invalid cached semantic/strong-solid evidence')
    # Do not trust caller-supplied cached pixels inside an excluded image area.
    h, w = classes.shape
    eligible = classes == 1
    eligible[:round(h * .40)] = False
    eligible[round(h * .94):] = False
    sh, sw = image.shape[:2]
    for obj in occluders:
        if obj.get('name') not in ('vehicle', 'person', 'two_wheeler') or obj.get('confidence', 0) < .35:
            continue
        x1, y1, x2, y2 = obj['box']  # V3 has already validated these boxes.
        if x2 <= 0 or y2 <= 0 or x1 >= sw or y1 >= sh:
            continue
        a, b = max(0, int(x1 * w / sw) - 1), max(0, int(y1 * h / sh) - 1)
        c, d = min(w, round(x2 * w / sw) + 2), min(h, round(y2 * h / sh) + 2)
        eligible[b:d, a:c] = False
    count, labels, stats, _ = cv2.connectedComponentsWithStats(eligible.astype(np.uint8), 8)
    paint = _paint_evidence(image, classes.shape, recovery)
    strict = base.InstancePolicy(max_gap_rows=1, max_run_width=recovery.maximum_paint_run_width,
        min_rows=recovery.minimum_rows, min_span=recovery.minimum_span,
        max_dx_dy=p.max_dx_dy, max_fit_error=recovery.maximum_fit_error,
        min_type_fraction=p.min_type_fraction, conflict_row_fraction=p.conflict_row_fraction,
        min_conflict_rows=p.min_conflict_rows, camera_center_fraction=p.camera_center_fraction)
    originals = copy.deepcopy(result['instances'])
    additions, audits = [], []
    next_id = max((i['id'] for i in originals), default=-1) + 1
    for component in range(1, count):
        x, y, cw, ch, area = map(int, stats[component])
        if area < 20 or ch < 12 or area / ch <= 24:
            continue
        selected = labels == component
        # This proposal is specifically for a component removed by the old width gate.
        if np.any(filtered[selected]):
            continue
        support = (selected & (strong_solid != 0) & paint).astype(np.uint8)
        n, stripes, stripe_stats, _ = cv2.connectedComponentsWithStats(support, 8)
        candidates = []
        audit = dict(semantic_component=component, box_work=[x, y, cw, ch],
            semantic_pixels=area, rejected_average_width=area / ch,
            strong_paint_pixels=int(support.sum()), paint_components=n - 1,
            accepted=False, reason='no_unique_parallel_paint_pair')
        for stripe in range(1, n):
            if stripe_stats[stripe, 3] < recovery.minimum_span:
                continue
            observed = (stripes == stripe).astype(np.uint8)
            traced = base.extract_instances(observed, image, occluders, strict,
                                           trace_function=v2.trace_without_stale_fragments)
            if len(traced['instances']) != 1:
                continue
            path = traced['instances'][0]
            span = path['y_range'][1] - path['y_range'][0] + 1
            xy = np.asarray(path['points_work']).round().astype(int)
            if (path['line_type'] != 'solid' or path['ambiguous']
                    or path['supported_rows'] / span < recovery.minimum_density
                    or not np.all(strong_solid[xy[:, 1], xy[:, 0]])):
                continue
            path = copy.deepcopy(path)
            path.update(id=len(candidates), geometry_origin='observed_paint_in_rejected_wide_solid_component',
                type_scope='observed_points_only', source_semantic_component=component,
                source_paint_component=stripe, minimum_solid_probability=recovery.minimum_probability)
            candidates.append(path)
        audit['qualified_paint_paths'] = len(candidates)
        if len(candidates) != 2:
            audits.append(audit)
            continue
        _, groups, _ = v2.group_solid_markings(candidates, image, classes.shape, p.camera_center_fraction)
        if len(groups) != 1 or len(groups[0]['member_ids']) != 2:
            audits.append(audit)
            continue
        common = groups[0]['common_y']
        if common[1] - common[0] + 1 < recovery.minimum_common_rows:
            audit['reason'] = 'insufficient_parallel_observation'
            audits.append(audit)
            continue
        if any(_overlaps_old_path(path, originals + additions) for path in candidates):
            audit['reason'] = 'duplicate_existing_path'
            audits.append(audit)
            continue
        accepted_ids = []
        id_map = {}
        for path in candidates:
            id_map[path['id']] = next_id
            path['id'] = next_id
            path['reason'] = 'parallel_paint_within_high_confidence_semantic_solid'
            next_id += 1
            accepted_ids.append(path['id'])
            additions.append(path)
        group = copy.deepcopy(groups[0])
        group['member_ids'] = [id_map[i] for i in group['member_ids']]
        group['representative_id'] = id_map[group['representative_id']]
        audit.update(accepted=True, reason='two_supported_parallel_solid_paint_paths',
                     added_path_ids=accepted_ids, parallel_group=group)
        audits.append(audit)
    result['instances'].extend(additions)
    result.update(wide_component_recovery=audits, recovered_path_ids=[a['id'] for a in additions],
                  recovery_policy=asdict(recovery), runtime_pair_preserved=True,
                  algorithm_version='v4_parallel_paint_from_rejected_wide_solid')
    # New geometry is diagnostic only. Keep all previous runtime pairing decisions.
    return result


def extract_instances(probabilities, image, occluders, policy=None, recovery_policy=None):
    recovery_policy = recovery_policy or RecoveryPolicy()
    filtered, classes, strong = prepare_evidence(probabilities, image.shape, occluders, recovery_policy)
    return extract_from_evidence(filtered, classes, strong, image, occluders, policy, recovery_policy)
