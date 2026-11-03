"""Current-frame path cleanup, paint-supported groups and nested brackets.

Grouping is a marking hypothesis. Camera center is assumed, not calibrated;
no path or bracket can set physical ego association or authorize a maneuver.
"""
from itertools import combinations

import cv2
import numpy as np

import lane_boundary_instances as base


def trace_without_stale_fragments(mask, blocked, policy):
    return base._trace(mask, blocked, policy, suppress_stale_fragments=True)


def paint_support(instance, image, shape):
    """Color evidence around observed centers; absence is not a dashed gap."""
    h, w = shape
    hsv = cv2.cvtColor(cv2.resize(image, (w, h)), cv2.COLOR_BGR2HSV)
    yellow = ((hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 40)
              & (hsv[:, :, 1] >= 80) & (hsv[:, :, 2] >= 105))
    white = (hsv[:, :, 1] <= 90) & (hsv[:, :, 2] >= 160)
    hits = {'yellow': [], 'white': []}
    for x, y in instance['points_work']:
        x, y = round(x), int(y)
        for name, pixels in (('yellow', yellow), ('white', white)):
            hits[name].append(bool(pixels[y, max(0, x - 4):min(w, x + 5)].any()))
    fractions = {name: float(np.mean(values)) for name, values in hits.items()}
    color = max(fractions, key=fractions.get)
    return dict(color=color if fractions[color] >= .55 else 'unknown', row_fractions=fractions)


def group_solid_markings(instances, image, shape, center_fraction):
    """Only unique, close, same-color, solid pairs become a group hypothesis."""
    h, w = shape
    center = center_fraction * w
    eligible = [p for p in instances if p['line_type'] == 'solid' and not p['ambiguous']]
    supports = {p['id']: paint_support(p, image, shape) for p in eligible}
    possibilities = []
    for a, b in combinations(eligible, 2):
        sa, sb = supports[a['id']], supports[b['id']]
        if sa['color'] == 'unknown' or sa['color'] != sb['color']:
            continue
        top, bottom = max(a['y_range'][0], b['y_range'][0]), min(a['y_range'][1], b['y_range'][1])
        min_span = min(a['y_range'][1] - a['y_range'][0], b['y_range'][1] - b['y_range'][0])
        if bottom - top < max(14, .60 * min_span) or bottom < .70 * h:
            continue
        ys = np.linspace(top, bottom, 9)
        ax, bx = np.polyval(a['x_of_y'], ys), np.polyval(b['x_of_y'], ys)
        delta = bx - ax
        if not (np.all(delta > 0) or np.all(delta < 0)):
            continue
        distance = np.abs(delta)
        if distance.min() < 3 or distance.max() > .065 * w or distance[-1] < .75 * distance[0]:
            continue
        if not (np.all((ax < center - .02 * w) & (bx < center - .02 * w)) or
                np.all((ax > center + .02 * w) & (bx > center + .02 * w))):
            continue
        angle_a = np.arctan(np.polyval(np.polyder(a['x_of_y']), ys))
        angle_b = np.arctan(np.polyval(np.polyder(b['x_of_y']), ys))
        if np.abs(angle_a - angle_b).max() > np.deg2rad(15):
            continue
        representative = a if abs(ax[-1] - center) < abs(bx[-1] - center) else b
        possibilities.append(dict(member_ids=[a['id'], b['id']], representative_id=representative['id'],
             color=sa['color'], common_y=[int(top), int(bottom)], separation_work=distance.tolist(),
             marking_group_verified=False, reason='close_parallel_solid_paths_with_same_color_paint'))
    # Do not transitively collapse three nearby lanes or competing groupings.
    groups = [g for g in possibilities if all(sum(i in q['member_ids'] for q in possibilities) == 1
                                                for i in g['member_ids'])]
    removed = {i for g in groups for i in g['member_ids'] if i != g['representative_id']}
    return [p for p in instances if p['id'] not in removed], groups, supports


def propose_inner_pair(instances, shape, center_fraction):
    proposal = base.propose_pair(instances, shape, center_fraction)
    proposal['selection_method'] = 'unique_bracket'
    if proposal['reason'] != 'multiple_image_center_pairs':
        return proposal
    paths = {p['id']: p for p in instances}
    alternatives = proposal['alternatives']
    w = shape[1]
    dominated = set()
    for i, inner in enumerate(alternatives):
        for j, outer in enumerate(alternatives):
            if i == j:
                continue
            top = max(inner['common_y'][0], outer['common_y'][0])
            bottom = min(inner['common_y'][1], outer['common_y'][1])
            if bottom - top < 12:
                continue
            ys = np.linspace(top, bottom, 9)
            il = np.polyval(paths[inner['left_id']]['x_of_y'], ys)
            ir = np.polyval(paths[inner['right_id']]['x_of_y'], ys)
            ol = np.polyval(paths[outer['left_id']]['x_of_y'], ys)
            oright = np.polyval(paths[outer['right_id']]['x_of_y'], ys)
            inside = np.all(il >= ol - 1) and np.all(ir <= oright + 1)
            strict = np.all(il - ol >= .02 * w) or np.all(oright - ir >= .02 * w)
            if inside and strict:
                dominated.add(j)
    remaining = [p for i, p in enumerate(alternatives) if i not in dominated]
    if len(remaining) != 1:
        proposal['selection_method'] = 'ambiguous_non_nested_or_near_duplicate_brackets'
        return proposal
    selected = remaining[0]
    proposal.update(state='geometric_candidate_unverified', reason='unique_innermost_supported_bracket',
                    left_id=selected['left_id'], right_id=selected['right_id'],
                    common_y=selected['common_y'], near_width=selected['near_width'],
                    selection_method='strict_nesting_in_common_observed_range')
    return proposal


def extract_instances(mask, image, occluders, policy=None):
    policy = policy or base.InstancePolicy()
    result = base.extract_instances(mask, image, occluders, policy, trace_function=trace_without_stale_fragments)
    shape = tuple(result['mask_size'][::-1])
    representatives, groups, supports = group_solid_markings(result['instances'], image, shape,
                                                            policy.camera_center_fraction)
    result['pair'] = propose_inner_pair(representatives, shape, policy.camera_center_fraction)
    result['marking_groups'] = groups
    result['paint_support'] = supports
    result['representative_path_ids'] = [p['id'] for p in representatives]
    result['algorithm_version'] = 'v2_stale_fragments_paint_groups_nested_pairs'
    # Type conflict rejection remains unchanged. A group is never a physical identity proof.
    return result
