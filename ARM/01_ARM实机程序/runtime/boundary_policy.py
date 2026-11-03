import numpy as np

def requested_boundaries(extraction, frame, image_shape, *, center_fraction=.5):
    """Select the nearest observed path on each side of a fixed virtual path.

    Evaluate only in observed y range, reject near duplicates/crossing geometry,
    and retain untyped paths as blockers. This is not camera calibration.
    """
    h, w = image_shape[:2]
    mh, mw = extraction['mask_size'][1], extraction['mask_size'][0]
    center = center_fraction * mw
    y_near = .82 * mh
    candidates = {'left': [], 'right': []}
    for path in extraction['instances']:
        low, high = path['y_range']
        if not (low <= y_near <= high and high - low >= .08 * mh):
            continue
        observed_y = np.asarray(path['points_work'], float)[:, 1]
        if not np.any(np.abs(observed_y - y_near) <= 1):
            continue
        ys = np.linspace(max(low, .65 * mh), min(high, .88 * mh), 9)
        xs = np.polyval(path['x_of_y'], ys)
        near = float(np.polyval(path['x_of_y'], y_near))
        if not np.isfinite(xs).all() or not np.all((xs >= 0) & (xs <= mw)):
            continue
        side = 'left' if np.all(xs < center - .02 * mw) else 'right' if np.all(xs > center + .02 * mw) else None
        if side is None or abs(near - center) > .43 * mw:
            continue
        candidates[side].append((abs(near - center), near, path))
    result = dict(**frame, physical_lane_association_verified=False,
                  association_scope='experimental_nearest_observed_side_of_image_center',
                  camera_calibrated=False, camera_center_fraction=center_fraction,
                  reference_y_fraction=.82, source_size=[w, h])
    for side, options in candidates.items():
        unknown = dict(line_type='unknown', state_valid=False, simulation_association_valid=False,
                       reason='no_near_current_observed_path', path_id=None)
        options.sort(key=lambda x: x[0])
        if not options:
            result[side] = unknown
            continue
        _, x, path = options[0]
        if len(options) > 1 and abs(options[1][1] - x) < .035 * mw:
            result[side] = dict(unknown, reason='near_duplicate_boundary_ambiguity')
            continue
        if path.get('ambiguous') or not path.get('state_valid') or path['line_type'] not in ('solid', 'dashed'):
            result[side] = dict(unknown, reason=path.get('reason', 'nearest_path_type_unknown'), path_id=path['id'])
            continue
        # V4/5/6 additions remain diagnostic; their prior runtime pair was frozen.
        if path.get('type_scope') in ('unknown_geometry_only', 'observed_points_only'):
            result[side] = dict(unknown, reason='diagnostic_recovery_path_not_control_candidate', path_id=path['id'])
            continue
        result[side] = dict(line_type=path['line_type'], state_valid=True,
                            simulation_association_valid=True, path_id=path['id'],
                            x_at_reference_native=x * w / mw,
                            reason='nearest_typed_path_under_virtual_center_assumption')
    left, right = result['left'], result['right']
    if left.get('state_valid') and right.get('state_valid'):
        width = right['x_at_reference_native'] - left['x_at_reference_native']
        if not .10 * w <= width <= .8 * w:
            for side in ('left', 'right'):
                result[side].update(state_valid=False, simulation_association_valid=False,
                                    reason='implausible_virtual_lane_width')
    return result
