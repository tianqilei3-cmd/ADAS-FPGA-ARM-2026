"""Checked OpenLane JSON relations and image-plane lane queries.

The adjacency matrix uses LIST order, never numeric/sorted object IDs. Queries
come from author 3D centerlines projected with author calibration. They are
oracle inputs for a relation experiment, not predictions of our ego lane.
"""
import math

import numpy as np


def finite_array(value, shape=None):
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all() or (shape is not None and array.shape != shape):
        raise ValueError('Invalid finite array or shape')
    return array


def validate_annotation(annotation, *, allow_duplicate_ids=False):
    lanes = annotation['lane_centerline']
    elements = annotation['traffic_element']
    for records in (lanes, elements):
        identifiers = [r['id'] for r in records]
        if len(set(identifiers)) != len(identifiers) and not allow_duplicate_ids:
            raise ValueError('Duplicate annotation ID')
    for lane in lanes:
        points = finite_array(lane['points'])
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
            raise ValueError('Invalid 3D lane centerline')
    for element in elements:
        box = finite_array(element['points'], (2, 2))
        if np.any(box[1] <= box[0]):
            raise ValueError('Invalid traffic-element box')
        if type(element['category']) is not int or element['category'] not in (0, 1, 2):
            raise ValueError('Unsupported traffic-element category')
    arrays = []
    for key, shape in [('topology_lcte', (len(lanes), len(elements))),
                       ('topology_lclc', (len(lanes), len(lanes)))]:
        matrix = finite_array(annotation[key])
        # JSON represents an empty (0,k) matrix as [], which loses its width.
        if shape[0] == 0 and matrix.size == 0:
            matrix = matrix.reshape(shape)
        if matrix.shape != shape or not np.isin(matrix, [0, 1]).all():
            raise ValueError('Adjacency dimensions/values do not match list order')
        arrays.append(matrix.astype(np.uint8))
    return lanes, elements, arrays[0], arrays[1]


def project_lane(points, sensor, shape, minimum_depth=.5):
    """Project using camera-to-vehicle extrinsics, as in the author PV devkit.

    Out-of-image/behind-camera gaps terminate a segment. Nothing bridges gaps.
    Distortion is not applied, matching the author's provided image projection.
    """
    if not math.isfinite(minimum_depth) or minimum_depth <= 0:
        raise ValueError('Positive projection depth required')
    points = finite_array(points)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError('Invalid centerline')
    rotation = finite_array(sensor['extrinsic']['rotation'], (3, 3))
    translation = finite_array(sensor['extrinsic']['translation'], (3,))
    intrinsic = finite_array(sensor['intrinsic']['K'], (3, 3))
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
        raise ValueError('Invalid camera-to-vehicle rotation')
    if np.linalg.det(rotation) < .99 or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        raise ValueError('Invalid camera calibration')
    if not np.allclose(intrinsic[2], [0, 0, 1], atol=1e-6):
        raise ValueError('Unsupported intrinsic matrix')
    height, width = shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError('Positive image size required')
    camera = (points - translation) @ rotation
    projected = camera @ intrinsic.T
    usable = camera[:, 2] > minimum_depth
    xy = np.zeros((len(points), 2), dtype=np.float64)
    xy[usable] = projected[usable, :2] / projected[usable, 2:3]
    usable &= ((xy[:, 0] >= 0) & (xy[:, 0] < width) &
               (xy[:, 1] >= 0) & (xy[:, 1] < height))
    chunks, current = [], []
    for point, keep in zip(xy, usable):
        if keep:
            current.append(point)
        elif current:
            if len(current) >= 2:
                chunks.append(np.asarray(current))
            current = []
    if len(current) >= 2:
        chunks.append(np.asarray(current))
    return chunks


def resample_polyline(points, count=16):
    points = finite_array(points)
    if points.ndim != 2 or points.shape[1] != 2 or count < 2:
        raise ValueError('Invalid image polyline')
    if len(points) < 2:
        raise ValueError('Polyline needs two points')
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-6]
    points = points[keep]
    if len(points) < 2:
        raise ValueError('Zero-length polyline')
    distance = np.r_[0, np.linalg.norm(np.diff(points, axis=0), axis=1).cumsum()]
    query = np.linspace(0, distance[-1], count)
    return np.stack([np.interp(query, distance, points[:, axis]) for axis in (0, 1)], 1)


def image_query(lane, sensor, shape):
    """Select one sufficiently long visible segment, independently of labels.

    A forward-oriented author lane is required. This does not determine whether
    that lane belongs to our vehicle. Training also keeps adjacent forward lanes.
    """
    points = finite_array(lane['points'])
    # Vehicle coordinates: X forward. Preserve authored direction, not lane ID.
    if points[-1, 0] - points[0, 0] <= 3:
        return None, 'not_forward_oriented'
    segments = project_lane(points, sensor, shape)
    if not segments:
        return None, 'no_visible_segment'
    segment = max(segments, key=lambda p: np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    height, width = shape[:2]
    if np.ptp(segment[:, 1]) < .06 * height:
        return None, 'short_visible_vertical_extent'
    query = resample_polyline(segment) / [width, height]
    return query, 'author_projected_forward_centerline'


def pair_geometry(query, box, shape):
    """Image-only numeric features; no IDs, source pose, color or relation truth."""
    query = finite_array(query, (16, 2))
    if np.any(query < 0) or np.any(query > 1):
        raise ValueError('Normalized in-image query required')
    height, width = shape[:2]
    a, b, c, d = finite_array(box, (4,)) / [width, height, width, height]
    if not (0 <= a < c <= 1 and 0 <= b < d <= 1):
        raise ValueError('In-image light box required')
    cx, cy = (a + c) / 2, (b + d) / 2
    near, far = query[0], query[-1]
    relative = [cx - near[0], cy - near[1], cx - far[0], cy - far[1],
                float(np.linalg.norm(query - [cx, cy], axis=1).min())]
    return np.asarray([*query.reshape(-1), a, b, c, d, cx, cy, c-a, d-b,
                       *relative], dtype=np.float32)
