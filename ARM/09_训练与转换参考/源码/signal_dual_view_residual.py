"""Experimental frozen-parent dual-view attribute residual. No motion authority."""
import numpy as np
import torch

from light_roi_common import logits_of
from signal_attribute_consensus import ConsensusAttributeClassifier
from signal_attribute_model import preprocess, direction_input, reject, TYPE_NAMES, DIRECTION_NAMES


def encode(reader, images):
    whole = np.stack([preprocess(im) for im in images]).transpose(0, 3, 1, 2).copy()
    shapes = [direction_input(im) for im in images]
    emission = np.stack([s[0] for s in shapes]).transpose(0, 3, 1, 2).copy()
    tensor = torch.from_numpy(np.concatenate([whole, emission])).float() / 255
    captured = []
    hook = reader.model.model[-1].linear.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone()))
    try:
        with torch.inference_mode():
            z = logits_of(reader.model(tensor)).clone()
    finally:
        hook.remove()
    if len(captured) != 1:
        raise RuntimeError('Expected one shared frozen backbone pass')
    n = len(images)
    return (torch.cat([captured[0][:n], captured[0][n:]], 1),
            torch.cat([z[:n, :6], z[n:, 6:]], 1),
            torch.tensor([s[1] for s in shapes], dtype=torch.float32).reshape(-1, 1))


def fit_transform(features, logits, quality, indices):
    # Each view has separate training-only PCA; validation never fits it.
    half = features.shape[1] // 2
    projections = []
    for part in (features[:, :half], features[:, half:]):
        mean = part[indices].mean(0)
        _, _, v = torch.linalg.svd(part[indices] - mean, full_matrices=False)
        projections.append(dict(mean=mean, components=v[:24].T.contiguous()))
    transform = dict(projections=projections, fit_indices=indices.clone())
    raw = raw_features(features, logits, quality, transform)
    transform.update(mean=raw[indices].mean(0), std=raw[indices].std(0).clamp_min(.05))
    return transform


def raw_features(features, logits, quality, transform):
    half = features.shape[1] // 2
    parts = []
    for values, p in zip((features[:, :half], features[:, half:]), transform['projections']):
        parts.append((values - p['mean']) @ p['components'])
    parts.extend([logits[:, :3].softmax(1), logits[:, 3:6].softmax(1), logits[:, 6:].softmax(1), quality])
    return torch.cat(parts, 1)


def transform_features(features, logits, quality, transform):
    return (raw_features(features, logits, quality, transform) - transform['mean']) / transform['std']


def make_head(dimensions):
    head = torch.nn.Sequential(torch.nn.Linear(dimensions, 32), torch.nn.ReLU(),
                               torch.nn.Dropout(.2), torch.nn.Linear(32, 10))
    torch.nn.init.zeros_(head[-1].weight)
    torch.nn.init.zeros_(head[-1].bias)
    return head


def output_rows(logits, quality):
    a = logits[:, :3].softmax(1).numpy()
    b = logits[:, 3:6].softmax(1).numpy()
    d = logits[:, 6:].softmax(1).numpy()
    rows = []
    for pa, pb, pd, good in zip(a, b, d, quality.flatten().tolist()):
        ta, tb = reject(pa, .85, .25), reject(pb, .85, .25)
        t = ta if ta == tb and ta >= 0 else -1
        direction = reject(pd, .80, .20) if t == 0 and good else -1
        rows.append(dict(lamp_type=TYPE_NAMES[t] if t >= 0 else 'unknown',
                         shape='round' if direction == 0 else 'arrow' if direction > 0 else 'unknown',
                         control_direction=DIRECTION_NAMES[direction] if direction > 0 else 'unknown',
                         type_head_probabilities=[pa.tolist(), pb.tolist()],
                         direction_probabilities=pd.tolist(), type_argmax=TYPE_NAMES[int(((pa+pb)/2).argmax())],
                         direction_argmax=DIRECTION_NAMES[int(pd.argmax())],
                         trusted_for_lane_association=False, physical_lane_association_verified=False))
    return rows


class DualViewResidualClassifier:
    def __init__(self, path):
        self.checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        c = self.checkpoint
        if c['format'] != 'dual_view_attribute_residual_v1':
            raise ValueError('Wrong candidate format')
        from benchmark_sequential_capture_pycharm import digest
        if digest(c['parent']) != c['parent_sha256']:
            raise RuntimeError('Frozen attribute parent changed')
        self.reader = ConsensusAttributeClassifier(c['parent'])
        self.transform = c['transform']
        self.head = make_head(len(self.transform['mean'])).eval()
        self.head.load_state_dict(c['model'])

    def predict(self, images):
        if not images:
            return []
        features, logits, quality = encode(self.reader, images)
        x = transform_features(features, logits, quality, self.transform)
        with torch.inference_mode():
            values = logits + self.head(x)
        return output_rows(values, quality)

