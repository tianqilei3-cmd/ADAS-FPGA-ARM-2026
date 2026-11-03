"""PyCharm: train typed lane-boundary segmentation from original BDD splits.

CPU defaults: 800 train /160 val images, 512x288, batch8, frozen local YOLOv8n
encoder,12 epochs. Completed plans are reused; last full epoch resumes after
interruption. Logs, optimizer, RNG state and source validation are retained.
No official video, board access, dependency changes or signal-weight updates.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'runs/adas_config'))

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from lane_boundary_segmentation import BoundarySegmenter, NAMES, boundary_loss, image_tensor
from prepare_lane_boundary_segmentation import OUT, prepare, save


class BoundaryDataset(Dataset):
    def __init__(self, folder, rows):
        self.folder, self.rows = folder, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = cv2.imdecode(np.fromfile(self.folder / row['image'], np.uint8), cv2.IMREAD_COLOR)
        target = cv2.imdecode(np.fromfile(self.folder / row['mask'], np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None or target is None or image.shape[:2] != target.shape:
            raise RuntimeError(f'Invalid prepared pair {row["image"]}')
        return image_tensor([image])[0], torch.from_numpy(target.astype(np.int64))


def confusion_metrics(matrix):
    matrix = np.asarray(matrix, dtype=np.int64)
    true_positive = np.diag(matrix)
    union = matrix.sum(0) + matrix.sum(1) - true_positive
    iou = true_positive / np.maximum(1, union)
    return dict(confusion_matrix=matrix.tolist(), iou=dict(zip(NAMES, map(float, iou))),
                foreground_mean_iou=float(iou[1:].mean()))


@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    matrix = np.zeros((3, 3), np.int64)
    near_prediction, prediction_count, near_truth, truth_count = 0, 0, 0, 0
    kernel = np.ones((7, 7), np.uint8)
    for images, target in loader:
        prediction = model(images).argmax(1).numpy()
        truth = target.numpy()
        valid = truth != 255
        matrix += np.bincount((3 * truth[valid] + prediction[valid]).ravel(), minlength=9).reshape(3, 3)
        for actual, predicted, usable in zip(truth, prediction, valid):
            a = ((actual > 0) & usable).astype(np.uint8)
            p = ((predicted > 0) & usable).astype(np.uint8)
            near_prediction += int(((cv2.dilate(a, kernel) > 0) & (p > 0)).sum())
            near_truth += int(((cv2.dilate(p, kernel) > 0) & (a > 0)).sum())
            prediction_count += int(p.sum())
            truth_count += int(a.sum())
    return dict(**confusion_metrics(matrix), localization_tolerance_input_pixels=3,
                tolerant_binary_precision=near_prediction / max(1, prediction_count),
                tolerant_binary_recall=near_truth / max(1, truth_count),
                scope='Prepared original BDD val corridor pixels; not official-road, exact paint or driving accuracy')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--train-images', type=int, default=800)
    parser.add_argument('--val-images', type=int, default=160)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--warmstart', type=Path, help='Initialize a NEW data/epoch experiment from a compatible best.pt')
    args = parser.parse_args()
    if min(args.epochs, args.train_images, args.val_images, args.batch) < 1:
        parser.error('Counts must be positive')
    torch.set_num_threads(8)
    cv2.setNumThreads(1)
    torch.manual_seed(20261017)
    np.random.seed(20261017)
    folder, manifest = prepare(args.train_images, args.val_images)
    save(OUT / 'latest_data.json', dict(dataset=str(folder), manifest=str(folder / 'manifest.json')))
    initialization = ROOT / 'yolov8n-seg.pt'
    plan = dict(dataset=str(folder), data_fingerprint=manifest['fingerprint'], epochs=args.epochs, batch=args.batch,
                initialization=str(initialization), initialization_sha256=hashlib.sha256(initialization.read_bytes()).hexdigest(),
                architecture='frozen_yolov8n_encoder_FPN32_v1', size=[512, 288], names=NAMES,
                seed=20261017, optimizer='AdamW', learning_rate=.001, weight_decay=.001,
                model_code_sha256=hashlib.sha256((ROOT / 'lane_boundary_segmentation.py').read_bytes()).hexdigest())
    if args.warmstart is not None:
        plan['warmstart'] = str(args.warmstart.resolve())
        plan['warmstart_sha256'] = hashlib.sha256(args.warmstart.read_bytes()).hexdigest()
    key = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:12]
    run = ROOT / 'runs/adas' / ('lane_boundary_seg_' + key)
    weights = run / 'weights'
    weights.mkdir(parents=True, exist_ok=True)
    status_path = run / 'status.json'
    pointer = dict(run=str(run), best=str(weights / 'best.pt'), last=str(weights / 'last.pt'),
                   status=str(status_path), dataset=str(folder), selected_for_deployment=False,
                   purpose='typed_boundary_corridors')
    save(OUT / 'latest_training.json', pointer)
    if status_path.exists() and json.loads(status_path.read_text(encoding='utf-8')).get('state') == 'completed':
        print('BOUNDARY_TRAINING_ALREADY_COMPLETED', run, flush=True)
        return
    loaders = {split: DataLoader(BoundaryDataset(folder, [row for row in manifest['records'] if row['split'] == split]),
                                batch_size=args.batch, shuffle=split == 'train', num_workers=0)
               for split in ('train', 'val')}
    model = BoundarySegmenter(initialization)
    if args.warmstart is not None:
        warm = torch.load(args.warmstart, map_location='cpu', weights_only=False)
        if warm.get('purpose') != 'typed_boundary_corridors' or warm.get('names') != NAMES:
            raise ValueError('Incompatible warm-start boundary model')
        model.load_state_dict(warm['model'])
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.001, weight_decay=.001)
    start, best_score, history, previous_seconds = 0, -1., [], 0.
    if (weights / 'last.pt').exists():
        checkpoint = torch.load(weights / 'last.pt', map_location='cpu', weights_only=False)
        if checkpoint['plan'] != plan:
            raise RuntimeError('Resume plan mismatch')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        torch.set_rng_state(checkpoint['rng_state'])
        start, best_score, history = checkpoint['epoch'], checkpoint['best_score'], checkpoint['history']
        previous_seconds = checkpoint['elapsed_training_seconds']
        print('BOUNDARY_RESUME', start, '/', args.epochs, flush=True)
    save(run / 'plan.json', plan)
    started = time.perf_counter()
    save(status_path, dict(state='running', plan=plan, last_completed_epoch=start))
    try:
        for epoch in range(start + 1, args.epochs + 1):
            model.train()
            loss_sum, seen = 0., 0
            for step, (images, target) in enumerate(loaders['train'], 1):
                mirrored = torch.rand(len(images)) < .5
                images[mirrored] = images[mirrored].flip(-1)
                target[mirrored] = target[mirrored].flip(-1)
                images = (images * (.85 + .30 * torch.rand(len(images), 1, 1, 1))).clamp(0, 1)
                optimizer.zero_grad(set_to_none=True)
                loss = boundary_loss(model(images), target)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite boundary loss')
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(images)
                seen += len(images)
                if step % 25 == 0:
                    print('BOUNDARY_BATCH', epoch, step, '/', len(loaders['train']), flush=True)
            metrics = evaluate(model, loaders['val'])
            improved = metrics['foreground_mean_iou'] > best_score
            best_score = max(best_score, metrics['foreground_mean_iou'])
            history.append(dict(epoch=epoch, loss=loss_sum / seen, metrics=metrics,
                                elapsed_training_seconds=previous_seconds + time.perf_counter() - started))
            checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                              best_score=best_score, history=history, rng_state=torch.get_rng_state(), names=NAMES,
                              purpose='typed_boundary_corridors', plan=plan,
                              elapsed_training_seconds=previous_seconds + time.perf_counter() - started)
            torch.save(checkpoint, weights / 'last.pt')
            if improved:
                torch.save(checkpoint, weights / 'best.pt')
            save(run / 'history.json', history)
            save(status_path, dict(state='running', plan=plan, last_completed_epoch=epoch))
            print('BOUNDARY_EPOCH', epoch, '/', args.epochs, 'loss', round(loss_sum / seen, 4),
                  'val_foreground_mIoU', round(metrics['foreground_mean_iou'], 5),
                  'per_class', metrics['iou'], flush=True)
    except BaseException as exc:
        save(status_path, dict(state='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                               error=repr(exc), resume_available=(weights / 'last.pt').exists(), plan=plan))
        raise
    best = torch.load(weights / 'best.pt', map_location='cpu', weights_only=False)
    result = dict(state='completed', epochs=history[-1]['epoch'], best_epoch=best['epoch'],
                  source_val=history[best['epoch'] - 1]['metrics'],
                  elapsed_training_seconds=previous_seconds + time.perf_counter() - started,
                  best=str(weights / 'best.pt'), best_sha256=hashlib.sha256((weights / 'best.pt').read_bytes()).hexdigest(),
                  completed_at=datetime.now().isoformat(), plan=plan, selected_for_deployment=False)
    save(status_path, result)
    print('BOUNDARY_TRAINING_COMPLETED', run, result, flush=True)


class Tee:
    def __init__(self, console, logfile):
        self.console, self.logfile = console, logfile

    def write(self, text):
        self.console.write(text)
        self.logfile.write(text)
        self.flush()

    def flush(self):
        self.console.flush()
        self.logfile.flush()


if __name__ == '__main__':
    logs = OUT / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.log')
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with log_path.open('w', encoding='utf-8', buffering=1) as stream:
        sys.stdout, sys.stderr = Tee(original_stdout, stream), Tee(original_stderr, stream)
        try:
            print('PERSISTENT_LOG', log_path, flush=True)
            main()
        except SystemExit:
            raise
        except BaseException:
            traceback.print_exc()
            raise SystemExit(1)
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr
