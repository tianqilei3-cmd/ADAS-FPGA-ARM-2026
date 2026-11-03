"""CPU capture-quality fine-tunes, isolated from the deployed ARM release.

PyCharm: run without arguments to train signal then navigation (8 epochs each).
Re-running resumes an interrupted epoch checkpoint or skips a completed task.
No board access, downloads, package installation, or deployment occurs here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["YOLO_AUTOINSTALL"] = "false"
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/arm_signal_navigation_20261029"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runs/adas_config"))

import cv2
import numpy as np
import psutil
import torch
from ultralytics import YOLO
from ultralytics.data.build import build_dataloader
from ultralytics.models.yolo.detect.train import DetectionTrainer

sys.path.insert(0, str(ROOT / "arm_unified_20261028"))
from vision_common import preprocess, decoded_output, decode
from training_memory_monitor import memory_snapshot, start_monitor


SPECS = {
    "signal": dict(
        shape=[256, 640], names=["red", "yellow", "green", "off"],
        initial="runs/basic_finish70_20261028/lamp_capture_aligned/trial/weights/last.pt",
        manifest="runs/basic_finish70_20261028/lamp_capture_aligned/dataset_manifest.json",
        lr0=0.00005, nms_iou=0.7,
    ),
    "navigation": dict(
        shape=[288, 512], names=["primary_instruction_card", "junction_instruction_header"],
        initial="runs/adas/navigation_card_af981433c999/weights/best.pt",
        manifest="datasets/adas/prepared/navigation_card_v1/63ebb512db00/manifest.json",
        lr0=0.0001, nms_iou=0.5,
    ),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def targets(path, shape):
    h, w = shape[:2]
    result = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        cls, cx, cy, bw, bh = map(float, line.split())
        result.append(dict(class_id=int(cls), box=[(cx-bw/2)*w, (cy-bh/2)*h,
                                                  (cx+bw/2)*w, (cy+bh/2)*h]))
    return result


def capture_view(image, variant):
    """Deterministic photometric views; no box/semantic changes or hue shifts."""
    if variant == "clean":
        return image.copy()
    if variant not in ("capture_mild", "capture_hard"):
        raise ValueError(variant)
    hard = variant == "capture_hard"
    h, w = image.shape[:2]
    ratio = 0.65 if hard else 0.82
    small = cv2.resize(image, (max(2, round(w*ratio)), max(2, round(h*ratio))),
                       interpolation=cv2.INTER_AREA)
    image = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if hard:
        image = cv2.GaussianBlur(image, (3, 3), 0.45)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 60 if hard else 78])
    if not ok:
        raise ValueError("JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("JPEG decoding failed")
    return decoded


def source_records(task):
    spec = SPECS[task]
    source_path = ROOT / spec["manifest"]
    manifest = load(source_path)
    records = []
    for row in manifest["records"]:
        if task == "signal":
            # Use each original prepared crop once; regenerate quality views below.
            if row["scale_simulation"] != 1.0:
                continue
            image, label = Path(row["image"]), Path(row["label"])
            group = row["source_pixel_sha256"]
            if row["split"] == "train":
                assert not any(x in str(row.get("video", "")) for x in ("112454", "113421", "113539"))
        else:
            image, label = source_path.parent / row["image"], source_path.parent / row["label_file"]
            assert digest(image) == row["sha256"] and digest(label) == row["label_sha256"]
            group = row["source_sha256"]
        records.append(dict(image=str(image), label=str(label), split=row["split"],
                            group=group, image_sha256=digest(image), label_sha256=digest(label)))
    groups = {s: {r["group"] for r in records if r["split"] == s} for s in ("train", "val")}
    assert groups["train"] and groups["val"] and not groups["train"] & groups["val"]
    return records, {s: len(v) for s, v in groups.items()}


def prepare(task):
    spec = SPECS[task]
    folder = OUT / task
    records, groups = source_records(task)
    config = dict(task=task, source_manifest_sha256=digest(ROOT / spec["manifest"]),
                  source_groups=groups, source_records=records,
                  train_views=["clean", "capture_mild", "capture_hard"] if task == "signal"
                  else ["clean", "capture_hard"], val_views=["clean"],
                  existing_train_val_roles_preserved=True, source_independent=False,
                  new_independent_sources=0, code_sha256=digest(__file__))
    path = folder / "dataset_manifest.json"
    if path.exists():
        previous = load(path)
        assert previous["config"] == config, "Dataset/source/code changed; use a new run directory"
        for row in previous["records"]:
            assert digest(row["image"]) == row["image_sha256"]
            assert digest(row["label"]) == row["label_sha256"]
        return previous
    prepared, lists = [], {s: [] for s in ("train", "val")}
    for index, row in enumerate(records):
        image = read_image(row["image"])
        views = config["train_views"] if row["split"] == "train" else ["clean"]
        for view in views:
            name = f"{index:04d}_{view}"
            target_image = folder / "dataset/images" / row["split"] / (name + ".png")
            target_label = folder / "dataset/labels" / row["split"] / (name + ".txt")
            target_image.parent.mkdir(parents=True, exist_ok=True)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            ok, encoded = cv2.imencode(".png", capture_view(image, view))
            assert ok
            encoded.tofile(str(target_image))
            shutil.copyfile(row["label"], target_label)
            # Extra green weighting is deliberately absent: retain all colors/negatives.
            lists[row["split"]].append(target_image.as_posix())
            prepared.append(dict(source=row, view=view, split=row["split"], group=row["group"],
                                 image=str(target_image), label=str(target_label),
                                 image_sha256=digest(target_image), label_sha256=digest(target_label)))
    dataset = folder / "dataset"
    for split, paths in lists.items():
        (dataset / f"{split}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    yaml = "path: " + dataset.as_posix() + "\ntrain: train.txt\nval: val.txt\nnames:\n"
    yaml += "".join(f"  {i}: {name}\n" for i, name in enumerate(spec["names"]))
    (dataset / "data.yaml").write_text(yaml, encoding="utf-8")
    result = dict(config=config, records=prepared, counts={s: len(v) for s, v in lists.items()},
                  data_yaml_sha256=digest(dataset / "data.yaml"))
    save(path, result)
    return result


class FixedShapeTrainer(DetectionTrainer):
    shape = None

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        assert rank == -1 and self.shape is not None
        dataset = self.build_dataset(dataset_path, mode, batch_size)
        assert dataset.rect
        dataset.batch_shapes[:] = np.asarray(self.shape, dtype=int)
        return build_dataloader(dataset, batch_size, self.args.workers, mode == "train", rank)

    def preprocess_batch(self, batch):
        result = super().preprocess_batch(batch)
        assert list(result["img"].shape[2:]) == self.shape
        return result


class RawHeads(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        _, features = self.model(x)
        return tuple(item for feature in features
                     for item in (feature[:, :64], feature[:, 64:].sigmoid()))


def iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    inter = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    union = np.maximum(0, a[2:]-a[:2]).prod() + np.maximum(0, b[2:]-b[:2]).prod() - inter
    return float(inter / max(1e-9, union))


def metrics(predictions, truth, names):
    totals = Counter(truth=len(truth), predictions=len(predictions))
    options = sorted([(iou(p["box"], t["box"]), pi, ti)
                      for pi, p in enumerate(predictions) for ti, t in enumerate(truth)
                      if p["class_id"] == t["class_id"]], reverse=True)
    used_p, used_t = set(), set()
    for overlap, pi, ti in options:
        if overlap < 0.5:
            break
        if pi in used_p or ti in used_t:
            continue
        used_p.add(pi)
        used_t.add(ti)
        totals["tp"] += 1
    totals.update(fp=len(predictions)-len(used_p), fn=len(truth)-len(used_t))
    for cls, name in enumerate(names):
        totals[name + "_truth"] = sum(t["class_id"] == cls for t in truth)
        totals[name + "_predictions"] = sum(p["class_id"] == cls for p in predictions)
        totals[name + "_tp"] = sum(truth[ti]["class_id"] == cls for ti in used_t)
    if names[0] == "red":
        totals["red_as_green_overlaps"] = sum(
            t["class_id"] == 0 and any(p["class_id"] == 2 and iou(p["box"], t["box"]) >= 0.5
                                       for p in predictions) for t in truth)
    return dict(totals)


def summarize(counts):
    counts = dict(counts)
    tp, fp, fn = (counts.get(k, 0) for k in ("tp", "fp", "fn"))
    counts.update(precision=tp/max(1, tp+fp), recall=tp/max(1, tp+fn),
                  f1=2*tp/max(1, 2*tp+fp+fn))
    return counts


def evaluate(task, weights, label, dataset):
    spec, folder = SPECS[task], OUT / task
    destination = folder / f"evaluation_{label}.json"
    fingerprint = dict(weights=str(weights), weights_sha256=digest(weights), shape=spec["shape"],
                       confidence=0.25, nms_iou=spec["nms_iou"], match_iou=0.5,
                       dataset_sha256=digest(folder / "dataset_manifest.json"), code_sha256=digest(__file__))
    if destination.exists():
        report = load(destination)
        assert report["config"] == fingerprint
        return report
    model = RawHeads(YOLO(str(weights)).model.eval()).eval()
    rows = [r for r in dataset["records"] if r["split"] == "val"]
    totals = {view: Counter() for view in ("clean", "capture_hard")}
    observations = []
    for index, row in enumerate(rows):
        original = read_image(row["image"])
        expected = targets(row["label"], original.shape)
        for view in totals:
            image = capture_view(original, view)
            rgb, geometry = preprocess(image, spec["shape"])
            started = time.monotonic()
            with torch.inference_mode():
                heads = model(torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255)
            output = decoded_output([x.numpy() for x in heads], spec["shape"], len(spec["names"]), score_threshold=0.1)
            predicted = decode(output, geometry, spec["names"], 0.25, spec["nms_iou"], 30)
            counts = metrics(predicted, expected, spec["names"])
            totals[view].update(counts)
            observations.append(dict(image=row["image"], view=view, truth=expected,
                                     predictions=predicted, counts=counts,
                                     processing_ms=(time.monotonic()-started)*1000))
        if index % 20 == 0:
            print("EVALUATION", task, label, index+1, len(rows), flush=True)
    report = dict(config=fingerprint, totals={k: summarize(v) for k, v in totals.items()},
                  observations=observations, independent_test=False,
                  scope="Previously observed temporal/source VAL. Stress views are correlated synthetic quality transforms, not new scenes or closed-loop hardware results.")
    save(destination, report)
    del model
    gc.collect()
    return report


def train_task(task, args):
    spec, folder = SPECS[task], OUT / task
    dataset = prepare(task)
    initial = ROOT / spec["initial"]
    plan = dict(task=task, epochs=args.epochs, batch=4, nbs=4, freeze=10,
                freeze_backbone_batchnorm=True, workers=0, cache=False, threads=args.threads,
                shape=spec["shape"], lr0=spec["lr0"], seed=20261029,
                initial=str(initial), initial_sha256=digest(initial),
                dataset_sha256=digest(folder / "dataset_manifest.json"),
                code_sha256=digest(__file__), checkpoint_selection="Fixed final epoch; validation does not select the candidate",
                confidence=0.25, min_iou=0.5, board_access=False, automatic_deployment=False,
                adoption_gate="Clean P/R cannot regress, stress F1 cannot regress; signal also requires no increased red-as-green and improved green recall; navigation requires strictly improved stress F1. Hardware closed-loop must be tested separately.",
                training_reference="https://docs.ultralytics.com/modes/train/")
    plan_path = folder / "plan.json"
    if plan_path.exists():
        assert load(plan_path) == plan, "Frozen training plan changed"
    else:
        save(plan_path, plan)
    status_path = folder / "status.json"
    status = load(status_path) if status_path.exists() else {}
    if status.get("state") == "completed":
        assert digest(status["weights"]) == status["weights_sha256"]
        print("ALREADY_COMPLETED", task, status["weights"], flush=True)
        return status
    status.update(state="baseline_evaluation", task=task, pid=os.getpid(),
                  started=datetime.now().astimezone().isoformat(), board_access=False, adopted=False)
    save(status_path, status)
    monitor_stop, monitor = start_monitor(folder / "memory")
    began = time.monotonic()
    try:
        baseline = evaluate(task, initial, "baseline", dataset)
        run = folder / "training"
        last = run / "weights/last.pt"
        checkpoint = torch.load(last, map_location="cpu", weights_only=False) if last.exists() else None
        already_trained = checkpoint is not None and (checkpoint.get("epoch", -1) + 1 >= args.epochs
                           or (checkpoint.get("epoch", -1) == -1 and status.get("completed_epochs", 0) == args.epochs))
        if not already_trained:
            model = YOLO(str(last if last.exists() else initial))
            FixedShapeTrainer.shape = spec["shape"]
            status.update(state="training", updates=status.get("updates", 0), completed_epochs=status.get("completed_epochs", 0))
            save(status_path, status)

            def on_start(trainer):
                status["batches_per_epoch"] = len(trainer.train_loader)
                status["trainable_parameters"] = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
                assert trainer.accumulate == 1 and trainer.model.model[-1].nc == len(spec["names"])
                def step(*_):
                    status["updates"] += 1
                trainer.optimizer.register_step_post_hook(step)

            def on_epoch_start(trainer):
                for block in trainer.model.model[:10]:
                    block.eval()
                snapshot = memory_snapshot()
                if snapshot["commit_free_GB"] < 2:
                    raise MemoryError("Windows commit headroom below 2 GiB; checkpoint from previous epoch retained")

            def on_epoch_saved(trainer):
                status.update(completed_epochs=trainer.epoch+1, elapsed_seconds=time.monotonic()-began,
                              metrics={k: float(v) for k, v in trainer.metrics.items()}, memory=memory_snapshot())
                save(status_path, status)
                print("TASK_EPOCH", task, status["completed_epochs"], "/", args.epochs,
                      "UPDATES", status["updates"], "SECONDS", round(status["elapsed_seconds"], 1), flush=True)

            model.add_callback("on_train_start", on_start)
            model.add_callback("on_train_epoch_start", on_epoch_start)
            model.add_callback("on_model_save", on_epoch_saved)
            options = dict(trainer=FixedShapeTrainer, data=str(folder / "dataset/data.yaml"),
                           epochs=args.epochs, batch=4, nbs=4, imgsz=max(spec["shape"]), rect=True,
                           device="cpu", workers=0, cache=False, freeze=10, optimizer="AdamW",
                           lr0=spec["lr0"], lrf=0.2, warmup_epochs=0.0, warmup_bias_lr=0.0,
                           mosaic=0.0, mixup=0.0, close_mosaic=0, degrees=0.0,
                           translate=0.01, scale=0.03, fliplr=0.0, flipud=0.0,
                           hsv_h=0.0, hsv_s=0.08, hsv_v=0.10, bgr=0.0,
                           amp=False, plots=False, val=False, patience=args.epochs+1,
                           save=True, save_period=-1, seed=20261029, deterministic=True,
                           project=str(folder), name="training", exist_ok=True, verbose=False)
            if last.exists():
                model.train(trainer=FixedShapeTrainer, resume=True, device="cpu", workers=0)
            else:
                model.train(**options)
            assert model.trainer.save_dir.resolve() == run.resolve()
            del model, checkpoint
            gc.collect()
        assert status["completed_epochs"] == args.epochs
        status.update(state="candidate_evaluation", weights=str(last), weights_sha256=digest(last))
        save(status_path, status)
        candidate = evaluate(task, last, "candidate", dataset)
        b, c = baseline["totals"], candidate["totals"]
        clean_ok = all(c["clean"][k] >= b["clean"][k] for k in ("precision", "recall"))
        stress_ok = c["capture_hard"]["f1"] >= b["capture_hard"]["f1"]
        if task == "signal":
            color_ok = all(c[v].get("red_as_green_overlaps", 0) <= b[v].get("red_as_green_overlaps", 0) for v in b)
            benefit = any(c[v].get("green_tp", 0) > b[v].get("green_tp", 0) for v in b)
        else:
            color_ok = True
            benefit = c["capture_hard"]["f1"] > b["capture_hard"]["f1"]
        result = dict(task=task, baseline=b, candidate=c,
                      clean_precision_recall_not_regressed=clean_ok, stress_f1_not_regressed=stress_ok,
                      red_as_green_not_increased=color_ok, targeted_benefit=benefit,
                      offline_gate_passed=clean_ok and stress_ok and color_ok and benefit,
                      hardware_acceptance=False, deployed=False,
                      action="Candidate only; board closed-loop verification required" if clean_ok and stress_ok and color_ok and benefit else "Keep stable model; candidate did not clear all offline gates")
        save(folder / "comparison.json", result)
        assert digest(initial) == plan["initial_sha256"]
        status.update(state="completed", elapsed_seconds=time.monotonic()-began,
                      comparison=result, adopted=False, weights_sha256=digest(last))
        save(status_path, status)
        print("TASK_COMPLETED", task, json.dumps(result, ensure_ascii=False), flush=True)
        return status
    except BaseException:
        status.update(state="interrupted" if sys.exc_info()[0] is KeyboardInterrupt else "failed", error=traceback.format_exc())
        save(status_path, status)
        raise
    finally:
        monitor_stop.set()
        monitor.join(timeout=2)


@contextmanager
def exclusive_run():
    # OS-owned byte lock is released even when the process exits unexpectedly.
    import msvcrt
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "runner.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["all", *SPECS], default="all")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.epochs <= 30 or not 1 <= args.threads <= 8:
        parser.error("epochs must be 1..30; threads must be 1..8")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    if memory_snapshot()["commit_free_GB"] < 4:
        raise MemoryError("Need at least 4 GiB free Windows commit")
    if shutil.disk_usage(ROOT).free < 3 * 2**30:
        raise OSError("Need at least 3 GiB free disk space")
    tasks = list(SPECS) if args.task == "all" else [args.task]
    with exclusive_run():
        for task in tasks:
            data = prepare(task)
            print("DATA_READY", task, data["counts"], data["config"]["source_groups"], flush=True)
        if args.prepare_only:
            return
        pipeline = dict(state="running", pid=os.getpid(), tasks=tasks, started=datetime.now().astimezone().isoformat(),
                        automatic_deployment=False, board_access=False)
        save(OUT / "pipeline_status.json", pipeline)
        try:
            for task in tasks:
                pipeline.update(current_task=task)
                save(OUT / "pipeline_status.json", pipeline)
                train_task(task, args)
                gc.collect()
            pipeline.update(state="completed", results={t: load(OUT / t / "comparison.json") for t in tasks})
        except BaseException:
            pipeline.update(state="failed", error=traceback.format_exc())
            raise
        finally:
            save(OUT / "pipeline_status.json", pipeline)


if __name__ == "__main__":
    main()
