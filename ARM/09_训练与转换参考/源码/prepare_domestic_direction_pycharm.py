"""Audit the new domestic dataset and prepare recording-separated direction crops.

The two label versions describe the same images. Only label1 is used. Unmapped
66-class data and author off/U-turn labels never become round-lamp targets.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import random
import re

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/domestic_direction_20261022"
SOURCE = ROOT / "素材/label1-20250121-trafficlight-zhuyaodiyilukou-20295"
NAMES = ["round_red", "round_yellow", "round_green", "left_red", "left_yellow", "left_green",
         "straight_red", "straight_yellow", "straight_green", "right_red", "right_yellow", "right_green",
         "uturn_red", "uturn_yellow", "uturn_green", "off"]
DIRECTIONS = ["round", "left", "straight", "right"]


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf8")
    temporary.replace(path)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def recording_id(stem):
    match = re.match(r"^(\d{8})_(\d{6})", stem)
    if match: return "".join(match.groups())
    match = re.match(r"^(\d{14})", stem)
    if match: return match[1]
    raise ValueError("Unrecognized recording prefix: " + stem)


def parse_label(item):
    split, path = item
    image = path.parent.parent / "images" / (path.stem + ".jpg")
    if not image.is_file(): raise FileNotFoundError(image)
    text = path.read_text(encoding="utf8")
    boxes = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip(): continue
        fields = line.split()
        if len(fields) != 5: raise ValueError(f"Malformed label {path}:{number}")
        class_id = int(fields[0]); xywh = list(map(float, fields[1:]))
        if not 0 <= class_id < 16 or not all(math.isfinite(v) for v in xywh):
            raise ValueError(f"Invalid class or nonfinite box {path}:{number}")
        cx, cy, w, h = xywh
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"Invalid normalized box {path}:{number}")
        boxes.append({"line": number, "class_id": class_id, "xywh": xywh})
    return {"image": str(image), "label": str(path), "author_split": split,
            "group": recording_id(path.stem), "label_sha256": hashlib.sha256(text.encode()).hexdigest(), "boxes": boxes}


def inventory():
    path = OUT / "source_inventory.json"
    if path.exists(): return json.loads(path.read_text(encoding="utf8"))
    OUT.mkdir(parents=True, exist_ok=True)
    items = [(s, p) for s in ("train", "val", "test") for p in sorted((SOURCE / s / "labels").glob("*.txt"))]
    frames = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, row in enumerate(pool.map(parse_label, items), 1):
            frames.append(row)
            if i % 2000 == 0: print("DOMESTIC_LABEL_AUDIT", i, "/", len(items), flush=True)
    counts = Counter(box["class_id"] for f in frames for box in f["boxes"])
    groups = defaultdict(Counter)
    for f in frames:
        for b in f["boxes"]: groups[f["group"]][str(b["class_id"])] += 1
    result = {"source": str(SOURCE), "frames": frames, "class_counts": dict(counts),
              "recording_class_counts": {g: dict(c) for g, c in sorted(groups.items())},
              "author_split_frames": dict(Counter(f["author_split"] for f in frames)),
              "schema_status": "candidate numeric mapping requires visual verification",
              "other_materials": "label2 is alternative annotations, not new images; root 66-class data excluded pending class map"}
    save(path, result)
    print("DOMESTIC_INVENTORY", json.dumps({k: v for k, v in result.items() if k != "frames"}, ensure_ascii=False), flush=True)
    return result


def group_split(data):
    """Deterministic recording grouping, with rare-direction coverage when possible.

    Original author's per-image random partitions are not used as independent
    partitions. No model output or visual score participates in this assignment.
    Recording IDs are not verified physical intersection IDs.
    """
    groups = sorted(data["recording_class_counts"])
    counts = {}
    for g in groups:
        values = data["recording_class_counts"][g]
        counts[g] = [sum(int(values.get(str(i), 0)) for i in range(d * 3, d * 3 + 3)) for d in range(4)]
    if len(groups) < 4: raise ValueError("Need at least four recording groups for a bounded train/calibration/test trial")
    rng = random.Random(20261022)
    best = None
    # Metadata-only allocation: favor all four directions in each partition.
    for _ in range(6000):
        shuffled = groups.copy(); rng.shuffle(shuffled)
        n = max(1, round(len(groups) * .15))
        partitions = {"test": shuffled[:n], "calibration": shuffled[n:2*n], "train": shuffled[2*n:]}
        totals = {s: [sum(counts[g][d] for g in gg) for d in range(4)] for s, gg in partitions.items()}
        if any(v == 0 for v in totals["train"]):
            continue
        priority = {"train": 2., "calibration": 1.2, "test": 1.}
        score = sum(priority[s] * min(v, 12 if s != "train" else 45) / (12 if s != "train" else 45)
                    for s, vv in totals.items() for v in vv)
        score += sum(min(v, 80) / 8000 for vv in totals.values() for v in vv)
        if best is None or score > best[0]: best = (score, partitions, totals)
    if best is None: raise RuntimeError("No split with all four directions available for training")
    _, partitions, totals = best
    mapping = {g: s for s, gg in partitions.items() for g in gg}
    save(OUT / "recording_split.json", {"recordings": partitions, "raw_direction_counts": totals,
         "method": "seeded metadata-only recording grouping; author per-image splits merged before regrouping",
         "physical_sites_independent_verified": False, "source_frames": len(data["frames"])})
    return mapping


def crop_box(image, box):
    cx, cy, w, h = box; iw, ih = image.size
    x1, y1, x2, y2 = (cx-w/2)*iw, (cy-h/2)*ih, (cx+w/2)*iw, (cy+h/2)*ih
    pad = max(2, round(max(w*iw, h*ih) * .08))
    region = (max(0, math.floor(x1)-pad), max(0, math.floor(y1)-pad), min(iw, math.ceil(x2)+pad), min(ih, math.ceil(y2)+pad))
    if region[2] <= region[0] or region[3] <= region[1]: raise ValueError("Empty crop")
    return image.crop(region), [x1, y1, x2, y2], [w*iw, h*ih]


def contact(rows, destination, cols=6, max_rows=8):
    rows = rows[:cols*max_rows]
    if not rows: return
    cw, ch = 170, 150
    canvas = Image.new("RGB", (cols*cw, math.ceil(len(rows)/cols)*ch), "#20242b")
    draw = ImageDraw.Draw(canvas)
    for i, row in enumerate(rows):
        x, y = i%cols*cw, i//cols*ch
        with Image.open(row["crop"]) as im:
            thumb = ImageOps.contain(im.convert("RGB"), (160, 112), Image.Resampling.NEAREST)
        canvas.paste(thumb, (x+(cw-thumb.width)//2, y+2))
        draw.text((x+3, y+117), f'{row["id"]} c{row["class_id"]} {NAMES[row["class_id"]]}', fill="white")
        draw.text((x+3, y+133), f'{row["native_wh"][0]:.0f}x{row["native_wh"][1]:.0f}px {row["split"]}', fill="#dddddd")
    canvas.save(destination, quality=95)


def schema_review(data, assignments):
    by_class = defaultdict(list)
    for frame in data["frames"]:
        if assignments[frame["group"]] != "train": continue
        for box in frame["boxes"]:
            by_class[box["class_id"]].append((frame, box))
    rows = []
    folder = OUT / "schema_review"; folder.mkdir(exist_ok=True)
    for class_id in range(16):
        candidates = sorted(by_class[class_id], key=lambda pair: -min(pair[1]["xywh"][2]*1920, pair[1]["xywh"][3]*1080))
        used = Counter(); selected = []
        for frame, box in candidates:
            if used[frame["group"]] >= 2: continue
            selected.append((frame, box)); used[frame["group"]] += 1
            if len(selected) >= 4: break
        for frame, box in candidates:
            if len(selected) >= 6: break
            if (frame, box) not in selected: selected.append((frame, box))
        current = []
        for frame, box in selected:
            with Image.open(frame["image"]) as image:
                crop, xyxy, wh = crop_box(image, box["xywh"])
            path = folder / f'c{class_id:02d}_{len(current):02d}.png'; crop.save(path)
            row = {"id": f"c{class_id:02d}_{len(current):02d}", "crop": str(path), "image": frame["image"],
                   "box": xyxy, "native_wh": wh, "class_id": class_id, "candidate_name": NAMES[class_id], "split": "train"}
            current.append(row); rows.append(row)
        contact(current, folder / f"class_{class_id:02d}.jpg", cols=6, max_rows=1)
    contact(rows[:48], folder / "schema_first8.jpg")
    contact(rows[48:], folder / "schema_last8.jpg")
    save(folder / "review_rows.json", rows)
    print("DOMESTIC_SCHEMA_CONTACTS_READY", folder, flush=True)


def prepare(data, assignments):
    review = OUT / "schema_review/verified_mapping.json"
    if not review.is_file(): raise RuntimeError("Inspect schema review contacts and save verified_mapping.json first")
    mapping_review = json.loads(review.read_text(encoding="utf8"))
    if mapping_review.get("direction_mapping_confirmed") is not True: raise ValueError("Direction mapping not confirmed")
    # Native-size filtering is quality selection, not proof that every frame is
    # visually clear. Only author active round/left/straight/right labels used.
    candidates = defaultdict(list)
    review_exclusion = OUT / "review_excluded_author_boxes.json"
    excluded_boxes = set()
    if review_exclusion.exists():
        excluded_boxes = {(r["source_image"], r["author_line"]) for r in json.loads(review_exclusion.read_text(encoding="utf8"))["source_boxes"]}
    for frame in data["frames"]:
        split = assignments[frame["group"]]
        for box in frame["boxes"]:
            if box["class_id"] >= 12: continue
            w, h = box["xywh"][2]*1920, box["xywh"][3]*1080
            if min(w, h) < (12 if box["class_id"] < 3 else 20) or w*h < 240: continue
            if (frame["image"], box["line"]) in excluded_boxes: continue
            candidates[split, box["class_id"]//3].append((frame, box))
    caps = {"train": [400, 400, 240, 220], "calibration": [120, 100, 100, 100], "test": [120, 100, 100, 100]}
    rows = []; rejected = Counter(); source_hash_splits = defaultdict(set)
    for (split, direction), pairs in sorted(candidates.items()):
        # Round-robin recordings and colors instead of taking an initial video.
        buckets = defaultdict(list)
        for frame, box in pairs: buckets[frame["group"], box["class_id"]].append((frame, box))
        rng = random.Random(20261022 + direction)
        for bucket in buckets.values(): rng.shuffle(bucket)
        keys = sorted(buckets); rng.shuffle(keys)
        pool = []
        while any(buckets.values()):
            for key in keys:
                if buckets[key]: pool.append(buckets[key].pop())
        used_frames = Counter(); per_group = Counter(); accepted = 0
        for frame, box in pool:
            if accepted >= caps[split][direction]: break
            # One target of a direction per frame; cap correlated recording load.
            if used_frames[frame["image"]] or per_group[frame["group"]] >= (100 if split == "train" else 45): continue
            with Image.open(frame["image"]) as source:
                im = source.convert("RGB"); crop, xyxy, wh = crop_box(im, box["xywh"])
                if min(wh) < (12 if direction == 0 else 20) or wh[0]*wh[1] < 240: rejected["native_size"] += 1; continue
                source_hash = hashlib.sha256(im.tobytes()).hexdigest()
            if source_hash_splits[source_hash] and split not in source_hash_splits[source_hash]:
                rejected["exact_source_duplicate_across_split"] += 1; continue
            source_hash_splits[source_hash].add(split)
            crop_path = OUT / "crops" / split / f"d{direction}_{accepted:04d}.png"
            crop_path.parent.mkdir(parents=True, exist_ok=True); crop.save(crop_path)
            rows.append({"id": f"{split}_d{direction}_{accepted:04d}", "crop": str(crop_path), "source_image": frame["image"],
                         "source_pixel_sha256": source_hash, "source_label": frame["label"], "source_label_sha256": frame["label_sha256"],
                         "author_split": frame["author_split"], "recording_group": frame["group"], "split": split,
                         "box": xyxy, "native_wh": wh, "class_id": box["class_id"], "author_name": NAMES[box["class_id"]],
                         "direction_id": direction, "direction": DIRECTIONS[direction], "label_origin": "author numeric annotation + visually checked schema",
                         "mirrored": False, "physical_lane_head_relation": "unknown"})
            used_frames[frame["image"]] += 1; per_group[frame["group"]] += 1; accepted += 1
            if accepted % 100 == 0: print("DOMESTIC_CROPS", split, DIRECTIONS[direction], accepted, flush=True)
    groups_by_split = {s: sorted({r["recording_group"] for r in rows if r["split"] == s}) for s in caps}
    assert not set(groups_by_split["train"]) & (set(groups_by_split["calibration"]) | set(groups_by_split["test"]))
    assert not set(groups_by_split["calibration"]) & set(groups_by_split["test"])
    manifest = {"source": str(SOURCE), "source_inventory_sha256": digest(OUT/"source_inventory.json"), "direction_names": DIRECTIONS,
                "records": rows, "groups_by_split": groups_by_split, "split_counts": {s: dict(Counter(r["direction"] for r in rows if r["split"] == s)) for s in caps},
                "rejected": dict(rejected), "schema_review_sha256": digest(review), "author_test_repartitioned": True,
                "scope": "Recording-separated crop classification pilot; physical sites not verified independent; no physical lane-head labels"}
    manifest["fingerprint"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    save(OUT / "manifest.json", manifest)
    train_rows = [r for r in rows if r["split"] == "train"]
    for d in range(4): contact([r for r in train_rows if r["direction_id"] == d], OUT/f"train_{DIRECTIONS[d]}.jpg")
    print("DOMESTIC_DATASET_PREPARED", manifest["split_counts"], flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--prepare", action="store_true"); args = parser.parse_args()
    data = inventory(); assignments = group_split(data)
    if args.prepare: prepare(data, assignments)
    else: schema_review(data, assignments)


if __name__ == "__main__": main()
