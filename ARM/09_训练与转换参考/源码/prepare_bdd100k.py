"""把本机 BDD100K 原始 JSON 转换为独立的 YOLO 四类小规模数据集。"""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "datasets" / "adas" / "raw" / "BDD100k"
IMAGE_ROOT = RAW / "bdd100k_images" / "bdd100k" / "images" / "100k"
LABEL_ROOT = RAW / "bdd100k_labels" / "bdd100k" / "labels" / "100k"
DEFAULT_OUTPUT = ROOT / "datasets" / "adas" / "prepared" / "bdd100k_4class_smoke"
CLASS_NAMES = ["traffic_light", "vehicle", "two_wheeler", "person"]
CATEGORY_MAP = {
    "traffic light": 0,
    "car": 1, "bus": 1, "truck": 1,
    "bike": 2, "motor": 2,
    "person": 3, "rider": 3,
}
VERSION = 1


def convert_annotations(record, width, height):
    """只取目标检测框，裁剪越界坐标，排除车道线和可行驶区域。"""
    frames = record.get("frames", [])
    if len(frames) != 1 or not isinstance(frames[0].get("objects"), list):
        raise ValueError("预期 BDD100K 单帧 frames/objects 标注格式。")
    lines, seen, stats = [], set(), Counter()
    for obj in frames[0]["objects"]:
        category = obj.get("category")
        if category not in CATEGORY_MAP:
            stats["ignored_other_categories"] += 1
            continue
        box = obj.get("box2d")
        if not box:
            raise ValueError(f"检测类别 {category} 缺少 box2d。")
        coords = [float(box[key]) for key in ("x1", "y1", "x2", "y2")]
        if not all(math.isfinite(value) for value in coords):
            stats["invalid_boxes"] += 1
            continue
        x1, y1, x2, y2 = coords
        clipped = [max(0.0, min(width, x1)), max(0.0, min(height, y1)),
                   max(0.0, min(width, x2)), max(0.0, min(height, y2))]
        if clipped != coords:
            stats["clipped_boxes"] += 1
        x1, y1, x2, y2 = clipped
        if x2 <= x1 or y2 <= y1:
            stats["invalid_boxes"] += 1
            continue
        cls = CATEGORY_MAP[category]
        values = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                  (x2 - x1) / width, (y2 - y1) / height)
        line = f"{cls} " + " ".join(f"{value:.8f}" for value in values)
        if line in seen:
            stats["duplicate_boxes"] += 1
            continue
        seen.add(line)
        lines.append(line)
        stats[CLASS_NAMES[cls]] += 1
    return lines, stats


def prepare_dataset(train_count=1000, val_count=200, seed=42, output=DEFAULT_OUTPUT):
    output = Path(output).resolve()
    if train_count < 1 or val_count < 1:
        raise ValueError("训练和验证图片数量必须为正整数。")
    if output == RAW.resolve() or RAW.resolve() in output.parents:
        raise ValueError("输出目录必须独立于原始数据目录。")
    config = {"version": VERSION, "train_count": train_count, "val_count": val_count,
              "seed": seed, "source": str(RAW.resolve()), "names": CLASS_NAMES}
    report_path = output / "preparation_report.json"
    yaml_path = output / "data.yaml"
    if output.exists():
        if not report_path.is_file() or not yaml_path.is_file():
            raise RuntimeError(f"输出目录已有文件但未完成准备，请使用新的输出目录：{output}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("config") != config:
            raise RuntimeError("已有数据集的准备参数不同，请使用新的输出目录。")
        for split, count in [("train", train_count), ("val", val_count)]:
            expected = set(report["splits"][split]["image_stems"])
            images = {p.stem for p in (output / "images" / split).glob("*.jpg")}
            labels = {p.stem for p in (output / "labels" / split).glob("*.txt")}
            if len(expected) != count or images != expected or labels != expected:
                raise RuntimeError(f"已准备的 {split} 图片或标签不完整，请检查：{output}")
        print(f"复用已准备的数据集：{output}", flush=True)
        return yaml_path

    selections = {}
    for split, count in [("train", train_count), ("val", val_count)]:
        directory = IMAGE_ROOT / split
        if not directory.is_dir() or not (LABEL_ROOT / split).is_dir():
            raise FileNotFoundError(f"缺少原始 {split} 图片或标注目录：{RAW}")
        images = sorted(directory.glob("*.jpg"))
        if len(images) < count:
            raise ValueError(f"{split} 只有 {len(images)} 张图片，无法选取 {count} 张。")
        selections[split] = random.Random(seed if split == "train" else seed + 1).sample(images, count)
    if {p.stem for p in selections["train"]} & {p.stem for p in selections["val"]}:
        raise ValueError("训练与验证图片重叠。")

    report = {"config": config, "splits": {}}
    for split, images in selections.items():
        image_out = output / "images" / split
        label_out = output / "labels" / split
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)
        stats = Counter()
        for index, source in enumerate(images, 1):
            annotation = LABEL_ROOT / split / f"{source.stem}.json"
            record = json.loads(annotation.read_text(encoding="utf-8"))
            if Path(record.get("name", "")).stem != source.stem:
                raise ValueError(f"图片与标注名称不一致：{annotation}")
            with Image.open(source) as image:
                width, height = image.size
                image.verify()
            # Ultralytics 会尝试修复缺少结束标记的 JPEG；硬链接前检查以保护原图。
            with source.open("rb") as image_file:
                image_file.seek(-2, os.SEEK_END)
                if image_file.read() != b"\xff\xd9":
                    raise ValueError(f"JPEG 缺少完整结束标记：{source}")
            lines, counts = convert_annotations(record, width, height)
            stats.update(counts)
            if not lines:
                stats["background_images"] += 1
            destination = image_out / source.name
            try:
                os.link(source, destination)
                stats["hardlinked_images"] += 1
            except OSError:
                shutil.copy2(source, destination)
                stats["copied_images"] += 1
            (label_out / f"{source.stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            if index % 200 == 0 or index == len(images):
                print(f"{split}：已准备 {index}/{len(images)} 张。", flush=True)
        if any(stats[name] == 0 for name in CLASS_NAMES):
            raise ValueError(f"{split} 未覆盖全部四类，请换随机种子或扩大图片数量。")
        report["splits"][split] = {"images": len(images), "stats": dict(stats),
                                   "image_stems": [p.stem for p in images]}
    yaml_text = f"path: {json.dumps(output.as_posix(), ensure_ascii=False)}\ntrain: images/train\nval: images/val\nnames:\n"
    yaml_text += "".join(f"  {index}: {name}\n" for index, name in enumerate(CLASS_NAMES))
    yaml_path.write_text(yaml_text, encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"准备完成：{yaml_path}", flush=True)
    return yaml_path


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-count", type=int, default=1000)
    parser.add_argument("--val-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    prepare_dataset(args.train_count, args.val_count, args.seed, args.output)
