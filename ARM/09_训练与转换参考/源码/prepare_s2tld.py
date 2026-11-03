"""S²TLD XML 转 YOLO：连续编号分组划分，保留原始文件。"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "datasets/adas/raw/S2TLD（720x1280）"
OUTPUT = ROOT / "datasets/adas/prepared/s2tld_light_states"
NAMES = ["red", "yellow", "green", "off"]
SEED = 42
BLOCK_SIZE = 50  # 同一来源中每连续 50 个编号归为一组


def prepare_dataset():
    from PIL import Image
    import yaml
    config = dict(version=1, seed=SEED, block_size=BLOCK_SIZE, names=NAMES, source=str(RAW))
    manifest = OUTPUT / "preparation_report.json"
    if OUTPUT.exists():
        if not manifest.is_file():
            raise RuntimeError(f"准备目录未完成，请检查或选择新 OUTPUT：{OUTPUT}")
        report = json.loads(manifest.read_text(encoding="utf-8"))
        if report["config"] != config:
            raise ValueError("准备参数变化，请使用新的 OUTPUT")
        for split, info in report["splits"].items():
            expected = set(info["image_stems"])
            if {p.stem for p in (OUTPUT / f"images/{split}").glob("*.jpg")} != expected or \
               {p.stem for p in (OUTPUT / f"labels/{split}").glob("*.txt")} != expected:
                raise ValueError(f"{split} 图片/标签不完整")
        if not (OUTPUT / "data.yaml").is_file():
            raise FileNotFoundError("缺少 data.yaml")
        print(f"复用数据：{OUTPUT}", flush=True)
        return OUTPUT / "data.yaml"

    records = []
    for source_group in ("normal_1", "normal_2"):
        image_dir = RAW / source_group / "JPEGImages"
        xml_dir = RAW / source_group / "Annotations"
        images = {p.stem: p for p in image_dir.glob("*.jpg")}
        xmls = {p.stem: p for p in xml_dir.glob("*.xml")}
        if not images or images.keys() != xmls.keys():
            raise ValueError(f"{source_group} 图片/XML 不完整")
        for stem in sorted(images):
            source = images[stem]
            content = source.read_bytes()
            if not content.endswith(b"\xff\xd9"):
                raise ValueError(f"JPEG 不完整：{source}")
            with Image.open(source) as image:
                width, height = image.size
                image.verify()
            xml = ET.parse(xmls[stem]).getroot()
            if Path(xml.findtext("filename", "")).stem != stem:
                raise ValueError(f"XML 图片名不匹配：{xmls[stem]}")
            if int(xml.findtext("size/width")) != width or int(xml.findtext("size/height")) != height:
                raise ValueError(f"XML 尺寸不匹配：{source}")
            lines, counts, seen = [], Counter(), set()
            for obj in xml.findall("object"):
                name = obj.findtext("name", "").strip()
                if name not in NAMES:
                    raise ValueError(f"未知类别 {name}：{xmls[stem]}")
                box = [float(obj.findtext(f"bndbox/{key}")) for key in ("xmin", "ymin", "xmax", "ymax")]
                if not all(math.isfinite(value) for value in box):
                    raise ValueError(f"非有限框：{xmls[stem]}")
                x1, y1, x2, y2 = box
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    raise ValueError(f"非法框 {box}：{xmls[stem]}")
                row = f"{NAMES.index(name)} " + " ".join(f"{value:.8f}" for value in (
                    (x1+x2)/(2*width), (y1+y2)/(2*height), (x2-x1)/width, (y2-y1)/height))
                if row not in seen:
                    seen.add(row)
                    lines.append(row)
                    counts[name] += 1
            records.append(dict(source=source, stem=f"{source_group}_{stem}", lines=lines,
                                counts=counts, digest=hashlib.sha256(content).hexdigest(),
                                group=f"{source_group}_{int(stem)//BLOCK_SIZE}"))
        print(f"已检查 {source_group}，累计 {len(records)} 张", flush=True)

    # 完全重复图片所在的编号组也必须同集合。
    parent = {record["group"]: record["group"] for record in records}
    def find(group):
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group
    hashes = {}
    for record in records:
        if record["digest"] in hashes:
            parent[find(record["group"])] = find(hashes[record["digest"]])
        else:
            hashes[record["digest"]] = record["group"]
    groups = {}
    for record in records:
        groups.setdefault(find(record["group"]), []).append(record)
    keys = sorted(groups)
    totals = Counter()
    for record in records:
        totals.update(record["counts"])
    # 在完整组级别搜索可复现的约 80/10/10 划分，确保每个集合包含四类。
    rng, best = random.Random(SEED), None
    for _ in range(1000):
        order = keys.copy()
        rng.shuffle(order)
        ntrain, nval = int(len(order)*0.8), max(1, int(len(order)*0.1))
        partitions = {"train": order[:ntrain], "val": order[ntrain:ntrain+nval], "test": order[ntrain+nval:]}
        score, selected = 0, {}
        for split, selected_groups in partitions.items():
            selected[split] = [record for group in selected_groups for record in groups[group]]
            counts = Counter()
            for record in selected[split]:
                counts.update(record["counts"])
            ratio = 0.8 if split == "train" else 0.1
            score += abs(len(selected[split])/len(records)-ratio)
            score += sum(abs(counts[name]/totals[name]-ratio) for name in NAMES)
            if any(counts[name] == 0 for name in NAMES):
                score += 100
        if best is None or score < best[0]:
            best = score, selected
    if best[0] >= 100:
        raise ValueError("无法保证每个集合覆盖四类，请调整分组")
    report = dict(config=config, total_images=len(records), total_boxes=dict(totals),
                  unique_image_hashes=len(hashes), splits={},
                  limitations=["连续50编号分组是近邻隔离启发式，非官方场景划分；未保证所有近重复帧隔离。",
                               "相同文件哈希不跨集合；重复图片保留，独立样本数应参考 unique_image_hashes。",
                               "XML 坐标直接按连续 xyxy 转换，未擅自做 VOC 一像素偏移。"])
    for split, items in best[1].items():
        image_out, label_out = OUTPUT / f"images/{split}", OUTPUT / f"labels/{split}"
        image_out.mkdir(parents=True)
        label_out.mkdir(parents=True)
        counts = Counter()
        for record in items:
            shutil.copy2(record["source"], image_out / (record["stem"] + ".jpg"))
            (label_out / (record["stem"] + ".txt")).write_text("\n".join(record["lines"]) + "\n", encoding="utf-8")
            counts.update(record["counts"])
        report["splits"][split] = dict(images=len(items), boxes=dict(counts), image_stems=[r["stem"] for r in items])
        print(f"{split}：{len(items)} 张，目标框：{dict(counts)}", flush=True)
    yaml_path = OUTPUT / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(dict(path=OUTPUT.as_posix(), train="images/train", val="images/val",
                                           test="images/test", names=dict(enumerate(NAMES))), sort_keys=False), encoding="utf-8")
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return yaml_path


if __name__ == "__main__":
    print(prepare_dataset())
