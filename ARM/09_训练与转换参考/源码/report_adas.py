"""验证一阶段的最佳权重，保存逐类指标和固定样例的标注/预测对照。"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

CLASS_ZH = {"traffic_light": "交通灯", "vehicle": "车辆", "two_wheeler": "两轮车", "person": "行人"}


def create_report(run_dir, imgsz=None):
    import cv2
    import torch
    import yaml
    from ultralytics import YOLO

    run_dir = Path(run_dir).resolve()
    weights = run_dir / "weights" / "best.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"训练尚未生成最佳权重：{weights}")
    args = yaml.safe_load((run_dir / "args.yaml").read_text(encoding="utf-8"))
    eval_imgsz = args["imgsz"] if imgsz is None else imgsz
    if eval_imgsz < 1:
        raise ValueError("验证图片尺寸必须为正整数。")
    data_path = Path(args["data"])
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    data_root = Path(data["path"])
    if not data_root.is_absolute():
        data_root = data_path.parent / data_root
    image_dir = data_root / data["val"]
    label_dir = data_root / "labels" / "val"
    report_dir = run_dir / ("report" if imgsz is None else f"report_{imgsz}")
    report_dir.mkdir(exist_ok=True)
    torch.set_num_threads(8)
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_path), imgsz=eval_imgsz, batch=args["batch"],
        device="cpu", workers=0, plots=False, save_json=False,
        conf=0.001, iou=0.7, project=str(report_dir), name="validation",
        exist_ok=True, verbose=True,
    )
    per_class = []
    for index, class_id in enumerate(metrics.ap_class_index):
        class_id = int(class_id)
        precision, recall, ap50, ap = metrics.class_result(index)
        name = metrics.names[class_id]
        per_class.append(dict(
            class_id=class_id, name=name, chinese_name=CLASS_ZH.get(name, name),
            instances=int(metrics.nt_per_class[class_id]),
            precision=float(precision), recall=float(recall),
            AP50=float(ap50), AP50_95=float(ap),
        ))
    report = dict(
        created=datetime.now().astimezone().isoformat(), weights=str(weights),
        training_epochs=args["epochs"], training_imgsz=args["imgsz"], initial_weights=str(args["model"]),
        validation=dict(data=str(data_path), image_count=len(list(image_dir.glob("*.jpg"))),
                        imgsz=eval_imgsz, conf=0.001, iou=0.7),
        overall={key: float(value) for key, value in metrics.results_dict.items()},
        classes=per_class, speed_ms_per_image={key: float(value) for key, value in metrics.speed.items()},
        limitations=["小规模验证集，不能代表比赛或实路部署效果。",
                     "精确率与召回率由验证程序选择置信度工作点，mAP 不是图片识别正确率。",
                     "CPU 验证耗时不是开发板或 FPGA 的实时性能。",
                     "不包含灯色/方向、车道线、斑马线和交通锥。"],
    )
    # 按文件名固定选取，并覆盖各类标注；不根据预测效果挑图。
    images = sorted(image_dir.glob("*.jpg"))
    selected = []
    for class_id in sorted(metrics.names):
        for path in images:
            rows = (label_dir / (path.stem + ".txt")).read_text(encoding="utf-8").splitlines()
            if any(row.split()[0] == str(class_id) for row in rows):
                if path not in selected:
                    selected.append(path)
                break
    for path in images:
        if len(selected) >= 6:
            break
        if path not in selected:
            selected.append(path)
    example_dir = report_dir / "examples"
    example_dir.mkdir(exist_ok=True)
    report["examples"] = []
    for source in selected:
        prediction = model.predict(str(source), imgsz=eval_imgsz, device="cpu",
                                   conf=0.25, iou=0.7, save=False, verbose=False)[0]
        truth = cv2.imread(str(source))
        height, width = truth.shape[:2]
        for row in (label_dir / (source.stem + ".txt")).read_text(encoding="utf-8").splitlines():
            cls, x, y, w, h = map(float, row.split())
            start = (round((x - w / 2) * width), round((y - h / 2) * height))
            end = (round((x + w / 2) * width), round((y + h / 2) * height))
            cv2.rectangle(truth, start, end, (0, 200, 0), 2)
            cv2.putText(truth, metrics.names[int(cls)], (start[0], max(14, start[1] - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
        predicted = prediction.plot(line_width=2, font_size=12)
        for frame, caption in ((truth, "Ground truth"), (predicted, "Prediction (conf >= 0.25)")):
            cv2.rectangle(frame, (0, 0), (width, 35), (20, 20, 20), -1)
            cv2.putText(frame, caption, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        target = example_dir / source.name
        if not cv2.imwrite(str(target), cv2.hconcat([truth, predicted])):
            raise IOError(f"无法保存检测对照图：{target}")
        report["examples"].append(dict(source=str(source), comparison=str(target),
                                      detections=len(prediction.boxes), confidence=0.25))
    (report_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    overall = report["overall"]
    lines = ["# 四类检测训练效果", "", f"最佳权重：`{weights}`", "",
             f"本阶段训练 {args['epochs']} 轮，训练尺寸 {args['imgsz']}；验证 {report['validation']['image_count']} 张图片，验证尺寸 {eval_imgsz}。", "",
             f"mAP50：{overall['metrics/mAP50(B)']:.2%}；mAP50–95：{overall['metrics/mAP50-95(B)']:.2%}。", "",
             "| 类别 | 验证标注数 | 精确率 | 召回率 | AP50 | AP50–95 |",
             "|---|---:|---:|---:|---:|---:|"]
    for item in per_class:
        lines.append(f"| {item['chinese_name']} | {item['instances']} | {item['precision']:.2%} | {item['recall']:.2%} | {item['AP50']:.2%} | {item['AP50_95']:.2%} |")
    lines += ["", "样例位于 examples：左侧原始标注，右侧模型预测（置信度至少 0.25）。", "",
              "限制：", ""] + [f"- {item}" for item in report["limitations"]]
    (report_dir / "训练效果.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--imgsz", type=int, help="可选：单独评估另一输入尺寸，并保存到独立报告目录")
    args = parser.parse_args()
    create_report(args.run_dir, args.imgsz)
