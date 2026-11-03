"""BDD100K 四类检测训练入口。直接运行开始训练；--check-only 仅检查。"""

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

from prepare_bdd100k import CLASS_NAMES, DEFAULT_OUTPUT, prepare_dataset

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "yolov8n.pt"


def training_options(epochs=3, imgsz=416, batch=2):
    return dict(
        data=str(DEFAULT_OUTPUT / "data.yaml"),
        epochs=epochs, imgsz=imgsz, batch=batch,
        device="cpu", workers=0, cache=False, amp=False,
        seed=42, deterministic=True,
        optimizer="AdamW", lr0=0.001, warmup_epochs=1.0,
        mosaic=0.0, close_mosaic=0, plots=False,
        project=str(ROOT / "runs" / "adas"),
        name="bdd100k_4class_smoke", exist_ok=False,
        save=True, val=True,
    )


def check_training_input(options, weights=WEIGHTS):
    """检查完整数据加载，并执行一次前向计算；不更新模型权重。"""
    import torch
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.nn.tasks import DetectionModel

    torch.set_num_threads(8)
    data = check_det_dataset(options["data"], autodownload=False)
    if list(data["names"].values()) != CLASS_NAMES:
        raise ValueError("训练配置的类别顺序与转换程序不一致。")
    cfg = get_cfg(overrides={**options, "task": "detect", "mode": "train"})
    train_dataset = None
    for split, expected in [("train", 1000), ("val", 200)]:
        dataset = build_yolo_dataset(
            cfg, data[split], options["batch"], data,
            mode=split, rect=split == "val", stride=32,
        )
        if len(dataset) != expected:
            raise ValueError(f"{split} 读取到 {len(dataset)} 张图片，预期 {expected} 张。")
        if sum(len(label["cls"]) for label in dataset.labels) == 0:
            raise ValueError(f"{split} 没有有效标注。")
        print(f"{split}：{len(dataset)} 张图片，标注加载通过。", flush=True)
        if split == "train":
            train_dataset = dataset
    batch = next(iter(build_dataloader(train_dataset, options["batch"], 0, shuffle=False)))
    pretrained = YOLO(str(weights)).model
    detector = DetectionModel(pretrained.yaml, nc=len(CLASS_NAMES), verbose=False)
    detector.load(pretrained)
    detector.eval()
    with torch.inference_mode():
        prediction = detector(batch["img"].float() / 255.0)[0]
    if prediction.shape[1] != 4 + len(CLASS_NAMES) or not torch.isfinite(prediction).all():
        raise ValueError("四类检测模型前向检查失败。")
    print(f"模型前向计算通过，输出形状：{tuple(prediction.shape)}。未启动训练。", flush=True)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="只检查数据和模型，不训练")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--weights", type=Path, default=WEIGHTS, help="预训练或上一阶段的最佳权重")
    parser.add_argument("--name", default="bdd100k_4class_smoke", help="本阶段结果目录名")
    parser.add_argument("--lr0", type=float, default=0.001, help="本阶段初始学习率")
    parser.add_argument("--report", action="store_true", help="训练后自动生成逐类验证报告与检测对照图")
    args = parser.parse_args()
    if min(args.epochs, args.imgsz, args.batch) < 1:
        parser.error("epochs、imgsz、batch 必须为正整数。")
    if not args.weights.is_file():
        raise FileNotFoundError(f"缺少本地训练权重：{args.weights}")
    if not 0 < args.lr0 < 1:
        parser.error("lr0 必须介于 0 和 1 之间。")
    prepare_dataset()
    options = training_options(args.epochs, args.imgsz, args.batch)
    options.update(name=args.name, lr0=args.lr0)
    print("训练类别：", ", ".join(CLASS_NAMES), flush=True)
    if args.check_only:
        check_training_input(options, args.weights)
        return
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(8)
    print(f"开始 CPU 训练：1000 张训练 / 200 张验证，本阶段 {args.epochs} 轮，尺寸 {args.imgsz}。", flush=True)
    print("这是小规模流程验证，正式模型需要后续扩大数据和训练轮数。", flush=True)
    model = YOLO(str(args.weights.resolve()))
    model.train(**options)
    print(f"训练完成。结果目录：{model.trainer.save_dir}", flush=True)
    print(f"最佳权重：{model.trainer.save_dir / 'weights' / 'best.pt'}", flush=True)
    if args.report:
        from report_adas import create_report
        create_report(model.trainer.save_dir)


if __name__ == "__main__":
    main()

