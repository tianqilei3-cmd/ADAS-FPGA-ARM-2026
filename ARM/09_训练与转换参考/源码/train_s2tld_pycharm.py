"""PyCharm 右键运行：训练红、黄、绿、熄灭交通灯框检测模型。"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runs/adas_config"))
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

WEIGHTS = ROOT / "yolov8n.pt"  # COCO 预训练权重，不使用 BDD 四类权重
EPOCHS = 20
IMAGE_SIZE = 640
BATCH_SIZE = 2
DEVICE = "cpu"
LEARNING_RATE = 0.001
CHECK_ONLY = False
RESUME_CHECKPOINT = ROOT / "runs/adas/s2tld_light_states/weights/last.pt"

def main():
    import json
    import sys
    import torch
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset, build_dataloader
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.nn.tasks import DetectionModel
    from prepare_s2tld import prepare_dataset, NAMES
    from mytrain import training_options

    for output in (sys.stdout, sys.stderr):
        if hasattr(output, "reconfigure"):
            output.reconfigure(encoding="utf-8")
    if not WEIGHTS.is_file():
        raise FileNotFoundError(WEIGHTS)
    if min(EPOCHS, IMAGE_SIZE, BATCH_SIZE) < 1 or not 0 < LEARNING_RATE < 1:
        raise ValueError("训练参数不合法")
    torch.set_num_threads(8)
    data_path = prepare_dataset()
    options = training_options(EPOCHS, IMAGE_SIZE, BATCH_SIZE)
    options.update(data=str(data_path), device=DEVICE, name="s2tld_light_states", lr0=LEARNING_RATE)
    data = check_det_dataset(str(data_path), autodownload=False)
    if list(data["names"].values()) != NAMES:
        raise ValueError("类别顺序错误")
    if CHECK_ONLY:
        cfg = get_cfg(overrides={**options, "task": "detect", "mode": "train"})
        train_dataset = None
        for split in ("train", "val", "test"):
            dataset = build_yolo_dataset(cfg, data[split], BATCH_SIZE, data, mode="train" if split=="train" else "val", rect=split!="train", stride=32)
            print(f"{split}：{len(dataset)} 张，标注加载通过", flush=True)
            if split == "train":
                train_dataset = dataset
        batch = next(iter(build_dataloader(train_dataset, BATCH_SIZE, 0, shuffle=False)))
        pretrained = YOLO(str(WEIGHTS)).model
        detector = DetectionModel(pretrained.yaml, nc=4, verbose=False)
        detector.load(pretrained)
        detector.eval()
        with torch.inference_mode():
            prediction = detector(batch["img"].float()/255)[0]
        if prediction.shape[1] != 8 or not torch.isfinite(prediction).all():
            raise ValueError("四类模型前向检查失败")
        print("检查通过，未启动训练。", flush=True)
        return
    print(f"开始灯状态训练：{EPOCHS} 轮，{IMAGE_SIZE} 尺寸，{DEVICE}；0红/1黄/2绿/3熄灭", flush=True)
    from training_memory_monitor import memory_snapshot, start_monitor
    snapshot = memory_snapshot()
    print(f"训练前内存：{snapshot}", flush=True)
    if snapshot["commit_free_GB"] < 2:
        raise RuntimeError("Windows 剩余提交额度不足 2 GB。请先开启系统管理的分页文件并关闭不必要的程序，再开始训练。")
    if RESUME_CHECKPOINT is not None:
        checkpoint = Path(RESUME_CHECKPOINT)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state.get("epoch", -1) < 0 or state.get("optimizer") is None:
            raise ValueError("此权重不是可恢复的训练断点")
        print(f"从已完成第 {state['epoch'] + 1} 轮恢复训练：{checkpoint}", flush=True)
        del state
        model = YOLO(str(checkpoint))
    else:
        model = YOLO(str(WEIGHTS))
    stop, monitor_thread = start_monitor(ROOT / "runs/s2tld_memory_logs")
    try:
        if RESUME_CHECKPOINT is not None:
            model.train(resume=True)
        else:
            model.train(**options)
    finally:
        stop.set()
        monitor_thread.join(timeout=2)
    run_dir = model.trainer.save_dir
    best = YOLO(str(run_dir / "weights/best.pt"))
    report_dir = run_dir / "report"
    report_dir.mkdir(exist_ok=True)
    # 只评估验证集，测试集保留到最终模型定稿后使用。
    metrics = best.val(data=str(data_path), split="val", imgsz=IMAGE_SIZE, batch=BATCH_SIZE,
                       device=DEVICE, workers=0, conf=0.001, plots=True,
                       project=str(report_dir), name="validation", exist_ok=True)
    classes = []
    for index, cls in enumerate(metrics.ap_class_index):
        precision, recall, ap50, ap = metrics.class_result(index)
        classes.append(dict(name=metrics.names[int(cls)], instances=int(metrics.nt_per_class[int(cls)]),
                            precision=float(precision), recall=float(recall), AP50=float(ap50), AP50_95=float(ap)))
    report = dict(weights=str(run_dir / "weights/best.pt"), data=str(data_path), imgsz=IMAGE_SIZE,
                  overall={key:float(value) for key,value in metrics.results_dict.items()}, classes=classes)
    (report_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成，最佳权重：{run_dir / 'weights/best.pt'}\n逐类指标：{report_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
