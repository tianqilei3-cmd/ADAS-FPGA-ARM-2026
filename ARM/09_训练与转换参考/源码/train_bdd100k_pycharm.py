"""在 PyCharm 中右键运行本文件，即可开始下一阶段 BDD100K 四类训练。"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runs" / "adas_config"))
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

# ==================== 在这里修改训练参数 ====================
DATA = ROOT / "datasets/adas/prepared/bdd100k_4class_smoke/data.yaml"
WEIGHTS = ROOT / "runs/adas/bdd100k_4class_stage2/weights/best.pt"
# 从头微调 COCO 预训练模型时，改为：WEIGHTS = ROOT / "yolov8n.pt"
EPOCHS = 20                 # 本阶段新增轮数
IMAGE_SIZE = 416            # 可改为 640；CPU 训练会更慢
BATCH_SIZE = 2
DEVICE = "cpu"             # 当前 yolo 环境为 CPU 版 PyTorch
LEARNING_RATE = 0.0003
RUN_NAME = "bdd100k_4class_stage3"
GENERATE_REPORT = True     # 完成后生成逐类指标和检测对照图
CHECK_ONLY = False         # True：只检查数据与前向计算，不训练
# ==========================================================


def main():
    import sys
    import torch
    from ultralytics import YOLO
    from mytrain import training_options, check_training_input

    for output in (sys.stdout, sys.stderr):
        if hasattr(output, "reconfigure"):
            output.reconfigure(encoding="utf-8")
    if not DATA.is_file():
        raise FileNotFoundError(f"找不到已准备的数据配置：{DATA}")
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"找不到模型权重：{WEIGHTS}")
    if min(EPOCHS, IMAGE_SIZE, BATCH_SIZE) < 1 or not 0 < LEARNING_RATE < 1:
        raise ValueError("轮数、图片尺寸、批大小必须为正整数，学习率必须在 (0, 1) 内")
    torch.set_num_threads(8)
    options = training_options(EPOCHS, IMAGE_SIZE, BATCH_SIZE)
    options.update(
        data=str(DATA), device=DEVICE, name=RUN_NAME,
        lr0=LEARNING_RATE, resume=False,
    )
    print(f"Python：{sys.executable}\n项目：{ROOT}\n数据：{DATA}\n初始权重：{WEIGHTS}", flush=True)
    if CHECK_ONLY:
        check_training_input(options, WEIGHTS)
        return
    print(f"开始新阶段训练：{EPOCHS} 轮，imgsz={IMAGE_SIZE}，batch={BATCH_SIZE}，device={DEVICE}", flush=True)
    print("继承最佳模型参数，重新建立优化器；结果保存到新目录。", flush=True)
    model = YOLO(str(WEIGHTS))
    model.train(**options)
    result_dir = model.trainer.save_dir
    print(f"训练完成：{result_dir}\n最佳权重：{result_dir / 'weights/best.pt'}", flush=True)
    if GENERATE_REPORT:
        from report_adas import create_report
        create_report(result_dir)


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
