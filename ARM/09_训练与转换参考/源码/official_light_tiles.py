"""官方灯色第二轮：训练、验证与视频共用的固定道路分块。"""
import math
import os

from official_light_config import ROOT, NAMES

os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runs/adas_config"))
TILE_VERSION = "horizontal_half_width_overlap_v1"
MERGE_IOU = 0.50
MAX_DETECTIONS = 300


def tile_windows(width, height):
    """左／中／右三个半宽窗口，保持道路区域的完整高度，与目标位置无关。"""
    if width < 2 or height < 1:
        raise ValueError("道路图像尺寸不合法")
    tile_width = (width + 1) // 2
    starts = sorted({0, (width - tile_width) // 2, width - tile_width})
    return [(x, 0, x + tile_width, height) for x in starts]


def read_boxes(label_path, width, height):
    """YOLO归一化标签转像素[class,x1,y1,x2,y2]。"""
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = list(map(float, line.split()))
        if len(values) != 5 or not all(math.isfinite(v) for v in values):
            raise ValueError(f"标签格式错误：{label_path}")
        cls, x, y, w, h = values
        if cls != int(cls) or not 0 <= cls < len(NAMES) or w <= 0 or h <= 0:
            raise ValueError(f"类别或框尺寸错误：{label_path}")
        if min(x-w/2, y-h/2) < -1e-7 or max(x+w/2, y+h/2) > 1+1e-7:
            raise ValueError(f"框越界：{label_path}")
        boxes.append([int(cls), max(0., (x-w/2)*width), max(0., (y-h/2)*height),
                      min(float(width), (x+w/2)*width), min(float(height), (y+h/2)*height)])
    return boxes


def crop_labels(boxes, window):
    """返回裁剪后的归一化标签；任一已知灯被切断则跳过整个窗口。

    不能将切断的灯忽略后把剩余可见灯孔作为负样本训练。
    """
    x0, y0, xend, yend = window
    width, height = xend-x0, yend-y0
    if min(width, height) <= 0:
        raise ValueError("裁剪窗口不合法")
    labels = []
    for cls, x1, y1, x2, y2 in boxes:
        if min(x2, xend) <= max(x1, x0) or min(y2, yend) <= max(y1, y0):
            continue
        if x1 < x0-1e-4 or y1 < y0-1e-4 or x2 > xend+1e-4 or y2 > yend+1e-4:
            return None
        x1, y1, x2, y2 = max(x1, x0), max(y1, y0), min(x2, xend), min(y2, yend)
        labels.append([cls, ((x1+x2)/2-x0)/width, ((y1+y2)/2-y0)/height,
                       (x2-x1)/width, (y2-y1)/height])
    return labels


def predict_lights(model, road_image, imgsz=640, conf=.25, device="cpu", tiled=True,
                   predict_iou=.7):
    """整幅道路区域＋固定三分块推理，回映坐标并对跨窗口重复框去重。

    各输入顺序执行，减少峰值内存；相同位置的不同颜色也参与去重。
    tile内部边界上的截断预测丢弃，整图和重叠窗口用于补充。
    """
    import numpy as np
    import torch
    from torchvision.ops import nms

    height, width = road_image.shape[:2]
    windows = [(0, 0, width, height)]
    if tiled:
        windows += tile_windows(width, height)
    outputs = []
    for i, (x0, y0, xend, yend) in enumerate(windows):
        result = model.predict(road_image[y0:yend, x0:xend], imgsz=imgsz, conf=conf,
                               iou=predict_iou, device=device, verbose=False,
                               max_det=MAX_DETECTIONS)[0]
        boxes = result.boxes.data.detach().cpu().clone()
        if not len(boxes):
            continue
        if i:
            # 只过滤内部裁剪边缘，不过滤原图边缘的合法可见目标。
            keep = torch.ones(len(boxes), dtype=torch.bool)
            if x0 > 0:
                keep &= boxes[:, 0] > 1.
            if xend < width:
                keep &= boxes[:, 2] < xend-x0-1.
            boxes = boxes[keep]
        if not len(boxes):
            continue
        boxes[:, [0, 2]] += x0
        boxes[:, [1, 3]] += y0
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, height)
        outputs.append(boxes)
    if not outputs:
        return np.empty((0, 6), dtype=np.float32)
    predictions = torch.cat(outputs).float()
    if tiled:
        keep = nms(predictions[:, :4], predictions[:, 4], MERGE_IOU)[:MAX_DETECTIONS]
        predictions = predictions[keep]
    return predictions.numpy()


def prediction_rows(predictions):
    return [dict(name=NAMES[int(row[5])], confidence=round(float(row[4]), 4),
                 box=[float(v) for v in row[:4]]) for row in predictions]


def evaluate_official(model, image_paths, output_dir, tiled=True, imgsz=640):
    """在原始官方val真值上计算指标；固定分块选择不读取真值框。"""
    import json
    import time
    import numpy as np
    import torch
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.utils.metrics import DetMetrics, box_iou
    from ultralytics.utils.patches import imread

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = DetMetrics(names=dict(enumerate(NAMES)))
    matcher = BaseValidator.__new__(BaseValidator)
    matcher.iouv = torch.linspace(.5, .95, 10)
    elapsed = []
    records = []
    for i, image in enumerate(image_paths):
        frame = imread(str(image))
        if frame is None:
            raise RuntimeError(f"无法读取验证图：{image}")
        height, width = frame.shape[:2]
        label = image.parents[2] / "labels" / image.parent.name / (image.stem+".txt")
        targets = np.asarray(read_boxes(label, width, height), dtype=np.float32).reshape(-1, 5)
        started = time.perf_counter()
        predictions = predict_lights(model, frame, imgsz, conf=.001, tiled=tiled)
        elapsed.append((time.perf_counter()-started)*1000)
        if len(targets) and len(predictions):
            ious = box_iou(torch.from_numpy(targets[:, 1:]), torch.from_numpy(predictions[:, :4]))
            tp = matcher.match_predictions(torch.from_numpy(predictions[:, 5]),
                                           torch.from_numpy(targets[:, 0]), ious).numpy()
        else:
            tp = np.zeros((len(predictions), 10), dtype=bool)
        metrics.update_stats(dict(tp=tp, conf=predictions[:, 4], pred_cls=predictions[:, 5],
                                  target_cls=targets[:, 0], target_img=np.unique(targets[:, 0])))
        records.append(dict(image=str(image), predictions=prediction_rows(predictions)))
        if (i+1) % 10 == 0 or i+1 == len(image_paths):
            print(f"{'分块' if tiled else '整图'}官方验证：{i+1}/{len(image_paths)}", flush=True)
    metrics.process(plot=False)
    classes = []
    for i, cls in enumerate(metrics.ap_class_index):
        p, r, a50, a = metrics.class_result(i)
        classes.append(dict(name=NAMES[int(cls)], instances=int(metrics.nt_per_class[int(cls)]),
                            precision=float(p), recall=float(r), AP50=float(a50), AP50_95=float(a)))
    report = dict(overall={k: float(v) for k, v in metrics.results_dict.items()}, classes=classes,
                  images=len(image_paths), tiled=tiled, imgsz=imgsz,
                  mean_inference_ms=float(np.mean(elapsed)),
                  coordinate_space="original_official_road_crop_pixels",
                  tile_version=TILE_VERSION, merge_iou=MERGE_IOU,
                  note="真值只用于匹配评分；输入窗口与目标位置无关。conf=.001用于AP计算，演示默认.25。")
    (output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "predictions.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
