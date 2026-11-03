"""Portable lamp tiling and coordinate merging, matching the selected PC strategy."""
import cv2
import json
import numpy as np
from pathlib import Path
from vision_common import box_iou, decode, decoded_output, preprocess


def check_light_precision(root, precision):
    status = Path(root)/'deployment_status.json'
    if precision == 'int8_heads' and status.exists():
        variant = json.loads(status.read_text(encoding='utf-8')).get('variants',{}).get(precision,{})
        if variant.get('status') == 'rejected':
            raise RuntimeError('Lamp INT8 candidate rejected: '+variant.get('reason','conversion parity failed'))


def light_windows(width, height):
    if width < 2 or height < 1:
        raise ValueError('Invalid road image dimensions')
    half = (width + 1) // 2
    starts = sorted({0, (width-half)//2, width-half})
    return [(0, 0, width, height)] + [(x, 0, x+half, height) for x in starts]


def merge_lights(items, iou=0.5, maximum=300):
    if not items:
        return []
    boxes = np.asarray([x['box'] for x in items], dtype=np.float32)
    scores = np.asarray([x['confidence'] for x in items], dtype=np.float32)
    order = np.argsort(-scores, kind='stable')
    kept = []
    while len(order) and len(kept) < maximum:
        current = int(order[0])
        kept.append(items[current])
        rest = order[1:]
        order = rest[box_iou(boxes[current], boxes[rest]) <= iou]
    return kept


def detect_lights(image, predict, metadata):
    height, width = image.shape[:2]
    items = []
    for index, (left, top, right, bottom) in enumerate(light_windows(width, height)):
        rgb, geometry = preprocess(image[top:bottom, left:right], metadata['image_size'])
        raw = decoded_output(predict(rgb), metadata['image_size'], len(metadata['names']),
                             score_threshold=metadata['conf'])
        detections = decode(raw, geometry, metadata['names'], metadata['conf'],
                            metadata['iou'], metadata['max_det'])
        for detection in detections:
            x1,y1,x2,y2 = detection['box']
            if index and ((left > 0 and x1 <= 1) or (right < width and x2 >= right-left-1)):
                continue
            items.append({**detection, 'box':[x1+left,y1+top,x2+left,y2+top]})
    return merge_lights(items, metadata['tile_merge_iou'], metadata['max_det'])


def draw_lights(image, lights):
    canvas = image.copy()
    colors = {'red':(0,0,255), 'yellow':(0,230,255), 'green':(0,210,0), 'off':(150,150,150)}
    for item in lights:
        x1,y1,x2,y2 = [round(x) for x in item['box']]
        color = colors[item['name']]
        cv2.rectangle(canvas,(x1,y1),(x2,y2),color,2)
        cv2.putText(canvas,f"{item['name']} {item['confidence']:.2f}",
                    (x1,max(18,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2)
    return canvas
