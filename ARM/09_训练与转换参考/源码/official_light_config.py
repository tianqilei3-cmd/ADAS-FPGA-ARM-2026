"""官方灯色专项数据公共配置。修改抽帧/划分参数时使用新的 POOL。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VIDEOS = Path("D:/BaiduNetdiskDownload/紫光同创杯赛题三参考素材")
POOL = ROOT / "datasets/adas/prepared/official_light_annotation_v1"
S2TLD = ROOT / "datasets/adas/prepared/s2tld_light_states"
WEIGHTS = ROOT / "runs/adas/s2tld_light_states/weights/best.pt"
NAMES = ["red", "yellow", "green", "off"]
SAMPLE_SECONDS = 1.0
TOP_RATIO = 0.60  # 与 demo_official_videos.py 中灯模型输入区域一致
SPLITS = {
    "train": ["112507", "112739", "113343", "113452"],
    "val": ["113539"],
    "test": ["112454", "113421"],
}


def load_manifest():
    import json
    return json.loads((POOL / "manifest.json").read_text(encoding="utf-8"))


def save_manifest(manifest):
    import json
    temporary = POOL / "manifest.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(POOL / "manifest.json")
