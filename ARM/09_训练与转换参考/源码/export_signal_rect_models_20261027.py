"""Preserve object pixel scale while eliminating square padding computation."""
import hashlib,json,os,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'runs/orangepi/export_dependencies'))
os.environ['YOLO_AUTOINSTALL']='false';os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'runs/adas_config'))
import torch
from ultralytics import YOLO
OUT=ROOT/'runs/basic_finish70_20261027/s3/rect_models'


def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(6);records=[]
    for name,source,height in [
        ('light_rect',ROOT/'runs/adas/s2tld_official_tiles_stage2/weights/deployment_best.pt',256),
        ('road_rect',ROOT/'runs/adas/bdd100k_4class_stage42/weights/best.pt',384)]:
        target=OUT/(name+'.pt');assert not target.exists(),'Existing export preserved'
        digest=hashlib.sha256(source.read_bytes()).hexdigest();shutil.copy2(source,target)
        path=YOLO(str(target)).export(format='onnx',imgsz=[height,640],batch=1,opset=12,
                                    simplify=False,dynamic=False,device='cpu',half=False)
        assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
        records.append(dict(name=name,source=str(source),source_sha256=digest,height=height,width=640,
            onnx=str(path),onnx_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()))
    (OUT/'export.json').write_text(json.dumps(dict(records=records,new_training=False,accepted=False),indent=2),encoding='utf8')
    print(json.dumps(dict(records=records,new_training=False)),flush=True)


if __name__=='__main__':main()
