"""Same road weights/shape, export sparse raw heads to avoid dense DFL work."""
import json,hashlib,os,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'runs/orangepi/export_dependencies'))
os.environ['YOLO_AUTOINSTALL']='false';os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'runs/adas_config'))
import torch,cv2,numpy as np
from ultralytics import YOLO
from export_signal_int8_heads_20261027 import Heads
OUT=ROOT/'runs/basic_finish70_20261028/road_sparse_heads'
REMOTE='/home/orangepi/adas2026/road_sparse_heads_20261028'


def main():
    OUT.mkdir(exist_ok=True);target=OUT/'road_heads.onnx';assert not target.exists()
    source=ROOT/'runs/adas/bdd100k_4class_stage42/weights/best.pt';digest=hashlib.sha256(source.read_bytes()).hexdigest()
    model=Heads(YOLO(source).model.eval()).eval();torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.onnx.export(model,torch.zeros(1,3,384,640),str(target),input_names=['images'],output_names=['box0','score0','box1','score1','box2','score2'],opset_version=12,dynamic_axes=None)
    parity=OUT/'parity';parity.mkdir();cases=[]
    for video in sorted(Path('D:/BaiduNetdiskDownload/紫光同创杯赛题三参考素材').glob('*.mp4')):
        cap=cv2.VideoCapture(str(video));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for fid in np.linspace(0,count-1,4,dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(fid));ok,im=cap.read();assert ok
            name=f'{video.stem}_{fid}.png';assert cv2.imwrite(str(parity/name),cv2.resize(im,(640,360)))
            cases.append(dict(image=name,source=str(video),frame=int(fid)))
        cap.release()
    (parity/'cases.json').write_text(json.dumps(cases,indent=2),encoding='utf8')
    archive=OUT/'inputs.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:
        tf.add(target,arcname=target.name)
        for p in parity.iterdir():tf.add(p,arcname='parity/'+p.name)
    assert digest==hashlib.sha256(source.read_bytes()).hexdigest()
    result=dict(source=str(source),source_sha256=digest,onnx_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),input_shape=[1,3,384,640],
        quantized=False,new_weights=False,archive=str(archive),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),samples=len(cases),adopted=False)
    (OUT/'export.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
