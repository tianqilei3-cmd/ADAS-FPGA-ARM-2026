"""Isolated sparse raw-head export and mixed-scene calibration; no PT changes."""
import hashlib,json,os,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'runs/orangepi/export_dependencies'))
os.environ['YOLO_AUTOINSTALL']='false';os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'runs/adas_config'))
import cv2,numpy as np,torch
from ultralytics import YOLO
OUT=ROOT/'runs/basic_finish70_20261027/s3/lamp_int8_heads'

class Heads(torch.nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,x):
        _,features=self.model(x)
        return tuple(v for f in features for v in (f[:,:64],f[:,64:].sigmoid()))

def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'lamp_rect_heads.onnx').exists()
    torch.set_num_threads(6)
    source=ROOT/'runs/adas/s2tld_official_tiles_stage2/weights/deployment_best.pt'
    digest=hashlib.sha256(source.read_bytes()).hexdigest();model=Heads(YOLO(source).model.eval()).eval()
    onnx=OUT/'lamp_rect_heads.onnx'
    torch.onnx.export(model,torch.zeros(1,3,256,640),str(onnx),input_names=['images'],
        output_names=['box0','score0','box1','score1','box2','score2'],opset_version=12,dynamic_axes=None)
    sys.path.insert(0,str(ROOT/'arm_screen_demo_a18_20261027'))
    from vision_common import preprocess
    folder=OUT/'calibration';folder.mkdir();records=[]
    videos=sorted(Path('D:/BaiduNetdiskDownload/紫光同创杯赛题三参考素材').glob('*.mp4'))
    for video in videos:
        cap=cv2.VideoCapture(str(video));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame_id in np.linspace(0,count-1,12,dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(frame_id));ok,image=cap.read()
            if not ok:continue
            image=cv2.resize(image,(960,540));image=image[:324]
            for tile in [image,image[:,240:720]]:
                rgb,_=preprocess(tile,[256,640]);name=f'calibration/{len(records):04d}.jpg'
                assert cv2.imwrite(str(OUT/name),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,85])
                records.append(dict(path=name,source=str(video),frame=int(frame_id)))
        cap.release()
    assert len(records)>=100
    selected=records[::2][:96]+records[1::4][:32]
    (OUT/'calibration.json').write_text(json.dumps(selected,indent=2,ensure_ascii=False),encoding='utf8')
    archive=OUT/'conversion_inputs.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:
        tf.add(onnx,arcname=onnx.name)
        for item in selected:tf.add(OUT/item['path'],arcname=item['path'])
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    result=dict(source=str(source),source_sha256=digest,onnx=str(onnx),
        onnx_sha256=hashlib.sha256(onnx.read_bytes()).hexdigest(),input_shape=[1,3,256,640],
        calibration_samples=len(selected),archive=str(archive),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        archive_bytes=archive.stat().st_size,accepted=False,new_training=False)
    (OUT/'export.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result))

if __name__=='__main__':main()
