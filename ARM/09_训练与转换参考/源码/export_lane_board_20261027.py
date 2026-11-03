"""Export the existing typed-boundary checkpoint; no training or model search."""
import hashlib,json,os,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'runs/orangepi/export_dependencies'))
os.environ['YOLO_AUTOINSTALL']='false';os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'runs/adas_config'))
import cv2,numpy as np,torch
from lane_boundary_segmentation import BoundaryPredictor,image_tensor
OUT=ROOT/'runs/basic_finish70_20261027/lane_keys/model'

def main():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'export.json').exists()
    cfg=json.loads((ROOT/'runs/lane_segmentation_20261017/pc_geometry_candidate.json').read_text(encoding='utf8'))
    weights=Path(cfg['weights']);assert hashlib.sha256(weights.read_bytes()).hexdigest()==cfg['sha256']
    torch.set_num_threads(6);cv2.setNumThreads(1)
    predictor=BoundaryPredictor(weights);w,h=predictor.size
    if not (OUT/'boundary.onnx').exists():
        torch.onnx.export(predictor.model,torch.zeros(1,3,h,w),str(OUT/'boundary.onnx'),
            input_names=['images'],output_names=['logits'],opset_version=12,dynamic_axes=None)
    folder=OUT/'parity';folder.mkdir(exist_ok=True);cases=[]
    for clip,fids in [('113343',[120,360,600,960]),('113539',[318,354,600,1170])]:
        cap=cv2.VideoCapture('D:/BaiduNetdiskDownload/紫光同创杯赛题三参考素材/video_20260929_'+clip+'.mp4')
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if any(fid>=count for fid in fids):fids=np.linspace(0,count-1,4,dtype=int).tolist()
        for fid in fids:
            cap.set(cv2.CAP_PROP_POS_FRAMES,fid);ok,image=cap.read();assert ok
            image=cv2.resize(image,(640,360));name=f'{clip}_{fid}.png';assert cv2.imwrite(str(folder/name),image)
            with torch.inference_mode():logits=predictor.model(image_tensor([cv2.resize(image,(w,h))]))[0].numpy()
            np.savez_compressed(folder/(name+'.npz'),logits=logits)
            cases.append(dict(image=name,reference=name+'.npz'))
        cap.release()
    (folder/'cases.json').write_text(json.dumps(cases,indent=2),encoding='utf8')
    archive=OUT/'inputs.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:
        tf.add(OUT/'boundary.onnx',arcname='boundary.onnx')
        for p in folder.iterdir():tf.add(p,arcname='parity/'+p.name)
    report=dict(weights=str(weights),weights_sha256=cfg['sha256'],size=[w,h],names=cfg['names'],
        onnx_sha256=hashlib.sha256((OUT/'boundary.onnx').read_bytes()).hexdigest(),
        archive=str(archive),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        new_training=False,physical_lane_association_verified=False,accepted=False)
    (OUT/'export.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
