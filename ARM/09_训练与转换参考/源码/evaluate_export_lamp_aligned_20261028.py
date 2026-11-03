"""Same-input held-out coverage comparison, then FP raw-head export only."""
import hashlib,json,os,sys,tarfile,time
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'runs/orangepi/export_dependencies'))
os.environ['YOLO_AUTOINSTALL']='false';os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'runs/adas_config'))
import cv2,numpy as np,torch
from ultralytics import YOLO
sys.path.insert(0,str(ROOT/'arm_unified_20261028'))
from vision_common import preprocess,decoded_output,decode,match_detections
from train_lamp_capture_aligned_20261028 import OUT,INITIAL,save,digest
from export_signal_int8_heads_20261027 import Heads


def box_iou(a,b):
    a=np.asarray(a);b=np.asarray(b);inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    return float(inter/max(1e-7,np.maximum(a[2:]-a[:2],0).prod()+np.maximum(b[2:]-b[:2],0).prod()-inter))


def infer(model,image):
    rgb,g=preprocess(image,[256,640])
    with torch.inference_mode():outputs=model(torch.from_numpy(rgb).permute(2,0,1)[None].float()/255)
    heads=[a.detach().numpy() for a in outputs]
    return decode(decoded_output(heads,[256,640],4,score_threshold=.1),g,['red','yellow','green','off'],.1,.7,30)


def main():
    state=json.loads((OUT/'status.json').read_text());assert state['state']=='completed'
    weights=Path(state['weights']);assert digest(weights)==state['weights_sha256']
    folder=OUT/'export';folder.mkdir(exist_ok=True);assert not (folder/'export.json').exists()
    torch.set_num_threads(4);cv2.setNumThreads(1)
    old=Heads(YOLO(INITIAL).model.eval()).eval();new=Heads(YOLO(weights).model.eval()).eval()
    data=json.loads((OUT/'dataset_manifest.json').read_text());records=[r for r in data['records'] if r['split']=='val']
    totals={'retained':Counter(),'trained':Counter()};evaluations=[]
    for index,r in enumerate(records):
        image=cv2.imdecode(np.fromfile(r['image'],dtype=np.uint8),cv2.IMREAD_COLOR);targets=[]
        for line in Path(r['label']).read_text().splitlines():
            cls,cx,cy,w,h=map(float,line.split());targets.append(dict(class_id=int(cls),box=[(cx-w/2)*640,(cy-h/2)*256,(cx+w/2)*640,(cy+h/2)*256]))
        pair=[]
        for label,model in [('retained',old),('trained',new)]:
            pred=[p for p in infer(model,image) if p['confidence']>=.25];options=sorted([(box_iou(p['box'],g['box']),pi,gi) for pi,p in enumerate(pred) for gi,g in enumerate(targets)],reverse=True)
            usedp=set();usedg=set();matched=[]
            for overlap,pi,gi in options:
                if overlap<.5:break
                if pi in usedp or gi in usedg:continue
                usedp.add(pi);usedg.add(gi);matched.append((pi,gi))
            counts=dict(images=1,truth=len(targets),predictions=len(pred),matched=len(matched),unmatched=len(pred)-len(matched),matched_color_correct=sum(pred[p]['class_id']==targets[g]['class_id'] for p,g in matched))
            totals[label].update(counts);pair.append(dict(model=label,counts=counts,predictions=pred))
        evaluations.append(dict(image=r['image'],video=r['video'],comparison=pair))
        if index%25==0:print('ALIGNED_EVAL',index+1,len(records),flush=True)
    report=dict(scope='Existing held video/source val roles; repeated development, not independent road accuracy',reference_checkpoint=str(INITIAL),reference_sha256=digest(INITIAL),candidate_sha256=digest(weights),totals={k:dict(v) for k,v in totals.items()},records=evaluations,adopted=False)
    save(OUT/'validation.json',report);print(json.dumps(dict(validation_totals=report['totals'])),flush=True)
    onnx=folder/'lamp_aligned_heads.onnx'
    torch.onnx.export(new,torch.zeros(1,3,256,640),str(onnx),input_names=['images'],output_names=['box0','score0','box1','score1','box2','score2'],opset_version=12,dynamic_axes=None)
    parity=folder/'parity';parity.mkdir(exist_ok=True);cases=[]
    indices=np.linspace(0,len(records)-1,16,dtype=int)
    for i in indices:
        image=cv2.imdecode(np.fromfile(records[int(i)]['image'],dtype=np.uint8),cv2.IMREAD_COLOR);name=f'{int(i):03d}.png';assert cv2.imwrite(str(parity/name),image)
        cases.append(dict(image=name,reference=infer(new,image)))
    save(parity/'cases.json',cases)
    archive=folder/'inputs.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:
        tf.add(onnx,arcname=onnx.name)
        for p in parity.iterdir():tf.add(p,arcname='parity/'+p.name)
    export=dict(weights=str(weights),weights_sha256=digest(weights),onnx_sha256=digest(onnx),archive=str(archive),archive_sha256=digest(archive),
        shape=[256,640],quantization=False,calibration_samples=0,adopted=False,validation_totals=report['totals'])
    save(folder/'export.json',export);print(json.dumps(export),flush=True)


if __name__=='__main__':main()
