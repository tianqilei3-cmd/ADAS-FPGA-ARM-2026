"""CPU fixed12-epoch small-head repair; existing TRAIN only, no board access."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
import hashlib,json,time,sys,re
from collections import Counter
from pathlib import Path
import cv2,numpy as np,torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.data.build import build_dataloader

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/basic_finish70_20261028/lamp_capture_aligned'
INITIAL=ROOT/'runs/adas/s2tld_official_tiles_stage2/weights/deployment_best.pt'
SHAPE=(256,640)
TRAIN_OVERRIDES={}
GREEN_REPEAT=1


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8');temp.replace(p)


class RectTrainer(DetectionTrainer):
    def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode='train'):
        assert rank==-1
        dataset=self.build_dataset(dataset_path,mode,batch_size)
        assert len(np.unique(dataset.batch_shapes,axis=0))==1
        if mode=='train':assert list(dataset.batch_shapes[0])==list(SHAPE)
        return build_dataloader(dataset,batch_size,self.args.workers,mode=='train',rank)
    def preprocess_batch(self,batch):
        result=super().preprocess_batch(batch);assert tuple(result['img'].shape[2:])==SHAPE;return result


def prepare():
    target=OUT/'dataset_manifest.json'
    if target.exists():return json.loads(target.read_text())
    original=ROOT/'runs/fast_lamp_rect_20261024/dataset_manifest.json';source=json.loads(original.read_text())
    rows=[];listing={'train':[],'val':[]};hashes={'train':set(),'val':set()}
    for index,record in enumerate(source['records']):
        split=record['split'];video=record.get('video') or ''
        if split=='train':assert not any(clip in video for clip in ['112454','113421','113539'])
        im=cv2.imdecode(np.fromfile(record['source'],dtype=np.uint8),cv2.IMREAD_COLOR);assert im is not None
        actual=hashlib.sha256(im.tobytes()).hexdigest();assert actual==record['source_pixel_sha256'];hashes[split].add(actual)
        a,b,c,d=record['source_window'];patch=im[b:d,a:c];h,w=patch.shape[:2]
        gain=min(SHAPE[1]/w,SHAPE[0]/h);rw,rh=round(w*gain),round(h*gain)
        left,top=round((SHAPE[1]-rw)/2-.1),round((SHAPE[0]-rh)/2-.1)
        prepared=cv2.copyMakeBorder(cv2.resize(patch,(rw,rh)),top,SHAPE[0]-rh-top,left,SHAPE[1]-rw-left,cv2.BORDER_CONSTANT,value=(114,114,114))
        labels=[]
        for cls,cx,cy,bw,bh in record['source_labels']:
            box=[(cx*w*gain+left)/SHAPE[1],(cy*h*gain+top)/SHAPE[0],bw*w*gain/SHAPE[1],bh*h*gain/SHAPE[0]]
            assert 0<=cls<4 and all(0<=x<=1 for x in box)
            labels.append(str(int(cls))+' '+' '.join(f'{x:.10f}' for x in box))
        for variant,ratio in enumerate([1.,.8,.65] if split=='train' else [1.]):
            image=prepared
            if ratio<1:
                image=cv2.resize(cv2.resize(image,(round(SHAPE[1]*ratio),round(SHAPE[0]*ratio)),interpolation=cv2.INTER_AREA),(SHAPE[1],SHAPE[0]),interpolation=cv2.INTER_LINEAR)
                image=cv2.GaussianBlur(image,(3,3),.45)
            name=f'{index:04d}_{variant}';path=OUT/'dataset/images'/split/(name+'.png');label=OUT/'dataset/labels'/split/(name+'.txt')
            path.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
            ok,data=cv2.imencode('.png',image);assert ok;data.tofile(str(path));label.write_text('\n'.join(labels)+('\n' if labels else ''),encoding='utf8')
            repeat=record['repeat']*(GREEN_REPEAT if split=='train' and any(int(l[0])==2 for l in record['source_labels']) else 1)
            listing[split].extend([path.as_posix()]*repeat)
            rows.append(dict(split=split,image=str(path),label=str(label),source=record['source'],source_pixel_sha256=actual,video=video,scale_simulation=ratio,source_labels_unchanged=True,repeat=repeat))
    assert not hashes['train']&hashes['val']
    for split,paths in listing.items():(OUT/'dataset'/(split+'.txt')).write_text('\n'.join(paths)+'\n',encoding='utf8')
    (OUT/'dataset/data.yaml').write_text('path: '+(OUT/'dataset').as_posix()+'\ntrain: train.txt\nval: val.txt\nnames:\n  0: red\n  1: yellow\n  2: green\n  3: off\n',encoding='utf8')
    manifest=dict(records=rows,sampling_rows={s:len(v) for s,v in listing.items()},unique_sources={s:len(v) for s,v in hashes.items()},
        source_manifest_sha256=digest(original),held_video_train_exclusion=['112454','113421','113539'],train_val_source_overlap=0,
        green_train_repeat=GREEN_REPEAT,
        scope='Three deterministic pixel-quality views of existing TRAIN; no new independent scenes; original val untouched; not acceptance video training')
    save(target,manifest);return manifest


def main():
    torch.set_num_threads(4);cv2.setNumThreads(1);OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'trial/weights/last.pt').exists(),'Preserve previous trial; inspect before resuming'
    data=prepare();plan=dict(initial=str(INITIAL),initial_sha256=digest(INITIAL),shape=SHAPE,epochs=12,batch=4,nbs=4,freeze=10,
        threads=4,workers=0,cache=False,checkpoint_selection='Fixed last epoch12',dataset_sha256=digest(OUT/'dataset_manifest.json'),automatic_deployment=False)
    plan.update(training_overrides=TRAIN_OVERRIDES,green_train_repeat=GREEN_REPEAT)
    for k in ['freeze','lr0']:
        if k in TRAIN_OVERRIDES:plan[k]=TRAIN_OVERRIDES[k]
    save(OUT/'plan.json',plan);status=dict(state='starting',pid=os.getpid(),epochs_completed=0,updates=0,started_unix=time.time());save(OUT/'status.json',status)
    model=YOLO(str(INITIAL));began=time.monotonic()
    def begin(trainer):
        assert trainer.accumulate==1 and trainer.model.model[-1].nc==4
        status.update(state='training',batches_per_epoch=len(trainer.train_loader),trainable_parameters=sum(p.numel() for p in trainer.model.parameters() if p.requires_grad))
        trainer.optimizer.register_step_post_hook(lambda *_:status.update(updates=status['updates']+1));save(OUT/'status.json',status)
    def epoch_end(trainer):
        status.update(epochs_completed=trainer.epoch+1,elapsed_seconds=time.monotonic()-began);save(OUT/'status.json',status)
        print('ALIGNED_LAMP_EPOCH',trainer.epoch+1,'UPDATES',status['updates'],'SECONDS',round(status['elapsed_seconds'],1),flush=True)
    model.add_callback('on_train_start',begin);model.add_callback('on_train_epoch_end',epoch_end)
    try:
        options=dict(trainer=RectTrainer,data=str(OUT/'dataset/data.yaml'),epochs=12,batch=4,nbs=4,imgsz=640,rect=True,
            device='cpu',workers=0,freeze=10,optimizer='AdamW',lr0=.00015,lrf=.2,warmup_epochs=0.,warmup_bias_lr=0.,
            project=str(OUT),name='trial',exist_ok=True,cache=False,amp=False,plots=False,patience=13,close_mosaic=0,mosaic=0.,mixup=0.,
            fliplr=0.,flipud=0.,degrees=0.,translate=.015,scale=.05,hsv_h=0.,hsv_s=.15,hsv_v=.15,seed=20261028,verbose=False,val=False,save_period=-1)
        options.update(TRAIN_OVERRIDES);model.train(**options)
        weights=OUT/'trial/weights/last.pt';assert status['epochs_completed']==12;assert digest(INITIAL)==plan['initial_sha256']
        status.update(state='completed',weights=str(weights),weights_sha256=digest(weights),elapsed_seconds=time.monotonic()-began,adopted=False);save(OUT/'status.json',status);print(json.dumps(status),flush=True)
    except BaseException as exc:
        status.update(state='failed',error=repr(exc));save(OUT/'status.json',status);raise


if __name__=='__main__':main()
