"""PyCharm 直接运行：优先恢复上次官方灯色训练；已完成则从最佳权重再训练。"""
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from official_light_config import ROOT, POOL, S2TLD, WEIGHTS, NAMES, load_manifest

os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'runs/adas_config'))
os.environ.setdefault('OMP_NUM_THREADS','8')
os.environ.setdefault('MKL_NUM_THREADS','8')
EPOCHS=15
IMAGE_SIZE=640
BATCH_SIZE=2
LEARNING_RATE=0.0003
OFFICIAL_TRAIN_REPEAT=4  # 同一训练图重复采样，不能计作独立样本
CHECK_ONLY=False
CONTINUE_PREVIOUS=True  # 中断：恢复剩余轮数；已完成：从 best.pt 开始新的 EPOCHS 轮
PREVIOUS_RUN=None  # None 自动找最新官方灯色结果，也可指定 ROOT/'runs/adas/s2tld_official_finetune'


def continuation_plan():
    """只读取本项目的官方灯色断点，不选择 BDD 或其他类别模型。"""
    import re
    import torch
    if not CONTINUE_PREVIOUS:
        return dict(mode='new',weights=WEIGHTS)
    if PREVIOUS_RUN is not None:
        run=Path(PREVIOUS_RUN).resolve()
        if not run.is_dir():
            raise FileNotFoundError(f'上次训练目录不存在：{run}')
    else:
        candidates=[]
        for folder in (ROOT/'runs/adas').glob('s2tld_official_finetune*'):
            if folder.is_dir() and re.fullmatch(r's2tld_official_finetune\d*',folder.name):
                checkpoints=[p for p in (folder/'weights/last.pt',folder/'weights/best.pt') if p.is_file()]
                if checkpoints:
                    candidates.append((max(p.stat().st_mtime_ns for p in checkpoints),folder))
        if not candidates:
            return dict(mode='new',weights=WEIGHTS)
        run=max(candidates,key=lambda item:item[0])[1]
    last=run/'weights/last.pt'
    if not last.is_file():
        raise FileNotFoundError(f'缺少最后断点：{last}')
    state=torch.load(last,map_location='cpu',weights_only=False)
    args=state.get('train_args') or {}
    saved_model=state.get('ema') or state.get('model')
    if getattr(saved_model,'names',None)!=dict(enumerate(NAMES)):
        raise ValueError(f'断点类别与红/黄/绿/熄灭不一致：{last}')
    epoch=int(state.get('epoch',-1))
    target=int(args.get('epochs',0))
    has_optimizer=state.get('optimizer') is not None
    del state,saved_model
    if epoch>=0 and epoch+1<target:
        if not has_optimizer:
            raise ValueError(f'未完成断点缺少优化器，不能恢复原进度：{last}')
        data_value=args.get('data')
        if not isinstance(data_value,str) or not Path(data_value).is_file():
            raise FileNotFoundError(f'断点的原训练配置不存在：{data_value}')
        return dict(mode='resume',weights=last,run=run,data=Path(data_value),
                    completed=epoch+1,target=target)
    best=run/'weights/best.pt'
    if not best.is_file():
        raise FileNotFoundError(f'缺少上轮最佳权重：{best}')
    return dict(mode='finetune',weights=best,run=run)


def verify_resume_lists(data_path,selected,s2train):
    """恢复时复用原列表，并核对其仍包含当前审核通过的图片。"""
    import yaml
    data=yaml.safe_load(data_path.read_text(encoding='utf-8'))
    if data.get('names')!=dict(enumerate(NAMES)):
        raise ValueError('原训练配置类别顺序错误')
    for split in ('train','val','test'):
        listing=Path(data[split])
        if not listing.is_absolute():
            listing=Path(data.get('path',data_path.parent))/listing
        paths=[Path(p).resolve() for p in listing.read_text(encoding='utf-8').splitlines() if p.strip()]
        expected=set(selected[split]) | ({p.resolve() for p in s2train} if split=='train' else set())
        if set(paths)!=expected or any(not p.is_file() for p in paths):
            raise ValueError(f'当前数据与上次 {split} 列表不一致；请先核对数据，不能直接续训。')
        if split!='train' and len(paths)!=len(expected):
            raise ValueError(f'原 {split} 列表有重复采样')


def checked_rows():
    if not (POOL/'manifest.json').is_file():
        raise FileNotFoundError('请先抽帧和标注')
    manifest=load_manifest()
    pending=[row for row in manifest['images'] if row['status']=='pending']
    if pending:
        raise ValueError(f"还有 {len(pending)} 张未审核。请标注、确认无灯或排除，不允许把空白当真值。")
    selected={split:[] for split in ('train','val','test')}
    counts={split:Counter() for split in selected}
    groups={split:set() for split in selected}
    hashes={split:set() for split in selected}
    for row in manifest['images']:
        if row['status']=='excluded':
            continue
        if row['status'] not in ('reviewed','negative'):
            raise ValueError(f"未知审核状态：{row['stem']}")
        image=POOL/row['image']
        label=POOL/f"labels/{row['split']}/{row['stem']}.txt"
        if not image.is_file() or not label.is_file():
            raise FileNotFoundError(f'缺少图片/标签：{image}')
        lines=label.read_text(encoding='utf-8').splitlines()
        if bool(lines)!=(row['status']=='reviewed'):
            raise ValueError(f"审核状态与标签不符：{label}")
        for line in lines:
            values=list(map(float,line.split()))
            if len(values)!=5:
                raise ValueError(f'标签格式错误：{label}')
            cls,x,y,w,h=values
            if cls!=int(cls) or not 0<=cls<4 or not all(__import__('math').isfinite(v) for v in values) or not (w>0 and h>0 and x-w/2>=-1e-7 and y-h/2>=-1e-7 and x+w/2<=1+1e-7 and y+h/2<=1+1e-7):
                raise ValueError(f'非法框：{label}')
            counts[row['split']][NAMES[int(cls)]]+=1
        selected[row['split']].append(image.resolve())
        groups[row['split']].add(row['video'])
        hashes[row['split']].add(row['digest'])
    for a,b in [('train','val'),('train','test'),('val','test')]:
        if groups[a]&groups[b] or hashes[a]&hashes[b]:
            raise ValueError(f'{a}/{b} 存在视频或完全重复帧泄漏')
    for split in selected:
        print(f"官方 {split}：{len(selected[split])} 张；目标 {dict(counts[split])}",flush=True)
        if not selected[split] or not counts[split]:
            raise ValueError(f'{split} 没有有效灯标注，请补充')
    return selected,counts


def evaluate(model,data_path,project,name):
    metrics=model.val(data=str(data_path),split='val',imgsz=IMAGE_SIZE,batch=BATCH_SIZE,
                      device='cpu',workers=0,conf=.001,iou=.7,plots=False,
                      project=str(project),name=name,exist_ok=True)
    classes=[]
    for i,cls in enumerate(metrics.ap_class_index):
        p,r,a50,a=metrics.class_result(i)
        classes.append(dict(name=metrics.names[int(cls)],instances=int(metrics.nt_per_class[int(cls)]),precision=float(p),recall=float(r),AP50=float(a50),AP50_95=float(a)))
    return dict(overall={key:float(value) for key,value in metrics.results_dict.items()},classes=classes)


def main():
    import sys
    import torch
    import yaml
    from ultralytics import YOLO
    from mytrain import training_options
    from training_memory_monitor import memory_snapshot,start_monitor
    for output in (sys.stdout,sys.stderr):
        if hasattr(output,'reconfigure'):
            output.reconfigure(encoding='utf-8')
    if min(EPOCHS,IMAGE_SIZE,BATCH_SIZE,OFFICIAL_TRAIN_REPEAT)<1 or not 0<LEARNING_RATE<1:
        raise ValueError('参数不合法')
    selected,counts=checked_rows()
    plan=continuation_plan()
    if not plan['weights'].is_file() or not (S2TLD/'data.yaml').is_file():
        raise FileNotFoundError('缺少已训练的 S²TLD 权重或数据')
    s2train=sorted((S2TLD/'images/train').glob('*.jpg'))
    if not s2train:
        raise ValueError('S²TLD 训练数据为空')
    for image in s2train:
        if not (S2TLD/'labels/train'/f'{image.stem}.txt').is_file():
            raise FileNotFoundError(image)
    if plan['mode']=='resume':
        data_path=plan['data']
        build=data_path.parent
        verify_resume_lists(data_path,selected,s2train)
        print(f"已完成 {plan['completed']}/{plan['target']} 轮；从第 {plan['completed']+1} 轮继续。",flush=True)
        print(f"恢复断点：{plan['weights']}\n沿用数据：{data_path}\n结果目录：{plan['run']}",flush=True)
        print('断点续训沿用原总轮数；顶部 EPOCHS 仅控制新一轮训练。',flush=True)
    else:
        build=ROOT/'datasets/adas/prepared'/datetime.now().strftime('official_light_mix_%Y%m%d_%H%M%S_%f')
        build.mkdir(parents=True)
        split_files={}
        for split in selected:
            paths=([p.resolve() for p in s2train]+selected[split]*OFFICIAL_TRAIN_REPEAT) if split=='train' else selected[split]
            target=build/f'{split}.txt'
            target.write_text('\n'.join(p.as_posix() for p in paths)+'\n',encoding='utf-8')
            split_files[split]=str(target)
        data_path=build/'data.yaml'
        data_path.write_text(yaml.safe_dump(dict(path=build.as_posix(),names=dict(enumerate(NAMES)),**split_files),sort_keys=False),encoding='utf-8')
        (build/'data_report.json').write_text(json.dumps(dict(unique_s2_train=len(s2train),official_images={k:len(v) for k,v in selected.items()},official_boxes={k:dict(v) for k,v in counts.items()},repeat=OFFICIAL_TRAIN_REPEAT,annotation_manifest=str(POOL/'manifest.json'),limitations=['官方验证只包含当前片段中的灯色；缺少类别不参与该集合mAP。','同路程相似画面仍可能跨集合；这些视频已被查看，不能作为盲测。']),ensure_ascii=False,indent=2),encoding='utf-8')
        print(f"从已有权重开始新一轮 {EPOCHS} 轮训练：{plan['weights']}",flush=True)
    torch.set_num_threads(8)
    model=YOLO(str(plan['weights']))
    if CHECK_ONLY:
        from ultralytics.cfg import get_cfg
        from ultralytics.data.utils import check_det_dataset
        from ultralytics.data.build import build_yolo_dataset
        options=training_options(EPOCHS,IMAGE_SIZE,BATCH_SIZE)
        options.update(data=str(data_path))
        data=check_det_dataset(str(data_path),autodownload=False)
        cfg=get_cfg(overrides={**options,'task':'detect','mode':'train'})
        for split in ('train','val','test'):
            dataset=build_yolo_dataset(cfg,data[split],BATCH_SIZE,data,mode='val',rect=True,stride=32)
            print(f'{split} 标签加载：{len(dataset)} 张',flush=True)
        model.predict(str(selected['val'][0]),imgsz=IMAGE_SIZE,device='cpu',verbose=False)
        print(f'检查通过，未训练；配置：{data_path}',flush=True)
        return
    if memory_snapshot()['commit_free_GB']<2:
        raise RuntimeError('提交额度不足2GB，请先处理系统内存与分页文件')
    if plan['mode']=='resume':
        baseline=build/'baseline.json'
        before=json.loads(baseline.read_text(encoding='utf-8')) if baseline.is_file() else None
        if before is None:
            print('原始训练前评估未保存，完成后只报告最终指标。',flush=True)
    else:
        before=evaluate(model,data_path,build,'baseline_official_val')
        (build/'baseline.json').write_text(json.dumps(before,indent=2),encoding='utf-8')
    options=training_options(EPOCHS,IMAGE_SIZE,BATCH_SIZE)
    options.update(data=str(data_path),name='s2tld_official_finetune',lr0=LEARNING_RATE,resume=False)
    stop,thread=start_monitor(ROOT/'runs/official_light_memory_logs')
    try:
        if plan['mode']=='resume':
            model.train(resume=True,device='cpu',batch=BATCH_SIZE,imgsz=IMAGE_SIZE)
        else:
            model.train(**options)
    finally:
        stop.set()
        thread.join(timeout=2)
    run_dir=model.trainer.save_dir
    report_dir=run_dir/'report'
    report_dir.mkdir(exist_ok=True)
    best=YOLO(str(run_dir/'weights/best.pt'))
    after=evaluate(best,data_path,report_dir,'official_val')
    # 同时复核原S²TLD验证集，观察是否遗忘原有能力。
    s2_after=evaluate(best,S2TLD/'data.yaml',report_dir,'s2tld_val')
    result=dict(weights=str(run_dir/'weights/best.pt'),data=str(data_path),before=before,after=after,
                delta_percentage_points={k:100*(after['overall'][k]-v) for k,v in (before['overall'] if before else {}).items() if k.startswith('metrics/')},s2tld_after=s2_after,
                continuation=dict(mode=plan['mode'],starting_weights=str(plan['weights'])),
                test_note='test片段未自动评估，待模型定稿再使用；这些参考视频不属于未知盲测。')
    (report_dir/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'完成：{run_dir}\n比较报告：{report_dir / "comparison.json"}',flush=True)


if __name__=='__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
