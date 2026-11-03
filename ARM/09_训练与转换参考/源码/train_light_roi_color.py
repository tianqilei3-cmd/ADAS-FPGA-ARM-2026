"""PyCharm entry: 5-epoch lamp ROI color experiment, with checkpoints and logs."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT/'runs/adas_config'))
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('MKL_NUM_THREADS', '8')

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from light_roi_common import NAMES, input_rgb, logits_of, make_model, read_image
from prepare_light_roi_color import prepare
from training_memory_monitor import memory_snapshot, start_monitor

EPOCHS = 5
IMAGE_SIZE = 128
BATCH_SIZE = 32
THREADS = 8


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def save_checkpoint(path, value):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)
    def flush(self):
        for stream in self.streams:
            stream.flush()


class Crops(Dataset):
    def __init__(self, folder, rows, size, augment=False):
        self.folder, self.rows, self.size, self.augment = folder, rows, size, augment
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        rgb = input_rgb(read_image(self.folder/row['image']), self.size)
        if self.augment:
            # Preserve hue and all arrow orientations; no random grayscale/crop/flip.
            gain = random.uniform(.85, 1.15)
            rgb = np.clip(rgb.astype(np.float32)*gain, 0, 255).astype(np.uint8)
            dx, dy = random.uniform(-.025,.025)*self.size, random.uniform(-.025,.025)*self.size
            rgb = cv2.warpAffine(rgb, np.float32([[1,0,dx],[0,1,dy]]),
                                 (self.size,self.size), borderValue=(114,114,114))
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).float()/255
        return tensor, row['class_id']


def summary_of(targets, probabilities):
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities)
    predictions = probabilities.argmax(axis=1)
    confusion = np.zeros((4,4), dtype=np.int64)
    np.add.at(confusion, (targets,predictions), 1)
    totals = confusion.sum(axis=1)
    recalls = np.divide(confusion.diagonal(), totals, out=np.zeros(4), where=totals>0)
    return {'samples': len(targets), 'accuracy': float((targets==predictions).mean()),
            'macro_recall_present': float(recalls[totals>0].mean()),
            'confusion_matrix': confusion.tolist(), 'class_order': NAMES,
            'classes': {name:{'samples':int(totals[i]), 'correct':int(confusion[i,i]),
                              'recall':float(recalls[i]) if totals[i] else None} for i,name in enumerate(NAMES)},
            'red_as_green': int(confusion[0,2]),
            'red_as_green_rate':float(confusion[0,2]/totals[0]) if totals[0] else 0.}


def evaluate(model, loader):
    model.eval()
    targets, probabilities = [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits = logits_of(model(images))
            probabilities.extend(logits.softmax(1).cpu().numpy().tolist())
            targets.extend(labels.tolist())
    return summary_of(targets, probabilities), {'targets':targets,'probabilities':probabilities}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch', type=int, default=BATCH_SIZE)
    parser.add_argument('--size', type=int, default=IMAGE_SIZE)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--init', type=Path, help='Initialize new experiment from our classifier checkpoint')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--new-run', action='store_true', help='Start another independent experiment')
    parser.add_argument('--dataset', type=Path, help='Prepared traceable crop dataset; keeps its existing split')
    parser.add_argument('--pointer', type=Path, default=ROOT/'runs/traffic_light_repair_plan_20261017/latest_classifier.json')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 2 or args.size < 64 or args.size%32:
        parser.error('Positive epochs, batch>=2, and size a multiple of32 >=64 required')
    if args.resume and args.init:
        parser.error('Use resume or init')
    torch.set_num_threads(THREADS)
    cv2.setNumThreads(1)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if memory_snapshot()['commit_free_GB'] < 2:
        raise RuntimeError('Less than2GiB free Windows commit')
    if args.dataset is None:
        folder, manifest = prepare()
    else:
        folder=args.dataset.resolve()
        manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        assert manifest['names']==NAMES
        assert all((folder/r['image']).is_file() for r in manifest['records'])
    if args.resume is None and not args.new_run:
        import psutil
        for candidate in sorted((ROOT/'runs/adas').glob('light_roi_color_*'), reverse=True):
            plan_path,status_path=candidate/'plan.json',candidate/'status.json'
            if not plan_path.exists() or not status_path.exists():
                continue
            existing=json.loads(plan_path.read_text(encoding='utf-8'))
            status=json.loads(status_path.read_text(encoding='utf-8'))
            if (existing.get('dataset_fingerprint')!=manifest['fingerprint'] or
                existing.get('epochs')!=args.epochs or existing.get('image_size')!=args.size or
                existing.get('batch')!=args.batch or
                existing.get('initial_checkpoint')!=(str(args.init.resolve()) if args.init else None)):
                continue
            if status['state']=='completed' and not args.check_only:
                print('MATCHING_EXPERIMENT_ALREADY_COMPLETED',candidate,
                      'Use evaluate_light_roi_color.py to inspect results; --new-run starts a separate experiment.',flush=True)
                return
            if status['state']=='running' and psutil.pid_exists(status.get('pid',-1)):
                process=psutil.Process(status['pid'])
                if 'python' in process.name().lower():
                    raise RuntimeError(f'An existing matching experiment is still running: {candidate}')
            if (candidate/'weights/last.pt').exists():
                args.resume=candidate
                args.init=None
                print('AUTO_RESUME',candidate,flush=True)
            break
    rows = {split:[r for r in manifest['records'] if r['split']==split]
            for split in ['train','s2_val','official_val']}
    model = make_model()
    start_epoch, history, best_score = 0, [], -1e9
    generator = torch.Generator().manual_seed(42)
    pretrained_hash = hashlib.sha256((ROOT/'yolov8n-cls.pt').read_bytes()).hexdigest()
    plan = {'architecture':'yolov8n-cls four-class head', 'names':NAMES,
            'image_size':args.size,'batch':args.batch,'epochs':args.epochs,'seed':42,
            'dataset':str(folder),'dataset_fingerprint':manifest['fingerprint'],
            'pretrained_file':str(ROOT/'yolov8n-cls.pt'),'pretrained_sha256':pretrained_hash,
            'preprocess':'BGR crop(padding8%)->RGB letterbox128(or configured size),float32/255; no mean/std',
            'sampling':'inverse sqrt class frequency; official default weight12, explicit per-record sampling_domain_weight honored',
            'augmentation':'brightness .85..1.15, translation +/-2.5%; no hue/gray/flips',
            'selection':'mean source/official macro recall minus half the sum of red-to-green rates',
            'scope':'ROI classification experiment; official validation is reviewed development data; no lane association or final competition accuracy'}
    if args.init:
        checkpoint = torch.load(args.init, map_location='cpu', weights_only=False)
        assert checkpoint['names']==NAMES
        model.load_state_dict(checkpoint['model_state'])
        plan['initial_checkpoint'] = str(args.init.resolve())
    if args.resume:
        run = args.resume.resolve()
        saved_plan = json.loads((run/'plan.json').read_text(encoding='utf-8'))
        for key in ['dataset_fingerprint','image_size','batch','epochs','pretrained_sha256']:
            assert plan[key] == saved_plan[key], f'Resume config mismatch: {key}'
        plan = saved_plan
    else:
        run = ROOT/'runs/adas'/('light_roi_color_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    counts = Counter(r['class_id'] for r in rows['train'])
    sample_weights = [r.get('sampling_domain_weight',12 if r['domain']=='official' else 1)/math.sqrt(counts[r['class_id']])
                      for r in rows['train']]
    sampler = WeightedRandomSampler(sample_weights, len(rows['train']), replacement=True, generator=generator)
    train_loader = DataLoader(Crops(folder,rows['train'],args.size,True),batch_size=args.batch,
                              sampler=sampler,num_workers=0,pin_memory=False)
    val_loaders = {s:DataLoader(Crops(folder,rows[s],args.size),batch_size=args.batch,num_workers=0)
                   for s in ['s2_val','official_val']}
    head_ids = {id(p) for p in model.model[-1].linear.parameters()}
    optimizer = torch.optim.AdamW([
        {'params':[p for p in model.parameters() if id(p) not in head_ids], 'lr':1e-4},
        {'params':list(model.model[-1].linear.parameters()), 'lr':5e-4}],weight_decay=1e-4)
    if args.resume:
        checkpoint = torch.load(run/'weights/last.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch, history, best_score = checkpoint['epoch'], checkpoint['history'], checkpoint['best_score']
        random.setstate(checkpoint['random_state']); np.random.set_state(checkpoint['numpy_state'])
        torch.set_rng_state(checkpoint['torch_state']); generator.set_state(checkpoint['sampler_state'])
    if args.check_only:
        images, labels = next(iter(train_loader))
        model.train()
        result = model(images)
        assert result.shape==(len(images),4) and torch.isfinite(result).all()
        loss = torch.nn.functional.cross_entropy(result,labels)
        loss.backward()
        print('INPUT_AND_GRADIENT_CHECK_PASS', result.shape, float(loss.detach()),memory_snapshot(),flush=True)
        return
    run.mkdir(parents=True, exist_ok=args.resume is not None)
    (run/'weights').mkdir(exist_ok=True)
    save_json(run/'plan.json',plan)
    log = (run/'console.log').open('a',encoding='utf-8',buffering=1)
    previous_out, previous_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = Tee(previous_out,log), Tee(previous_err,log)
    stop, monitor = start_monitor(run/'memory')
    state = {'state':'running','pid':os.getpid(),'run':str(run),'completed_epochs':start_epoch,'target_epochs':args.epochs}
    save_json(run/'status.json',state)
    started=time.perf_counter()
    try:
        print('RUN_DIRECTORY',run,flush=True)
        print('COUNTS',manifest['counts'],'parameters',sum(p.numel() for p in model.parameters()),flush=True)
        for epoch in range(start_epoch,args.epochs):
            epoch_started=time.perf_counter()
            # Frozen pretrained features in the first epoch; full fine-tuning follows.
            for parameter in model.parameters():
                parameter.requires_grad_(epoch>0 or id(parameter) in head_ids)
            model.train()
            for module in model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm): module.eval()
            factor = .2+.8*.5*(1+math.cos(math.pi*epoch/max(1,args.epochs-1)))
            optimizer.param_groups[0]['lr']=1e-4*factor
            optimizer.param_groups[1]['lr']=5e-4*factor
            loss_total, samples = 0.,0
            for step,(images,labels) in enumerate(train_loader):
                optimizer.zero_grad(set_to_none=True)
                logits=model(images)
                loss=torch.nn.functional.cross_entropy(logits,labels)
                if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                optimizer.step()
                loss_total+=float(loss.detach())*len(labels); samples+=len(labels)
                if (step+1)%50==0 or step+1==len(train_loader):
                    state.update(epoch=epoch+1,step=step+1,total_steps=len(train_loader),
                                 loss=loss_total/samples,last_update=datetime.now().isoformat())
                    save_json(run/'status.json',state)
                    print(f'EPOCH {epoch+1}/{args.epochs} step={step+1}/{len(train_loader)} loss={loss_total/samples:.4f} '
                          f'elapsed={time.perf_counter()-epoch_started:.1f}s',flush=True)
            metrics, predictions={},{}
            for split,loader in val_loaders.items():
                metrics[split],predictions[split]=evaluate(model,loader)
            score=.5*sum(m['macro_recall_present'] for m in metrics.values())-.5*sum(m['red_as_green_rate'] for m in metrics.values())
            improved=score>best_score+1e-8
            if improved: best_score=score
            entry={'epoch':epoch+1,'loss':loss_total/samples,'seconds':time.perf_counter()-epoch_started,
                   'selection_score':score,'metrics':metrics,'memory':memory_snapshot()}
            history.append(entry)
            saved={'epoch':epoch+1,'names':NAMES,'image_size':args.size,'model_state':model.state_dict(),
                   'optimizer_state':optimizer.state_dict(),'history':history,'best_score':best_score,
                   'random_state':random.getstate(),'numpy_state':np.random.get_state(),
                   'torch_state':torch.get_rng_state(),'sampler_state':generator.get_state(),'plan':plan}
            save_checkpoint(run/'weights/last.pt',saved)
            if improved:
                save_checkpoint(run/'weights/best.pt',saved)
                save_json(run/'best_validation_predictions.json',predictions)
                save_json(run/'best_metrics.json',entry)
            save_json(run/'history.json',history)
            state.update(completed_epochs=epoch+1,best_score=best_score)
            save_json(run/'status.json',state)
            print('EPOCH_RESULT',json.dumps(entry,ensure_ascii=False),flush=True)
        state.update(state='completed',elapsed_seconds=time.perf_counter()-started,best_checkpoint=str(run/'weights/best.pt'))
        save_json(run/'status.json',state)
        save_json(args.pointer,
                  {'run':str(run),'best':str(run/'weights/best.pt'),'state':'completed','dataset':str(folder)})
        print('TRAINING_COMPLETED',json.dumps(state,ensure_ascii=False),flush=True)
    except BaseException as exc:
        state.update(state='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc))
        save_json(run/'status.json',state)
        traceback.print_exc()
        raise
    finally:
        stop.set(); monitor.join(timeout=2)
        sys.stdout,sys.stderr=previous_out,previous_err
        log.close()


if __name__=='__main__':
    main()
