"""Two eight-epoch frozen-feature trials; final errors/coverage drive selection."""
from collections import Counter
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from prepare_type_direction_repair_pycharm import ROOT,OUT,prepare,digest,save
from signal_type_direction_repair import (PARENT,PARENT_SHA,make_type_head,make_direction_head,decode,summarize)
from signal_attribute_consensus import ConsensusAttributeClassifier
from signal_dual_view_residual import encode
from light_roi_common import read_image,crop_head
from train_visual_lane_relations_pycharm import atomic_torch_save

EPOCHS=8
SEED=20261022
CALIBRATION={'domestic_calibration','original_validation','calibration'}


def feature_cache(reader,manifest):
    rows=manifest['records'];path=OUT/'features.pt'
    config=dict(manifest_sha256=digest(OUT/'manifest.json'),parent_sha256=PARENT_SHA,
                encoder_sha256=digest(ROOT/'signal_dual_view_residual.py'),
                images=[dict(id=r['id'],path=r['image'],sha256=digest(r['image'])) for r in rows])
    if path.exists():
        cache=torch.load(path,map_location='cpu',weights_only=True)
        if cache['config']!=config:raise RuntimeError('Frozen feature provenance differs')
        return cache
    ff=[];zz=[];qq=[]
    for start in range(0,len(rows),32):
        f,z,q=encode(reader,[read_image(r['image']) for r in rows[start:start+32]])
        ff.append(f);zz.append(z);qq.append(q)
        if start%160==0 or start+32>=len(rows):print('REPAIR_FEATURES',min(start+32,len(rows)),'/',len(rows),flush=True)
    cache=dict(features=torch.cat(ff),logits=torch.cat(zz),quality=torch.cat(qq),config=config)
    atomic_torch_save(path,cache);return cache


def balanced_ce(logits,labels):
    return torch.stack([F.cross_entropy(logits[labels==c],labels[labels==c]) for c in labels.unique()]).mean()


def selection_metrics(rows,t,d,q):
    predictions=decode(t,d,q)
    pooled=summarize(rows,predictions)
    old_ids=[i for i,r in enumerate(rows) if r['role']=='original_validation' and r['type_id']==0]
    old_vehicle=sum(predictions[i]['type_guard']==0 for i in old_ids)
    right=pooled['by_direction'].get('right',dict(correct=0,wrong=0))
    return dict(pooled=pooled,old_vehicle_type_accepted=old_vehicle,right_correct=right['correct'],right_wrong=right['wrong'])


def type_rank(m,baseline,loss):
    p,b=m['pooled'],baseline['pooled']
    violations=int(p['negative_vehicle_accepted']>b['negative_vehicle_accepted'])
    violations+=int(m['right_correct']<baseline['right_correct'] or m['right_wrong']>baseline['right_wrong'])
    violations+=int(m['old_vehicle_type_accepted']<baseline['old_vehicle_type_accepted'])
    return [violations,p['negative_vehicle_accepted'],-p['vehicle_type_accepted'],float(loss)]


def direction_rank(m,baseline,loss):
    p,b=m['pooled'],baseline['pooled']
    violations=int(p['negative_vehicle_accepted']>b['negative_vehicle_accepted'])
    violations+=int(m['right_correct']<baseline['right_correct'] or m['right_wrong']>baseline['right_wrong'])
    violations+=int(p['wrong']>b['wrong'])
    return [violations,p['negative_vehicle_accepted'],p['wrong'],-p['correct'],float(loss)]


def fit(mode,reader,cache,rows,type_checkpoint=None):
    folder=OUT/mode;folder.mkdir(exist_ok=True)
    result_path=folder/'results.json'
    if result_path.exists():
        result=json.loads(result_path.read_text(encoding='utf8'))
        if digest(result['weights'])!=result['weights_sha256']:raise RuntimeError('Completed trial changed')
        return result
    torch.manual_seed(SEED)
    train=torch.tensor([i for i,r in enumerate(rows) if r['role']=='train' and (mode=='types' or r['direction_id']>=0)])
    val=torch.tensor([i for i,r in enumerate(rows) if r['role'] in CALIBRATION])
    val_rows=[rows[i] for i in val.tolist()]
    ytype=torch.tensor([r['type_id'] for r in rows]);ydir=torch.tensor([r['direction_id'] for r in rows])
    with torch.inference_mode():baseline=selection_metrics(val_rows,cache['logits'][val,:6],cache['logits'][val,6:],cache['quality'][val])
    types=make_type_head(reader).eval()
    if type_checkpoint:types.load_state_dict(torch.load(type_checkpoint,map_location='cpu',weights_only=True)['model'])
    model=types if mode=='types' else make_direction_head(reader)
    if mode!='types':types.requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001 if mode=='types' else .0003,weight_decay=.001)
    target=ytype if mode=='types' else ydir
    x=cache['features'][:,:1280] if mode=='types' else cache['features'][:,1280:]
    plan=dict(epochs=EPOCHS,mode=mode,parent_sha256=PARENT_SHA,manifest_sha256=digest(OUT/'manifest.json'),
        feature_sha256=digest(OUT/'features.pt'),type_initialization_sha256=digest(type_checkpoint) if type_checkpoint else None,
        train_indices=train.tolist(),selection_indices=val.tolist(),calibration_roles=sorted(CALIBRATION),
        gates=dict(type_probability=.85,type_margin=.25,direction_probability=.8,direction_margin=.2),
        frozen_backbone=True,color_weights_unchanged=True,direction_input='emitting_shape_only',
        train_updates_per_epoch=8,teacher_retention_weight=0.5 if mode!='types' else 0,
        teacher_retention_targets='only TRAIN round/right examples; teacher soft outputs are regularization, not new labels',
        heldout_or_public_used_for_training=False,physical_lane_association=False)
    save(folder/'plan.json',plan)
    history=[];best=None;first=1
    if (folder/'last.pt').exists():
        state=torch.load(folder/'last.pt',map_location='cpu',weights_only=True)
        if state['plan']!=plan:raise RuntimeError('Resume configuration differs')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        history=state['history'];best=state['best'];first=state['epoch']+1
    started=time.perf_counter()
    for epoch in range(first,EPOCHS+1):
        model.train();losses=[]
        for _ in range(8):
            optimizer.zero_grad(set_to_none=True);logits=model(x[train])
            if mode=='types':
                loss=(balanced_ce(logits[:,:4],target[train])+balanced_ce(logits[:,4:],target[train]))/2
            else:
                loss=balanced_ce(logits,target[train])
                keep=(target[train]==0)|(target[train]==3)
                teacher=cache['logits'][train[keep],6:].softmax(1)
                retention=F.kl_div(logits[keep].log_softmax(1),teacher,reduction='batchmean')
                loss=loss+.5*retention
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite repair loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            t=model(x[val]) if mode=='types' else types(cache['features'][val,:1280])
            d=cache['logits'][val,6:] if mode=='types' else model(x[val])
            m=selection_metrics(val_rows,t,d,cache['quality'][val])
            if mode=='types':score=float((balanced_ce(t[:,:4],ytype[val])+balanced_ce(t[:,4:],ytype[val]))/2)
            else:
                known=ydir[val]>=0;score=float(balanced_ce(d[known],ydir[val][known]))
        rank=type_rank(m,baseline,score) if mode=='types' else direction_rank(m,baseline,score)
        entry=dict(epoch=epoch,loss=float(np.mean(losses)),calibration_loss=score,rank=rank,metrics=m)
        history.append(entry)
        if best is None or rank<best:
            best=rank
            atomic_torch_save(folder/'best.pt',dict(format='type_dual4_repair_v1' if mode=='types' else 'emitting_direction_retention_v1',
                model=model.state_dict(),plan=plan,epoch=epoch,rank=rank,metrics=m,parent=str(PARENT)))
        atomic_torch_save(folder/'last.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),history=history,best=best,epoch=epoch,plan=plan))
        save(folder/'history.json',history);save(OUT/'status.json',dict(state='training',mode=mode,epoch=epoch,epochs=EPOCHS,pid=os.getpid()))
        print('REPAIR_EPOCH',mode,epoch,'/',EPOCHS,'rank',rank,'right',m['right_correct'],'neg',m['pooled']['negative_vehicle_accepted'],flush=True)
    selected=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
    result=dict(state='completed',mode=mode,epochs=EPOCHS,selected_epoch=selected['epoch'],selection_rank=selected['rank'],
        baseline_calibration=baseline,selected_calibration=selected['metrics'],weights=str(folder/'best.pt'),weights_sha256=digest(folder/'best.pt'),
        training_seconds=time.perf_counter()-started,calibration_constraints_passed=selected['rank'][0]==0,adopted=False)
    save(result_path,result);return result


def evaluate_all(reader,cache,rows,type_result,direction_result):
    types=make_type_head(reader).eval();types.load_state_dict(torch.load(type_result['weights'],map_location='cpu',weights_only=True)['model'])
    direction=make_direction_head(reader).eval();direction.load_state_dict(torch.load(direction_result['weights'],map_location='cpu',weights_only=True)['model'])
    with torch.inference_mode():t=types(cache['features'][:,:1280]);d=direction(cache['features'][:,1280:])
    predictions=dict(original=decode(cache['logits'][:,:6],cache['logits'][:,6:],cache['quality']),
        type_only=decode(t,cache['logits'][:,6:],cache['quality']),combined=decode(t,d,cache['quality']))
    # Prior mixed-view candidate is a comparison, never a new initialization.
    prior=ROOT/'runs/domestic_direction_20261022/frozen_classifier/best.pt'
    state=torch.load(prior,map_location='cpu',weights_only=True)['model']
    with torch.inference_mode():mixed=F.linear((cache['features'][:,:1280]+cache['features'][:,1280:])/2,state['2.linear.weight'],state['2.linear.bias'])
    predictions['type_plus_prior_mixed_direction']=decode(t,mixed,cache['quality'])
    metrics={}
    for role in sorted({r['role'] for r in rows}):
        ii=[i for i,r in enumerate(rows) if r['role']==role]
        metrics[role]={k:summarize([rows[i] for i in ii],[p[i] for i in ii]) for k,p in predictions.items()}
    save(OUT/'component_evaluation.json',metrics)
    save(OUT/'prediction_rows.json',[dict(id=r['id'],role=r['role'],type_id=r['type_id'],direction_id=r['direction_id'],
        predictions={k:p[i] for k,p in predictions.items()}) for i,r in enumerate(rows)])
    # All eighteen labelled public cases and unknown probes remain diagnostic.
    audit=json.loads((ROOT/'runs/direction_path_relation_20261021/visual_direction_audit.json').read_text(encoding='utf8'))['records']
    public_rows=[];public_predictions={k:[] for k in ('original','type_only','combined')}
    from signal_attribute_model import DIRECTION_NAMES
    for r in audit:
        if digest(r['image'])!=r['image_sha256']:raise RuntimeError('Known diagnostic source changed')
        f,z,q=encode(reader,[crop_head(read_image(r['image']),r['box'])])
        with torch.inference_mode():pt=types(f[:,:1280]);pd=direction(f[:,1280:])
        for key,tt,dd in [('original',z[:,:6],z[:,6:]),('type_only',pt,z[:,6:]),('combined',pt,pd)]:
            public_predictions[key].append(decode(tt,dd,q)[0])
        public_rows.append(dict(id=r['review_id'],image=r['image'],box=r['box'],type_id=0,
            direction_id=DIRECTION_NAMES.index(r['direction_label']) if r['direction_label'] is not None else -100))
    known=[i for i,r in enumerate(public_rows) if r['direction_id']>=0]
    public={k:summarize([public_rows[i] for i in known],[p[i] for i in known]) for k,p in public_predictions.items()}
    save(OUT/'known_public_diagnostic.json',dict(metrics=public,records=[dict(**r,predictions={k:p[i] for k,p in public_predictions.items()}) for i,r in enumerate(public_rows)],
        scope='Previously reviewed direction-labelled heads; vehicle type used as diagnostic assumption, not new physical lane truth'))
    return metrics,public


def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(4);cv2.setNumThreads(1);torch.manual_seed(SEED)
    if digest(PARENT)!=PARENT_SHA:raise RuntimeError('Original parent changed')
    complete=OUT/'training_results.json'
    if complete.exists():
        result=json.loads(complete.read_text(encoding='utf8'))
        for r in result['trials']:
            if digest(r['weights'])!=r['weights_sha256']:raise RuntimeError('Frozen candidate changed')
        print('TYPE_DIRECTION_REPAIR_ALREADY_COMPLETED',flush=True);return
    from training_memory_monitor import start_monitor,memory_snapshot
    if memory_snapshot()['commit_free_GB']<2:raise RuntimeError('Insufficient commit headroom')
    stop,monitor=start_monitor(OUT/'memory');started=time.perf_counter()
    try:
        manifest=prepare();rows=manifest['records'];reader=ConsensusAttributeClassifier(PARENT)
        save(OUT/'status.json',dict(state='caching_features',pid=os.getpid()))
        cache=feature_cache(reader,manifest)
        with torch.inference_mode():
            linear=reader.model.model[-1].linear
            reconstructed=torch.cat([linear(cache['features'][:,:1280])[:,:6],linear(cache['features'][:,1280:])[:,6:]],1)
            maximum=float((reconstructed-cache['logits']).abs().max())
        if maximum>1e-4:raise RuntimeError('Frozen feature/logit identity failed')
        save(OUT/'feature_verification.json',dict(max_logit_difference=maximum,threshold=1e-4))
        types=fit('types',reader,cache,rows)
        direction=fit('direction',reader,cache,rows,types['weights'])
        save(OUT/'selection_before_diagnostics.json',dict(types=types,direction=direction,
            domestic_review_used_for_selection=False,additional_type_and_device_checks_used_for_selection=False,
            known_public_direction_diagnostic_used_for_selection=False,gates_changed=False))
        metrics,public=evaluate_all(reader,cache,rows,types,direction)
        result=dict(state='completed',epochs=16,trials=[types,direction],metrics=metrics,known_public_diagnostic=public,
            parent_unchanged=digest(PARENT)==PARENT_SHA,elapsed_seconds=time.perf_counter()-started,
            model_adopted=False,physical_lane_association_verified=False,hardware_accessed=False,
            all_evaluation_materials_are_known_regression=True)
        save(complete,result);save(OUT/'status.json',dict(state='completed',model_adopted=False))
        print('TYPE_DIRECTION_REPAIR_TRAINED',dict(types=types['selected_epoch'],direction=direction['selected_epoch'],public=public),flush=True)
    finally:stop.set();monitor.join(timeout=2)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        save(OUT/'status.json',dict(state='failed',error=repr(exc)));raise
