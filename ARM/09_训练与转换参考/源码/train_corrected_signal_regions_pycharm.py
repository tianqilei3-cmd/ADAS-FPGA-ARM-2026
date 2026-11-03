"""Train on corrected type truth and genuine separated timer/device negatives."""
from collections import Counter
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from light_roi_common import ROOT,read_image,crop_head
from prepare_signal_hard_examples_pycharm import OUT,PRIOR
from prepare_corrected_signal_training_pycharm import main as prepare
from prepare_type_direction_repair_pycharm import digest,save
from signal_type_direction_repair import PARENT,PARENT_SHA,make_type_head,make_direction_head,decode,summarize
from signal_attribute_consensus import ConsensusAttributeClassifier
from signal_dual_view_residual import encode
from signal_corrected_regions import encode_detail
from train_type_direction_repair_pycharm import balanced_ce
from train_visual_lane_relations_pycharm import atomic_torch_save

CALIBRATION={'domestic_calibration','original_validation','calibration','new_calibration'}
TYPE_EPOCHS=16
DIRECTION_EPOCHS=12


def features(reader,rows):
    path=OUT/'corrected_features.pt'
    config=dict(manifest_sha256=digest(OUT/'corrected_manifest.json'),encoder_sha256=PARENT_SHA,
         previous_feature_sha256=digest(PRIOR/'features.pt'),detail_encoder_sha256=digest(ROOT/'signal_corrected_regions.py'),
         images=[dict(id=r['id'],image=r['image'],sha256=digest(r['image'])) for r in rows])
    if path.exists():
        result=torch.load(path,map_location='cpu',weights_only=True)
        assert result['config']==config,'Feature provenance changed'
        return result
    old=torch.load(PRIOR/'features.pt',map_location='cpu',weights_only=True)
    f=torch.empty((len(rows),2560));z=torch.empty((len(rows),10));q=torch.empty((len(rows),1))
    fresh=[]
    for i,r in enumerate(rows):
        if 'cache_index' in r:
            j=r['cache_index']; assert old['config']['images'][j]['id']==r['id']
            assert old['config']['images'][j]['sha256']==r['image_sha256']
            f[i]=old['features'][j];z[i]=old['logits'][j];q[i]=old['quality'][j]
        else:fresh.append(i)
    for start in range(0,len(fresh),32):
        ids=fresh[start:start+32]
        ff,zz,qq=encode(reader,[read_image(rows[i]['image']) for i in ids])
        f[ids]=ff;z[ids]=zz;q[ids]=qq
    detail=[];quality=[]
    for start in range(0,len(rows),48):
        df,dq=encode_detail(reader,[read_image(r['image']) for r in rows[start:start+48]])
        detail.append(df);quality.append(dq)
        if start%240==0 or start+48>=len(rows):print('CORRECTED_DETAIL_FEATURES',min(start+48,len(rows)),'/',len(rows),flush=True)
    result=dict(features=f,logits=z,quality=q,detail_features=torch.cat(detail),detail_quality=torch.cat(quality),config=config)
    atomic_torch_save(path,result)
    return result


def type_loss(logits, labels, rows):
    pieces=[]
    for c in labels.unique():
        mask=labels==c
        if int(c)==0:
            domains=[r['origin'] for r in rows]
        elif int(c)==3:
            domains=[r.get('other_kind','legacy_background') for r in rows]
        else:domains=['single']*len(rows)
        losses=[]
        for name in sorted({domains[i] for i in range(len(rows)) if mask[i]}):
            subset=mask&torch.tensor([v==name for v in domains])
            losses.append(F.cross_entropy(logits[subset],labels[subset]))
        pieces.append(torch.stack(losses).mean())
    return torch.stack(pieces).mean()


def metrics(rows,t,d,q):
    p=decode(t,d,q);m=summarize(rows,p)
    old=sum(v['type_guard']==0 for r,v in zip(rows,p) if r['role']=='original_validation' and r['type_id']==0)
    ped_false=sum(v['type_guard']==0 for r,v in zip(rows,p) if r['type_id']==1)
    right=m['by_direction'].get('right',dict(correct=0,wrong=0))
    return dict(pooled=m,old_vehicle_accepted=old,pedestrian_false_vehicle=ped_false,
         right_correct=right['correct'],right_wrong=right['wrong'])


def rank(m,baseline,score,mode):
    p=m['pooled'];b=baseline['pooled']
    violations=int(m['old_vehicle_accepted']<38)+int(m['pedestrian_false_vehicle']>0)
    violations+=int(p['negative_vehicle_accepted']>0)
    violations+=int(m['right_correct']<(27 if mode=='type' else baseline['right_correct']) or m['right_wrong']>baseline['right_wrong'])
    if mode=='direction':violations+=int(p['wrong']>b['wrong'])
    if mode=='type':return [violations,p['negative_vehicle_accepted'],-p['vehicle_type_accepted'],float(score)]
    return [violations,p['negative_vehicle_accepted'],p['wrong'],-p['correct'],float(score)]


def train(mode,reader,cache,rows,types=None):
    folder=OUT/('type_training' if mode=='type' else 'direction_training');folder.mkdir(exist_ok=True)
    if (folder/'results.json').exists():return json.loads((folder/'results.json').read_text(encoding='utf8'))
    torch.manual_seed(20261023)
    epochs=TYPE_EPOCHS if mode=='type' else DIRECTION_EPOCHS
    train=torch.tensor([i for i,r in enumerate(rows) if r['role']=='train' and (mode=='type' or r['direction_id']>=0)])
    val=torch.tensor([i for i,r in enumerate(rows) if r['role'] in CALIBRATION])
    tr=[rows[i] for i in train.tolist()];vr=[rows[i] for i in val.tolist()]
    yt=torch.tensor([r['type_id'] for r in rows]);yd=torch.tensor([r['direction_id'] for r in rows])
    f,z,q=cache['features'],cache['logits'],cache['quality']
    old_dir=make_direction_head(reader).eval()
    old_dir.load_state_dict(torch.load(PRIOR/'direction/best.pt',map_location='cpu',weights_only=True)['model'])
    with torch.no_grad():old_direction=old_dir(f[:,1280:])
    model=make_type_head(reader) if mode=='type' else make_direction_head(reader)
    if mode=='direction':model.load_state_dict(old_dir.state_dict());types.requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.001)
    x=f[:,:1280] if mode=='type' else cache['detail_features']
    if mode=='type':
        baseline=metrics(vr,z[val,:6],old_direction[val],q[val])
        keep=torch.tensor([r['origin']!='domestic_author' and r['type_id']<3 for r in tr])
        pa,pb=z[train,:3].softmax(1),z[train,3:6].softmax(1)
        keep&=(pa.argmax(1)==yt[train])&(pb.argmax(1)==yt[train])
        keep&=(pa.max(1).values>=.85)&(pb.max(1).values>=.85)
        ta=torch.cat([pa[keep]*(1-1e-6),torch.full((int(keep.sum()),1),1e-6)],1)
        tb=torch.cat([pb[keep]*(1-1e-6),torch.full((int(keep.sum()),1),1e-6)],1)
    else:
        with torch.no_grad():fixed_type=types(f[:,:1280])
        baseline=metrics(vr,fixed_type[val],old_direction[val],q[val])
        teacher=old_direction[train].softmax(1)
        keep=((yd[train]==0)|(yd[train]==3))&(teacher.argmax(1)==yd[train])&(teacher.max(1).values>=.8)
        teacher=teacher[keep]
    plan=dict(epochs=epochs,mode=mode,learning_rate=.0003,updates_per_epoch=8,retention_weight=.5,
         train_indices=train.tolist(),selection_indices=val.tolist(),teacher_indices=train[keep].tolist(),
         calibration_roles=sorted(CALIBRATION),manifest_sha256=digest(OUT/'corrected_manifest.json'),
         feature_sha256=digest(OUT/'corrected_features.pt'),parent_sha256=PARENT_SHA,
         type_checkpoint=None if mode=='type' else digest(OUT/'type_training/best.pt'),
         direction_teacher_checkpoint=digest(PRIOR/'direction/best.pt'),
         thresholds=dict(type_probability=.85,type_margin=.25,direction_probability=.8,direction_margin=.2),
         train_synthetic=False,heldout_targets_in_loss=False,previous_poisoned_type_head_initialization=False,
         whole_type_input=True,direction_input='native_mask_without_close_fill' if mode=='direction' else 'retained_emitter',
         physical_lane_labels=False)
    save(folder/'plan.json',plan)
    first=1;history=[];best=None
    if (folder/'last.pt').exists():
        state=torch.load(folder/'last.pt',map_location='cpu',weights_only=True)
        assert state['plan']==plan
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        first,history,best=state['epoch']+1,state['history'],state['best']
    started=time.perf_counter()
    for epoch in range(first,epochs+1):
        model.train();losses=[]
        for _ in range(8):
            optimizer.zero_grad(set_to_none=True);output=model(x[train])
            if mode=='type':
                loss=(type_loss(output[:,:4],yt[train],tr)+type_loss(output[:,4:],yt[train],tr))/2
                retention=(F.kl_div(output[keep,:4].log_softmax(1),ta,reduction='batchmean')+
                           F.kl_div(output[keep,4:].log_softmax(1),tb,reduction='batchmean'))/2
            else:
                loss=balanced_ce(output,yd[train])
                retention=F.kl_div(output[keep].log_softmax(1),teacher,reduction='batchmean')
            loss=loss+.5*retention;assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            if mode=='type':
                t=model(x[val]);d=old_direction[val];quality=q[val]
                score=float((balanced_ce(t[:,:4],yt[val])+balanced_ce(t[:,4:],yt[val]))/2)
            else:
                t=fixed_type[val];d=model(x[val]);quality=cache['detail_quality'][val]
                known=yd[val]>=0;score=float(balanced_ce(d[known],yd[val][known]))
            m=metrics(vr,t,d,quality)
        score_rank=rank(m,baseline,score,mode)
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)),calibration_loss=score,rank=score_rank,metrics=m))
        if best is None or score_rank<best:
            best=score_rank
            atomic_torch_save(folder/'best.pt',dict(format='corrected_region_type_v1' if mode=='type' else 'native_mask_direction_v1',
                 model=model.state_dict(),epoch=epoch,rank=score_rank,metrics=m,plan=plan))
        atomic_torch_save(folder/'last.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
             history=history,best=best,plan=plan))
        save(folder/'history.json',history)
        save(OUT/'status.json',dict(state='training',mode=mode,epoch=epoch,pid=os.getpid()))
        print('CORRECTED_REGION_EPOCH',mode,epoch,'rank',score_rank,'old',m['old_vehicle_accepted'],
             'right',m['right_correct'],'neg',m['pooled']['negative_vehicle_accepted'],flush=True)
    selected=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
    result=dict(mode=mode,epochs=epochs,selected_epoch=selected['epoch'],rank=selected['rank'],metrics=selected['metrics'],
         baseline=baseline,weights=str(folder/'best.pt'),sha256=digest(folder/'best.pt'),constraints_passed=selected['rank'][0]==0,
         training_seconds=time.perf_counter()-started)
    save(folder/'results.json',result);return result


def evaluate(reader,cache,rows,types,direction):
    f,z,q=cache['features'],cache['logits'],cache['quality']
    previous_type=make_type_head(reader).eval()
    previous_type.load_state_dict(torch.load(PRIOR/'types/best.pt',map_location='cpu',weights_only=True)['model'])
    previous_direction=make_direction_head(reader).eval()
    previous_direction.load_state_dict(torch.load(PRIOR/'direction/best.pt',map_location='cpu',weights_only=True)['model'])
    def variants(ff,zz,qq,df,dq):
        with torch.inference_mode():
            newt=types(ff[:,:1280]);oldd=previous_direction(ff[:,1280:])
            return dict(original=decode(zz[:,:6],zz[:,6:],qq),
                 previous_poisoned_type=decode(previous_type(ff[:,:1280]),oldd,qq),
                 corrected_types=decode(newt,oldd,qq),detail_candidate=decode(newt,direction(df),dq))
    predictions=variants(f,z,q,cache['detail_features'],cache['detail_quality'])
    result={}
    for role in sorted({r['role'] for r in rows}):
        ids=[i for i,r in enumerate(rows) if r['role']==role]
        result[role]={k:summarize([rows[i] for i in ids],[p[i] for i in ids]) for k,p in predictions.items()}
    save(OUT/'corrected_evaluation.json',result)
    save(OUT/'corrected_prediction_rows.json',[dict(id=r['id'],role=r['role'],type_id=r['type_id'],direction_id=r['direction_id'],
         predictions={k:p[i] for k,p in predictions.items()}) for i,r in enumerate(rows)])
    # Native outside-source diagnostics run only after both selections are frozen.
    public_rows=json.loads((PRIOR/'known_public_diagnostic.json').read_text(encoding='utf8'))['records']
    images=[crop_head(read_image(r['image']),r['box']) for r in public_rows]
    pf,pz,pq=encode(reader,images);df,dq=encode_detail(reader,images)
    public=variants(pf,pz,pq,df,dq);known=[i for i,r in enumerate(public_rows) if r['direction_id']>=0]
    public_metrics={k:summarize([public_rows[i] for i in known],[p[i] for i in known]) for k,p in public.items()}
    save(OUT/'known_direction_diagnostic.json',dict(metrics=public_metrics,records=[dict(id=r['id'],image=r['image'],box=r['box'],
         type_id=0,direction_id=r['direction_id'],predictions={k:p[i] for k,p in public.items()}) for i,r in enumerate(public_rows)],
         known_development_cases=True,motor_type_is_diagnostic_assumption=True,physical_lane_truth=False))
    return result,public_metrics


def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(4);cv2.setNumThreads(1)
    target=OUT/'corrected_training_results.json'
    if target.exists():
        print('CORRECTED_SIGNAL_TRAINING_ALREADY_COMPLETED',flush=True);return
    from training_memory_monitor import start_monitor,memory_snapshot
    if memory_snapshot()['commit_free_GB']<2:raise RuntimeError('Insufficient commit memory')
    assert digest(PARENT)==PARENT_SHA
    stop,monitor=start_monitor(OUT/'memory');started=time.perf_counter()
    try:
        manifest=prepare();rows=manifest['records'];reader=ConsensusAttributeClassifier(PARENT)
        cache=features(reader,rows)
        type_result=train('type',reader,cache,rows)
        types=make_type_head(reader).eval();types.load_state_dict(torch.load(type_result['weights'],map_location='cpu',weights_only=True)['model'])
        direction_result=train('direction',reader,cache,rows,types)
        direction=make_direction_head(reader).eval();direction.load_state_dict(torch.load(direction_result['weights'],map_location='cpu',weights_only=True)['model'])
        save(OUT/'selection_before_diagnostics.json',dict(type=type_result,direction=direction_result,
             new_review_or_public_used_for_selection=False,thresholds_changed=False,old_label_corrections_before_training=True))
        result,public=evaluate(reader,cache,rows,types,direction)
        save(target,dict(state='completed',epochs=28,trials=[type_result,direction_result],metrics=result,known_direction_diagnostic=public,
             corrected_ground_truth_used_for_all_comparisons=True,prior_digit_metrics_superseded=True,
             elapsed_seconds=time.perf_counter()-started,model_adopted=False,hardware_accessed=False,physical_lane_labels=False))
        save(OUT/'status.json',dict(state='completed',epochs=28,model_adopted=False))
        print('CORRECTED_SIGNAL_TRAINING_COMPLETED',public,flush=True)
    finally:stop.set();monitor.join(timeout=2)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        save(OUT/'status.json',dict(state='failed',error=repr(exc)));raise
