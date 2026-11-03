"""Train matched geometry/visual controls on the fixed 18-site curated split.

CPU 4 threads. Backbones and feature cache stay frozen. Only train/calibration
sites select checkpoints and thresholds. Completed runs are reused; the last
complete epoch stores optimizer/RNG state. No scene default or board changes.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from cache_visual_lane_relations_pycharm import OUT,main as cache_main
from expanded_relation_data import ROOT,digest,save
from light_roi_common import read_image
from visual_lane_relation_evaluation import (balanced_site_weights,ranking_indices,
    relation_metrics,select_positive_threshold)
from visual_lane_relation_model import (FORMAT,SEED,VisualLaneRelationModel,make_head,
    fit_training_transform,transformed_features)


def atomic_torch_save(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp');torch.save(value,temp);temp.replace(path)


def objective(logits,labels,weights,indices,ranking):
    bce=(F.binary_cross_entropy_with_logits(logits[indices],labels[indices],reduction='none')*weights[indices]).mean()
    rank=F.softplus(logits[ranking[:,1]]-logits[ranking[:,0]]+.5).mean()
    return bce+.25*rank


def train_mode(cache,feature_record,manifest,mode,common):
    config=dict(**common,mode=mode)
    fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:12]
    run=ROOT/'runs/adas'/f'lane_relation_{mode}_{fingerprint}';run.mkdir(parents=True,exist_ok=True)
    output=run/'results.json'
    if output.exists():
        result=json.loads(output.read_text(encoding='utf-8'))
        if result['config']!=config or digest(result['weights'])!=result['weights_sha256']:
            raise RuntimeError('Completed relation experiment changed')
        print('RELATION_MODE_ALREADY_COMPLETED',mode,run,flush=True);return result
    metadata=cache['metadata']
    train=[i for i,r in enumerate(metadata) if r['split']=='train']
    calibration=[i for i,r in enumerate(metadata) if r['split']=='calibration']
    if any(metadata[i]['variant']!='base' for i,r in enumerate(metadata) if r['split']!='train'):
        raise RuntimeError('Held-out sites must not contain augmented evaluation copies')
    labels=torch.tensor([r['label'] for r in metadata],dtype=torch.float32)
    weights=balanced_site_weights(metadata,train);cal_weights=balanced_site_weights(metadata,calibration)
    train_rank=ranking_indices(metadata,train);cal_rank=ranking_indices(metadata,calibration)
    train_idx=torch.tensor(train,dtype=torch.int64);cal_idx=torch.tensor(calibration,dtype=torch.int64)
    history=[];first=1;best_score=float('inf');best_epoch=None
    last_path=run/'last.pt';last=None
    if last_path.exists():
        last=torch.load(last_path,map_location='cpu',weights_only=True)
        if last['config']!=config:raise RuntimeError('Resume configuration changed')
        transform=last['transform'];history=last['history'];first=last['epoch']+1
        best_score=last['best_score'];best_epoch=last['best_epoch']
    else:transform=fit_training_transform(cache,mode)
    x=transformed_features(cache,transform)
    if not torch.isfinite(x).all():raise RuntimeError('Invalid transformed features')
    torch.manual_seed(SEED)
    model=make_head(x.shape[1]);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.02)
    if last:
        model.load_state_dict(last['model']);optimizer.load_state_dict(last['optimizer']);torch.set_rng_state(last['rng_state'])
    save(run/'plan.json',config)
    started=time.perf_counter();previous=history[-1]['training_seconds'] if history else 0.
    for epoch in range(first,common['epochs']+1):
        model.train();loss_value=0.
        for _ in range(common['full_batch_updates_per_epoch']):
            optimizer.zero_grad(set_to_none=True)
            logits=model(x[train_idx]).flatten()
            # Ranking indices refer to full-cache rows; map them to train rows.
            positions=torch.full((len(metadata),),-1,dtype=torch.int64);positions[train_idx]=torch.arange(len(train_idx))
            local_rank=positions[train_rank]
            if (local_rank<0).any():raise RuntimeError('Training ranking includes a held-out row')
            bce=(F.binary_cross_entropy_with_logits(logits,labels[train_idx],reduction='none')*weights[train_idx]).mean()
            rank=F.softplus(logits[local_rank[:,1]]-logits[local_rank[:,0]]+.5).mean()
            loss=bce+.25*rank
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite relation training loss')
            loss.backward();optimizer.step();loss_value+=float(loss.detach())
        model.eval()
        with torch.inference_mode():
            cal_logits=model(x[cal_idx]).flatten()
            positions=torch.full((len(metadata),),-1,dtype=torch.int64);positions[cal_idx]=torch.arange(len(cal_idx))
            local_rank=positions[cal_rank]
            if (local_rank<0).any():raise RuntimeError('Calibration ranking includes a different split')
            cal_bce=(F.binary_cross_entropy_with_logits(cal_logits,labels[cal_idx],reduction='none')*cal_weights[cal_idx]).mean()
            cal_ranking=F.softplus(cal_logits[local_rank[:,1]]-cal_logits[local_rank[:,0]]+.5).mean()
            score=float(cal_bce+.25*cal_ranking)
        if score<best_score:
            best_score=score;best_epoch=epoch
            atomic_torch_save(run/'selected.pt',dict(model=model.state_dict(),epoch=epoch,score=score))
        history.append(dict(epoch=epoch,train_objective=loss_value/common['full_batch_updates_per_epoch'],
            calibration_objective=score,calibration_balanced_bce=float(cal_bce),calibration_ranking_loss=float(cal_ranking),
            training_seconds=previous+time.perf_counter()-started))
        atomic_torch_save(last_path,dict(config=config,transform=transform,model=model.state_dict(),optimizer=optimizer.state_dict(),
            rng_state=torch.get_rng_state(),epoch=epoch,history=history,best_score=best_score,best_epoch=best_epoch))
        save(run/'history.json',history);save(run/'status.json',dict(state='running',epoch=epoch,epochs=common['epochs']))
        if epoch==1 or epoch%10==0:
            print('LANE_RELATION_EPOCH',mode,epoch,'/',common['epochs'],'loss',round(history[-1]['train_objective'],4),
                  'cal',round(score,4),flush=True)
    selected=torch.load(run/'selected.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(selected['model']);model.eval()
    # Freeze the checkpoint and positive gate before reading review predictions.
    with torch.inference_mode():cal_probability=model(x[cal_idx]).flatten().sigmoid().numpy()
    cal_records=[metadata[i] for i in calibration]
    gate=select_positive_threshold(cal_records,cal_probability)
    checkpoint=dict(format=FORMAT,mode=mode,model=model.state_dict(),transform=transform,config=config,
                    epoch=selected['epoch'],threshold=gate['threshold'],calibration_gate=gate,
                    physical_lane_association_verified=False,motion_authorized=False)
    weights_path=run/'weights.pt';atomic_torch_save(weights_path,checkpoint)
    with torch.inference_mode():probability=model(x).flatten().sigmoid().numpy()
    metrics={};predictions=[]
    for split in ['train','calibration','review']:
        ii=[i for i,r in enumerate(metadata) if r['split']==split and r['variant']=='base']
        rr=[metadata[i] for i in ii]
        metrics[split]=dict(calibrated=relation_metrics(rr,probability[ii],gate['threshold']),
                            diagnostic_05=relation_metrics(rr,probability[ii],.5))
        predictions.extend(dict(**r,probability=float(probability[i]),
                                candidate_positive=bool(gate['threshold'] is not None and probability[i]>=gate['threshold']))
                           for i,r in zip(ii,rr))
    save(run/'predictions.json',predictions)
    # Saved runtime re-decodes all review images and extracts features afresh.
    runtime=VisualLaneRelationModel(weights_path)
    cache_lookup={(r['frame_key'],r['lane_index'],r['element_index']):i
                  for i,r in enumerate(metadata) if r['split']=='review'}
    records_by_frame={}
    for row in manifest['records']:
        if row['split']=='review':records_by_frame.setdefault(row['frame_key'],[]).append(row)
    runtime_rows=[]
    for frame in manifest['frames']:
        if frame['split']!='review':continue
        rows=records_by_frame[frame['frame_key']]
        lane_ids=sorted({r['lane_index'] for r in rows});head_ids=sorted({r['element_index'] for r in rows})
        queries=[next(r['query'] for r in rows if r['lane_index']==li) for li in lane_ids]
        boxes=[next(r['box'] for r in rows if r['element_index']==hi) for hi in head_ids]
        image=read_image(frame['image']);fresh=runtime.predict_frame(image,queries,boxes)
        errors=[];paired=0
        for prediction in fresh:
            key=frame['frame_key'],lane_ids[prediction['lane_slot']],head_ids[prediction['head_slot']]
            if key not in cache_lookup:continue
            index=cache_lookup[key];errors.append(abs(prediction['lane_light_probability']-float(probability[index])));paired+=1
        runtime_rows.append(dict(frame_key=frame['frame_key'],labelled_pairs=paired,
                                 max_probability_error=max(errors,default=0)))
    max_error=max(r['max_probability_error'] for r in runtime_rows)
    if max_error>1e-4:raise RuntimeError('Saved fresh-image runtime differs from evaluated model: '+str(max_error))
    save(run/'runtime_review.json',dict(frames=runtime_rows,max_probability_error=max_error))
    review=metrics['review']['calibrated']
    passed=bool(gate['eligible'] and review['precision'] is not None and review['precision']>=.95
                and review['fp']<=3 and review['tp']>=8 and review['strict_false_accepts']==0 and review['positive_covered_sites']>=2)
    result=dict(created=datetime.now().isoformat(),completed=True,mode=mode,config=config,run=str(run),
        weights=str(weights_path),weights_sha256=digest(weights_path),selected_epoch=selected['epoch'],
        selected_calibration_objective=selected['score'],calibration_gate=gate,metrics=metrics,
        parameter_count=sum(p.numel() for p in model.parameters()),training_seconds=history[-1]['training_seconds'],
        runtime_review_frames=len(runtime_rows),runtime_review_pairs=sum(r['labelled_pairs'] for r in runtime_rows),
        max_saved_runtime_probability_error=max_error,public_oracle_review_passed=passed,
        adopted_for_scene_runtime=False,physical_lane_association_verified=False,hardware_accessed=False,
        review_used_for_epoch_or_threshold_selection=False,unknown_relation_pairs_used_as_negatives=False)
    save(output,result);save(run/'status.json',dict(state='completed',epoch=common['epochs']))
    print('LANE_RELATION_MODE_COMPLETED',mode,'gate',gate['threshold'],'review',json.dumps(review),'run',run,flush=True)
    return result


def main():
    torch.set_num_threads(4);cv2.setNumThreads(1)
    summary_path=OUT/'stage_results.json'
    if summary_path.exists():
        result=json.loads(summary_path.read_text(encoding='utf-8'))
        for name,expected in result['common']['source_sha256'].items():
            if digest(ROOT/name)!=expected:raise RuntimeError('Completed experiment source changed')
        for mode in ['geometry','visual']:
            if digest(result[mode]['weights'])!=result[mode]['weights_sha256']:raise RuntimeError('Completed weights changed')
        print('VISUAL_LANE_RELATION_STAGE_ALREADY_COMPLETED',summary_path,flush=True);return result
    feature_record=cache_main()
    cache=torch.load(feature_record['features'],map_location='cpu',weights_only=True)
    manifest_path=Path(feature_record['config']['manifest'])
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    common=dict(manifest=str(manifest_path),manifest_sha256=digest(manifest_path),
        feature_cache=feature_record['features'],feature_sha256=feature_record['feature_sha256'],
        road_parent=feature_record['config']['road_parent'],road_sha256=feature_record['config']['road_sha256'],
        head_parent=feature_record['config']['head_parent'],head_sha256=feature_record['config']['head_sha256'],
        epochs=80,full_batch_updates_per_epoch=4,learning_rate=.001,weight_decay=.02,dropout=.2,
        seed=SEED,ranking_weight=.25,ranking_margin=.5,head_projection_dimensions=16,road_projection_dimensions=32,
        checkpoint_selection='calibration site/class balanced BCE + .25 strict-pair ranking loss',
        threshold_policy='Calibration TP>=8, FP<=3, precision>=.95, strict FP=0, covered sites>=2; otherwise no positive gate',
        source_sha256={name:digest(ROOT/name) for name in ['train_visual_lane_relations_pycharm.py',
            'visual_lane_relation_features.py','visual_lane_relation_model.py','visual_lane_relation_evaluation.py',
            'cache_visual_lane_relations_pycharm.py']})
    save(OUT/'training_plan.json',common)
    geometry=train_mode(cache,feature_record,manifest,'geometry',common)
    visual=train_mode(cache,feature_record,manifest,'visual',common)
    eligible=[r for r in [geometry,visual] if r['calibration_gate']['eligible']]
    proposed=max(eligible,key=lambda r:(r['calibration_gate']['chosen']['tp'],
        -r['calibration_gate']['chosen']['fp'],r['calibration_gate']['chosen']['strict_joint_correct']))['mode'] if eligible else None
    summary=dict(created=datetime.now().isoformat(),common=common,geometry=geometry,visual=visual,
        proposed_by_calibration_only=proposed,source_sites=18,paired_native_frames=130,
        curated_relations=2728,unknown_relations_excluded=312,
        feature_extraction_seconds=feature_record['feature_extraction_seconds'],
        default_models_changed=False,physical_lane_association_verified=False,hardware_accessed=False)
    save(summary_path,summary)
    print('VISUAL_LANE_RELATION_STAGE_COMPLETED',str(summary_path),'calibration proposal',proposed,flush=True)
    return summary


if __name__=='__main__':main()
