"""Pair/site/strict-contrast metrics and a calibration-only positive gate."""
from collections import defaultdict

import numpy as np
import torch


def balanced_site_weights(metadata,indices):
    sites=defaultdict(list)
    for i in indices:sites[metadata[i]['site_group']].append(i)
    weights=torch.zeros(len(metadata),dtype=torch.float32)
    for members in sites.values():weights[members]=len(indices)/(len(sites)*len(members))
    labels=torch.tensor([r['label'] for r in metadata],dtype=torch.long)
    totals=torch.stack([weights[labels==k].sum() for k in (0,1)])
    if not (totals>0).all():raise ValueError('Both relation classes are required')
    weights*=weights.sum()/(2*totals[labels])
    return weights/weights[indices].mean()


def ranking_indices(metadata,indices):
    allowed=set(indices);lookup={}
    for i in indices:
        r=metadata[i];lookup[(r['frame_key'],r['variant'],r['lane_index'],r['element_index'])]=i
    anchors=[]
    for i in indices:
        r=metadata[i]
        if r['relation_kind']!='parallel_wrong_lane_contrast':continue
        for positive in r['parallel_positive_lane_indices']:
            key=r['frame_key'],r['variant'],positive,r['element_index']
            if key in lookup:
                j=lookup[key]
                if not metadata[j]['label'] or metadata[i]['label']:raise ValueError('Invalid strict contrast labels')
                anchors.append((j,i))
    if not anchors:raise ValueError('Partition lacks strict lane contrasts')
    return torch.tensor(anchors,dtype=torch.int64)


def relation_metrics(records,probabilities,threshold):
    p=np.asarray(probabilities,dtype=np.float64);y=np.asarray([r['label'] for r in records],bool)
    if p.shape!=(len(records),) or not np.isfinite(p).all():raise ValueError('Finite relation scores required')
    accepted=p>=threshold if threshold is not None else np.zeros(len(p),bool)
    tp=int((accepted&y).sum());fp=int((accepted&~y).sum());fn=int((~accepted&y).sum());tn=int((~accepted&~y).sum())
    sites=defaultdict(list);lookup={};strict_cases=defaultdict(set)
    for i,r in enumerate(records):
        sites[r['site_group']].append(i)
        lookup[(r['frame_key'],r['lane_index'],r['element_index'])]=i
        if r['relation_kind']=='parallel_wrong_lane_contrast':
            strict_cases[(r['frame_key'],r['element_index'])].add(i)
            for lane in r['parallel_positive_lane_indices']:
                key=r['frame_key'],lane,r['element_index']
                # Positives are added after all records have been indexed.
    orders=[];strict_neg=[];joint=0
    for key,negative_indices in strict_cases.items():
        members=set(negative_indices)
        for i in negative_indices:
            strict_neg.append(i)
            for lane in records[i]['parallel_positive_lane_indices']:
                j=lookup[(key[0],lane,key[1])]
                members.add(j);orders.append(p[j]>p[i])
        joint+=all(accepted[i]==y[i] for i in members)
    per_site={}
    for site,ii in sites.items():
        per_site[site]=dict(tp=int((accepted[ii]&y[ii]).sum()),fp=int((accepted[ii]&~y[ii]).sum()),
                           fn=int((~accepted[ii]&y[ii]).sum()),tn=int((~accepted[ii]&~y[ii]).sum()))
    return dict(threshold=threshold,tp=tp,fp=fp,fn=fn,tn=tn,
        precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
        positive_covered_sites=sum(v['tp']>0 for v in per_site.values()),site_count=len(sites),
        strict_negative_count=len(strict_neg),strict_false_accepts=int(accepted[strict_neg].sum()),
        strict_order_pairs=len(orders),strict_order_correct=int(sum(orders)),
        strict_order_accuracy=float(np.mean(orders)) if orders else None,
        strict_head_contrast_cases=len(strict_cases),strict_joint_correct=int(joint),per_site=per_site,
        scope='Labelled oracle-query edges, correlated within frames/journeys; not full-scene detection accuracy')


def select_positive_threshold(records,probabilities):
    trials=[]
    for threshold in [.50,.60,.70,.80,.90,.95,.99]:
        value=relation_metrics(records,probabilities,threshold)
        value['eligible']=(value['tp']>=8 and value['fp']<=3 and value['precision'] is not None
            and value['precision']>=.95 and value['strict_false_accepts']==0 and value['positive_covered_sites']>=2)
        trials.append(value)
    feasible=[v for v in trials if v['eligible']]
    chosen=max(feasible,key=lambda v:(v['tp'],-v['fp'],v['strict_joint_correct'],-v['threshold'])) if feasible else None
    return dict(threshold=chosen['threshold'] if chosen else None,eligible=chosen is not None,
                chosen=chosen,trials=trials,
                policy='Calibration only: TP>=8, FP<=3, precision>=.95, no strict wrong-lane accepts, >=2 positive-covered sites')
