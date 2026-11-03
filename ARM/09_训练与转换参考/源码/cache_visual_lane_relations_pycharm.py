"""Cache frozen features once; augment only training sites and exclude unknowns."""
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import cv2
import torch

from expanded_relation_data import ROOT,digest,save
from light_roi_common import read_image
from visual_lane_relation_features import (FrozenLaneVisualEncoder,augment_frame,ROAD_PARENT,ROAD_SHA256,
                                           HEAD_PARENT,HEAD_SHA256)

OUT=ROOT/'runs/visual_lane_relation_20261021'


def main():
    torch.set_num_threads(4);cv2.setNumThreads(1)
    pointer=ROOT/'runs/signal_lane_relations_expanded_20261021/latest_curated_dataset.json'
    p=json.loads(pointer.read_text(encoding='utf-8'));manifest_path=Path(p['manifest'])
    if digest(manifest_path)!=p['manifest_sha256']:raise RuntimeError('Curated dataset changed')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if digest(ROAD_PARENT)!=ROAD_SHA256 or digest(HEAD_PARENT)!=HEAD_SHA256:
        raise RuntimeError('Frozen feature parent changed')
    config=dict(manifest=str(manifest_path),manifest_sha256=p['manifest_sha256'],
        road_parent=str(ROAD_PARENT),road_sha256=ROAD_SHA256,head_parent=str(HEAD_PARENT),head_sha256=HEAD_SHA256,
        train_variants=['base','dark','bright','mirror'],calibration_and_review_variants=['base'],
        input_size=512,geometry_dimensions=45,head_dimensions=1280,road_dimensions=1536,
        source_sha256={n:digest(ROOT/n) for n in ['cache_visual_lane_relations_pycharm.py','visual_lane_relation_features.py']},
        scope='Frozen grayscale image features and normalized query/box geometry only')
    key=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:12]
    folder=OUT/('features_'+key);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'completed.json').exists():
        done=json.loads((folder/'completed.json').read_text(encoding='utf-8'))
        if done['config']!=config or digest(folder/'features.pt')!=done['feature_sha256']:
            raise RuntimeError('Completed feature cache changed')
        save(OUT/'latest_features.json',done);print('RELATION_FEATURES_ALREADY_CACHED',folder,flush=True);return done
    by_frame=defaultdict(list)
    for index,row in enumerate(manifest['records']):by_frame[row['frame_key']].append((index,row))
    encoder=FrozenLaneVisualEncoder();started=time.perf_counter();shards=[]
    for number,frame in enumerate(manifest['frames']):
        rows=by_frame[frame['frame_key']]
        if digest(frame['image'])!=frame['image_sha256']:raise RuntimeError('Current source image changed')
        variants=config['train_variants'] if frame['split']=='train' else ['base']
        for variant in variants:
            target=folder/'shards'/f'{number:03d}_{variant}.pt';target.parent.mkdir(exist_ok=True)
            if target.exists():shards.append(target);continue
            original=read_image(frame['image'])
            lane_indices=sorted({r['lane_index'] for _,r in rows});head_indices=sorted({r['element_index'] for _,r in rows})
            lane_lookup={i:s for s,i in enumerate(lane_indices)};head_lookup={i:s for s,i in enumerate(head_indices)}
            query=[next(r['query'] for _,r in rows if r['lane_index']==i) for i in lane_indices]
            boxes=[next(r['box'] for _,r in rows if r['element_index']==i) for i in head_indices]
            image,queries,boxes=augment_frame(original,query,boxes,variant)
            encoded=encoder.encode_frame(image,queries,boxes)
            head=[];road=[];geometry=[];metadata=[]
            for record_index,row in rows:
                li,hi=lane_lookup[row['lane_index']],head_lookup[row['element_index']]
                features=encoder.pair(encoded,li,hi,queries[li],boxes[hi],image.shape)
                head.append(features['head']);road.append(features['road']);geometry.append(features['geometry'])
                metadata.append(dict(record_index=record_index,variant=variant,split=row['split'],
                    frame_key=row['frame_key'],site_group=row['site_group'],source_group=row['source_group'],
                    lane_index=row['lane_index'],element_index=row['element_index'],label=row['label'],
                    relation_kind=row['relation_kind'],parallel_positive_lane_indices=row['parallel_positive_lane_indices']))
            torch.save(dict(head=torch.stack(head),road=torch.stack(road),geometry=torch.stack(geometry),
                            metadata=metadata,config_fingerprint=key,frame_image_sha256=frame['image_sha256']),target)
            shards.append(target)
        if number%10==0 or number+1==len(manifest['frames']):
            print('RELATION_FEATURE_FRAME',number+1,'/',len(manifest['frames']),
                  'elapsed',round(time.perf_counter()-started,1),'seconds',flush=True)
    parts=[torch.load(s,map_location='cpu',weights_only=True) for s in shards]
    payload={k:torch.cat([r[k] for r in parts]) for k in ['head','road','geometry']}
    payload['metadata']=[m for r in parts for m in r['metadata']]
    payload['config']=config
    torch.save(payload,folder/'features.pt')
    result=dict(created=datetime.now().isoformat(),config=config,folder=str(folder),features=str(folder/'features.pt'),
                feature_sha256=digest(folder/'features.pt'),pair_feature_rows=len(payload['metadata']),
                original_pairs=len(manifest['records']),original_frames=len(manifest['frames']),
                feature_extraction_seconds=time.perf_counter()-started,unknown_pairs_used_for_training=False,
                model_backbones_updated=False,hardware_accessed=False)
    save(folder/'completed.json',result);save(OUT/'latest_features.json',result)
    print('RELATION_VISUAL_FEATURES_READY',len(payload['metadata']),'rows',round(result['feature_extraction_seconds'],2),'seconds',flush=True)
    return result


if __name__=='__main__':main()
