"""Compare existing checkpoints under a conservative, signal-control guard.

Unlit-device type errors remain visible in reported metrics. They are not
equated with a confirmed direction. This is experimental, with no motion trust.
"""
import json
import torch

from light_roi_common import read_image,crop_head
from prepare_signal_hard_examples_pycharm import OUT,PRIOR
from prepare_type_direction_repair_pycharm import digest,save
from signal_type_direction_repair import PARENT,make_type_head,make_direction_head,decode,summarize
from signal_attribute_consensus import ConsensusAttributeClassifier
from signal_dual_view_residual import encode
from signal_corrected_regions import encode_detail
from train_corrected_signal_regions_pycharm import CALIBRATION,metrics
from train_visual_lane_relations_pycharm import atomic_torch_save

FOLDER=OUT/'control_guard'


def guarded_decode(types,original,direction,quality):
    predictions=decode(types,direction,quality)
    old=decode(original,direction,quality)
    for p,o in zip(predictions,old):
        pa,pb=p['type_head_probabilities']
        agree_motor=int(torch.tensor(pa).argmax())==0 and int(torch.tensor(pb).argmax())==0
        p.update(original_type_retained=False,promotion_without_emission_blocked=False,
                 original_type_head_probabilities=o['type_head_probabilities'])
        # Old motor evidence cannot overrule a newly learned nonmotor argmax.
        if o['type_guard']==0 and p['type_guard']<0 and agree_motor:
            p.update(type_guard=0,lamp_type='vehicle_signal',original_type_retained=True,
                 direction_guard=o['direction_guard'],shape=o['shape'],control_direction=o['control_direction'])
        if o['type_guard']!=0 and p['type_guard']==0 and not p['emitting_available']:
            p.update(type_guard=-1,lamp_type='unknown',promotion_without_emission_blocked=True,
                 direction_guard=-1,shape='unknown',control_direction='unknown')
    return predictions


def guard_metrics(rows,predictions):
    m=summarize(rows,predictions)
    right=m['by_direction'].get('right',dict(correct=0,wrong=0))
    return dict(pooled=m,old_vehicle_accepted=sum(p['type_guard']==0 for r,p in zip(rows,predictions)
         if r['role']=='original_validation' and r['type_id']==0),right_correct=right['correct'],right_wrong=right['wrong'])


def guard_rank(m):
    p=m['pooled']
    violations=int(m['old_vehicle_accepted']<38)+int(m['right_correct']<28)+int(m['right_wrong']>0)
    violations+=int(p['negative_direction_accepted']>0)
    return [violations,p['negative_direction_accepted'],max(0,38-m['old_vehicle_accepted']),
         max(0,28-m['right_correct']),p['wrong'],p['negative_vehicle_accepted'],-p['correct'],-p['vehicle_type_accepted']]


class CorrectedGuardClassifier:
    def __init__(self):
        self.reader=ConsensusAttributeClassifier(PARENT)
        state=torch.load(FOLDER/'selected_types.pt',map_location='cpu',weights_only=True)
        self.types=make_type_head(self.reader).eval();self.types.load_state_dict(state['model'])
        plan=json.loads((FOLDER/'selection_before_diagnostics.json').read_text(encoding='utf8'))['selected']
        self.mode=plan['direction_mode']
        self.direction=make_direction_head(self.reader).eval()
        self.direction.load_state_dict(torch.load(plan['direction_checkpoint'],map_location='cpu',weights_only=True)['model'])

    def predict(self,images):
        if not images:return []
        f,z,q=encode(self.reader,images)
        if self.mode=='detail':df,q=encode_detail(self.reader,images)
        else:df=f[:,1280:]
        with torch.inference_mode():return guarded_decode(self.types(f[:,:1280]),z[:,:6],self.direction(df),q)


def main():
    torch.set_num_threads(4);FOLDER.mkdir(exist_ok=True)
    if (FOLDER/'evaluation.json').exists():print('CORRECTED_GUARD_ALREADY_COMPLETED',flush=True);return
    rows=json.loads((OUT/'corrected_manifest.json').read_text(encoding='utf8'))['records']
    cache=torch.load(OUT/'corrected_features.pt',map_location='cpu',weights_only=True)
    f,z,q=cache['features'],cache['logits'],cache['quality']
    val=torch.tensor([i for i,r in enumerate(rows) if r['role'] in CALIBRATION]);vr=[rows[i] for i in val.tolist()]
    type_paths=[OUT/'type_training/best.pt',OUT/'type_training/last.pt']
    direction_paths=[('retained_emitter',PRIOR/'direction/best.pt'),('detail',OUT/'direction_training/best.pt')]
    save(FOLDER/'policy_plan_before_diagnostics.json',dict(
         rule='Retain original accepted motor only if both adapted argmaxes remain motor; newly promoted motor requires current emission',
         type_checkpoints=[dict(path=str(p),sha256=digest(p)) for p in type_paths],
         direction_checkpoints=[dict(mode=m,path=str(p),sha256=digest(p)) for m,p in direction_paths],
         candidates=4,selection_indices=val.tolist(),selection_roles=sorted(CALIBRATION),
         control_relevant_negative_direction_errors_prioritized=True,
         unlit_false_motor_types_still_reported_and_block_global_adoption=True,
         new_review_or_public_used_for_ranking=False,all_materials_development=True))
    reader=ConsensusAttributeClassifier(PARENT);choices=[]
    for path in type_paths:
        types=make_type_head(reader).eval();types.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model'])
        with torch.inference_mode():t=types(f[:,:1280])
        for mode,dpath in direction_paths:
            direction=make_direction_head(reader).eval();direction.load_state_dict(torch.load(dpath,map_location='cpu',weights_only=True)['model'])
            with torch.inference_mode():
                d=direction(cache['detail_features'] if mode=='detail' else f[:,1280:])
            quality=cache['detail_quality'] if mode=='detail' else q
            m=guard_metrics(vr,guarded_decode(t[val],z[val,:6],d[val],quality[val]))
            choices.append(dict(type_checkpoint=str(path),type_sha256=digest(path),direction_checkpoint=str(dpath),
                 direction_sha256=digest(dpath),direction_mode=mode,rank=guard_rank(m),metrics=m))
    selected=min(choices,key=lambda r:r['rank'])
    atomic_torch_save(FOLDER/'selected_types.pt',torch.load(selected['type_checkpoint'],map_location='cpu',weights_only=True))
    save(FOLDER/'selection_before_diagnostics.json',dict(choices=choices,selected=selected,control_calibration_passed=selected['rank'][0]==0))
    types=make_type_head(reader).eval();types.load_state_dict(torch.load(selected['type_checkpoint'],map_location='cpu',weights_only=True)['model'])
    direction=make_direction_head(reader).eval();direction.load_state_dict(torch.load(selected['direction_checkpoint'],map_location='cpu',weights_only=True)['model'])
    def predict(ff,zz,qq,df,dq):
        with torch.inference_mode():return guarded_decode(types(ff[:,:1280]),zz[:,:6],
             direction(df if selected['direction_mode']=='detail' else ff[:,1280:]),dq if selected['direction_mode']=='detail' else qq)
    predictions=predict(f,z,q,cache['detail_features'],cache['detail_quality']);result={}
    for role in sorted({r['role'] for r in rows}):
        ids=[i for i,r in enumerate(rows) if r['role']==role]
        result[role]=summarize([rows[i] for i in ids],[predictions[i] for i in ids])
    save(FOLDER/'prediction_rows.json',[dict(id=r['id'],prediction=p) for r,p in zip(rows,predictions)])
    public_rows=json.loads((PRIOR/'known_public_diagnostic.json').read_text(encoding='utf8'))['records']
    images=[crop_head(read_image(r['image']),r['box']) for r in public_rows]
    pf,pz,pq=encode(reader,images);df,dq=encode_detail(reader,images);pp=predict(pf,pz,pq,df,dq)
    known=[i for i,r in enumerate(public_rows) if r['direction_id']>=0]
    public=summarize([public_rows[i] for i in known],[pp[i] for i in known])
    save(FOLDER/'public_prediction_rows.json',[dict(id=r['id'],image=r['image'],box=r['box'],direction_id=r['direction_id'],prediction=p) for r,p in zip(public_rows,pp)])
    save(FOLDER/'evaluation.json',dict(selected=selected,metrics=result,known_direction_diagnostic=public,
         model_adopted=False,hardware_accessed=False,all_results_are_development_regression=True))
    print('CORRECTED_GUARD_COMPLETED',dict(selected=selected['type_checkpoint'],direction=selected['direction_mode'],
         rank=selected['rank'],public=public),flush=True)


if __name__=='__main__':main()
