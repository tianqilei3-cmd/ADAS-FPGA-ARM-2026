"""Export the exact retained spatial type + direction-residual path; no training."""
import ast,hashlib,json,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'runs/orangepi/export_dependencies'))
import cv2,numpy as np,torch,onnx
from signal_direction_residual_tail import DirectionResidualClassifier
from signal_spatial_tail import encode_spatial,native_descriptors
from light_roi_common import read_image,crop_head,write_image

OUT=ROOT/'runs/arm_stage_lamp_20261024'
PACKAGE=ROOT/'arm_lamp_attrs_20261024'
DECISIONS=['lamp_type','type_guard','direction_guard','shape','control_direction','emitting_available',
           'original_type_retained','promotion_without_emission_blocked']


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


class ExportGraph(torch.nn.Module):
    def __init__(self,reader):
        super().__init__()
        modules=reader.reader.model.model
        assert all(m.f==-1 for m in modules),'Review non-sequential backbone before export'
        self.encoder=modules[:-1];self.original=modules[-1]
        self.types=reader.types;self.direction=reader.direction;self.residual=reader.residual

    def forward(self,whole,emitter,descriptors):
        n=whole.shape[0]
        spatial=self.encoder(torch.cat([whole,emitter],0))
        pooled=self.original.pool(self.original.conv(spatial)).flatten(1)
        original_logits=self.original.linear(pooled)
        types=self.types(spatial[:n])
        rp=self.residual.pool(self.residual.conv(spatial)).flatten(1)
        features=torch.cat([rp[:n],rp[n:]],1)
        native=(descriptors-self.residual.descriptor_mean)/self.residual.descriptor_std
        fused=torch.cat([self.residual.norm(features),native],1)
        delta=self.residual.out(torch.nn.functional.silu(self.residual.hidden(fused)))
        directions=self.direction(pooled[n:])+delta
        return types,original_logits[:n,:6],directions,delta


def main():
    OUT.mkdir(exist_ok=True);(PACKAGE/'models').mkdir(exist_ok=True)
    assert not (OUT/'export_verification.json').exists(),'Keep prior verified export'
    torch.set_num_threads(4);cv2.setNumThreads(1)
    # Exact existing preprocessing functions, without importing PyTorch on ARM.
    locations={'light_roi_common.py':['input_rgb','color_evidence'],
               'signal_attribute_model.py':['preprocess','direction_input'],
               'signal_spatial_tail.py':['native_descriptors']}
    functions=[];sources={}
    for filename,names in locations.items():
        path=ROOT/filename;tree=ast.parse(path.read_text(encoding='utf-8'));sources[filename]=digest(path)
        for name in names:
            node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
            functions.append(ast.unparse(node))
    (PACKAGE/'lamp_attrs_preprocess.py').write_text('"""Exact exported preprocessing; see metadata source hashes."""\nimport cv2\nimport numpy as np\n\n'+'\n\n'.join(functions)+'\n',encoding='utf-8')
    sys.path.insert(0,str(PACKAGE))
    from lamp_attrs_portable import inputs,PortableAttributes,guarded_decode
    frozen=[ROOT/'runs/adas/signal_attributes_consensus_20fc4d9bebcd/weights/best.pt',
        ROOT/'runs/signal_spatial_tail_20261022/types/best.pt',
        ROOT/'runs/signal_spatial_tail_20261022/direction_residual/best.pt',
        ROOT/'runs/signal_type_direction_repair_20261022/direction/best.pt',
        ROOT/'runs/signal_hard_examples_20261022/control_guard/selected_types.pt']
    checkpoints={str(p):digest(p) for p in frozen}
    reader=DirectionResidualClassifier();graph=ExportGraph(reader).eval()
    source=ROOT/'runs/signal_object_acceptance_20261023'
    original=json.loads((source/'scene_proposals.json').read_text(encoding='utf-8'))['records']
    evidence=json.loads((source/'lower_band_evidence.json').read_text(encoding='utf-8'))['records']
    samples=[];seen=set();folder=PACKAGE/'parity_crops';folder.mkdir(exist_ok=True)
    def add(image,origin):
        key=hashlib.sha256(image.tobytes()+str(image.shape).encode()).hexdigest()
        if key in seen:return
        seen.add(key);name=f'crop_{len(samples):03d}.png';write_image(folder/name,image)
        samples.append(dict(path='parity_crops/'+name,pixel_sha256=key,origin=origin))
    for record in evidence:
        frame=next(r for r in original if (r['clip'],r['frame_id'])==(record['clip'],record['frame_id']))
        assert digest(frame['image'])==frame['image_sha256'];image=read_image(frame['image'])
        for candidate in record['candidates']+record['supplemental']:
            add(crop_head(image,candidate['box']),dict(clip=record['clip'],frame_id=record['frame_id'],kind='actual_detector_proposal'))
    manifest=json.loads((ROOT/'runs/signal_spatial_tail_20261022/manifest.json').read_text(encoding='utf-8'))['records']
    for typ in range(4):
        selected=[r for r in manifest if r['type_id']==typ and r['role']!='train'][:6]
        for r in selected:add(read_image(r['image']),dict(id=r['id'],kind='existing_development_crop',type_id=typ))
    images=[read_image(PACKAGE/r['path']) for r in samples]
    print('ATTR_EXPORT_SAMPLES',len(images),flush=True)
    references=[];output_arrays=[[],[],[],[]];all_values=[];qualities=[]
    for start in range(0,len(images),4):
        batch=images[start:start+4];values,q=inputs(batch)
        s,f,z,quality=encode_spatial(reader.reader,batch)
        desc=torch.from_numpy(np.stack([native_descriptors(im) for im in batch]))
        with torch.inference_mode():
            delta=reader.residual(s,desc)
            expected=(reader.types(s[:,0]),z[:,:6],reader.direction(f[:,1280:])+delta,delta)
            actual=graph(*(torch.from_numpy(values[name]) for name in ['whole','emitter','descriptors']))
        assert max(float((a-b).abs().max()) for a,b in zip(expected,actual))<1e-4,'Graph differs from retained hooks'
        from calibrate_corrected_signal_guard_pycharm import guarded_decode as torch_decode
        p=torch_decode(expected[0],expected[1],expected[2],quality)
        for row,change in zip(p,delta.tolist()):row['direction_residual_logits']=change
        references+=p
        for target,array in zip(output_arrays,expected):target.append(array.numpy())
        all_values.append(values);qualities+=q
    merged={key:np.concatenate([v[key] for v in all_values]) for key in all_values[0]}
    arrays=[np.concatenate(v) for v in output_arrays]
    np.savez_compressed(PACKAGE/'parity_inputs.npz',**merged,quality=np.asarray(qualities),
        **{name:value for name,value in zip(['expected_types','expected_original','expected_direction','expected_delta'],arrays)})
    model=PACKAGE/'models/lamp_attributes.onnx'
    torch.onnx.export(graph,tuple(torch.from_numpy(merged[name][:1]) for name in ['whole','emitter','descriptors']),
        str(model),input_names=['whole','emitter','descriptors'],
        output_names=['types','original','direction','delta'],opset_version=12,dynamo=False,
        dynamic_axes={name:{0:'batch'} for name in ['whole','emitter','descriptors','types','original','direction','delta']})
    onnx.checker.check_model(onnx.load(model));portable=PortableAttributes(model)
    errors=[];decisions=[]
    for start in range(0,len(images),4):
        values,q=inputs(images[start:start+4]);actual=portable.session.run(None,values)
        expected=[a[start:start+4] for a in arrays]
        errors.append(max(float(np.abs(a-b).max()) for a,b in zip(expected,actual)))
        p=guarded_decode(*actual[:3],q,actual[3])
        for index,row in enumerate(p,start):
            assert all(row[key]==references[index][key] for key in DECISIONS),(index,row,references[index])
        decisions+=p
    # Explicit dynamic sizes exercise batch reshaping, not just the traced size.
    for size in [1,3,8]:
        values,q=inputs(images[:size]);actual=portable.session.run(None,values)
        assert all(a.shape[0]==size for a in actual)
    assert max(errors)<1e-3
    assert all(digest(p)==sha for p,sha in checkpoints.items())
    save(PACKAGE/'parity_cases.json',dict(samples=samples,references=references,decision_keys=DECISIONS,
        scope='Existing reviewed development inputs for export parity; not new accuracy or lane-association validation'))
    save(PACKAGE/'metadata.json',dict(checkpoints=checkpoints,preprocessing_sources=sources,
        onnx_sha256=digest(model),samples=len(samples),dynamic_batches_checked=[1,3,8],
        control_enabled=False,physical_lane_association_verified=False,
        model_name='retained_spatial_type_and_direction_residual',not_new_training=True))
    save(OUT/'export_verification.json',dict(samples=len(samples),max_logit_error=max(errors),
        all_discrete_decisions_equal=True,dynamic_batches_checked=[1,3,8],model=str(model),
        source_checkpoints_unchanged=True,board_verified=False,source_hashes=checkpoints))
    print('LAMP_ATTRS_EXPORT_PARITY_PASS',len(samples),max(errors),model,flush=True)


if __name__=='__main__':main()
