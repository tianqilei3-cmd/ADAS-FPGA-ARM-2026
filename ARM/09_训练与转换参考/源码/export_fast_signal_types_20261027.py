"""Remove unused emitter-batch computation from the frozen type-only graph."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'runs/orangepi/export_dependencies'))
sys.path.insert(0,str(ROOT/'arm_lamp_attrs_20261024'))
import numpy as np
import onnx
from onnx.utils import Extractor
import onnxruntime as ort


def main():
    source=ROOT/'arm_lamp_attrs_20261024/models/lamp_attributes.onnx'
    out=ROOT/'runs/basic_finish70_20261027/s3/fast_types'
    out.mkdir(exist_ok=True)
    target=out/'lamp_types_whole_only.onnx'
    assert not target.exists(),'Existing export preserved'
    graph=onnx.load(str(source))
    concat=next(n for n in graph.graph.node if n.name=='/Concat')
    assert list(concat.input)==['whole','emitter'] and concat.attribute[0].i==0
    output=concat.output[0]
    consumers=[n for n in graph.graph.node if output in n.input]
    assert len(consumers)==1 and consumers[0].op_type=='Conv'
    consumers[0].input[0]='whole'
    graph=Extractor(graph).extract_model(['whole'],['types','original'])
    onnx.checker.check_model(graph)
    onnx.save(graph,str(target))
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    options.add_session_config_entry('session.intra_op.allow_spinning','0')
    full=ort.InferenceSession(str(source),options,providers=['CPUExecutionProvider'])
    fast=ort.InferenceSession(str(target),options,providers=['CPUExecutionProvider'])
    cache=np.load(ROOT/'arm_lamp_attrs_20261024/parity_inputs.npz')
    errors=[]
    count=len(cache['whole'])
    for start in range(0,count,4):
        values={k:cache[k][start:start+4] for k in ['whole','emitter','descriptors']}
        reference=full.run(['types','original'],values)
        actual=fast.run(None,{'whole':values['whole']})
        errors.append(max(float(np.max(np.abs(a-b))) for a,b in zip(reference,actual)))
    for size in [1,3,8]:
        values={k:cache[k][:size] for k in ['whole','emitter','descriptors']}
        a=fast.run(None,{'whole':values['whole']})
        b=full.run(['types','original'],values)
        errors.append(max(float(np.max(np.abs(x-y))) for x,y in zip(a,b)))
    assert max(errors)<1e-3,errors
    report=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        exported_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),model=str(target),
        prepared_crops=count,dynamic_batches=[1,3,8],maximum_type_logit_error=max(errors),
        type_thresholds_changed=False,weights_changed=False,new_training=False,
        description='Same whole-image encoder and type heads; emitter/direction excluded',
        scope='Type-output numerical parity only; round/arrow policy separately evaluated')
    (out/'export_parity.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report))


if __name__=='__main__':main()
