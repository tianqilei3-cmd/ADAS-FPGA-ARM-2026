"""FP batch3 conversion for existing classifiers, no retraining/quantization."""
import json
from deploy_unified_arm_20261028 import ROOT,ssh,stop
OUT=ROOT/'runs/basic_finish70_20261028/batched_classifiers'
REMOTE='/home/orangepi/adas2026/batched_classifiers_20261028'
SOURCE=r'''from pathlib import Path
import os,json,time,hashlib,shutil
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2';os.nice(5)
from rknn.api import RKNN
root=Path(__file__).parent;base=Path('/home/orangepi/adas2026/signal_classifiers_npu_20261027');records=[]
for name,input_name in [('color_guard','images'),('lamp_types','whole')]:
 model=RKNN(verbose=False);start=time.monotonic();source=base/(name+'.onnx');digest=hashlib.sha256(source.read_bytes()).hexdigest()
 try:
  model.config(mean_values=[[0,0,0]],std_values=[[255,255,255]],target_platform='rk3566',optimization_level=3)
  assert model.load_onnx(model=str(source),inputs=[input_name],input_size_list=[[3,3,128,128]])==0
  assert model.build(do_quantization=False)==0
  target=root/(name+'_batch3_fp.rknn');assert model.export_rknn(str(target))==0
  assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
  records.append(dict(name=name,path=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),source_onnx_sha256=digest,seconds=time.monotonic()-start,batch=3))
 finally:model.release()
(root/'conversion.json').write_text(json.dumps(dict(records=records,new_weights=False,quantized=False,adopted=False),indent=2))
'''
if __name__=='__main__':
    OUT.mkdir(exist_ok=True);stop()
    print(ssh("from pathlib import Path;r=Path("+repr(REMOTE)+");assert not r.exists();r.mkdir();(r/'convert.py').write_text("+repr(SOURCE)+");print('prepared')"),flush=True)
    code='''from pathlib import Path
import subprocess,os
root=Path(REMOTE)
p=subprocess.run(['python3','-u',str(root/'convert.py')],cwd=root,capture_output=True,text=True,timeout=75,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
(root/'conversion.log').write_text(p.stdout+chr(10)+p.stderr);assert p.returncode==0,p.stdout+p.stderr
print((root/'conversion.json').read_text())
'''.replace('REMOTE',repr(REMOTE))
    data=json.loads(ssh(code));(OUT/'conversion.json').write_text(json.dumps(data,indent=2),encoding='utf8');print(json.dumps(data),flush=True)
