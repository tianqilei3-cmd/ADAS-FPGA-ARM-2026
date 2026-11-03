"""FP conversion and native parity of the completed fixed-epoch candidate."""
import json,hashlib
from pathlib import Path
from deploy_arm_a06_20261027 import run
from deploy_unified_arm_20261028 import ROOT,ssh,stop
OUT=ROOT/'runs/basic_finish70_20261028/lamp_capture_aligned/export'
REMOTE='/home/orangepi/adas2026/lamp_capture_aligned_20261028'
SOURCE=r'''from pathlib import Path
import os,json,hashlib,time
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2';os.nice(5)
from rknn.api import RKNN
root=Path(__file__).parent;model=RKNN(verbose=False);began=time.monotonic()
try:
 model.config(mean_values=[[0,0,0]],std_values=[[255,255,255]],target_platform='rk3566',optimization_level=3)
 assert model.load_onnx(model=str(root/'lamp_aligned_heads.onnx'))==0
 assert model.build(do_quantization=False)==0
 path=root/'lamp_aligned_fp_heads.rknn';assert model.export_rknn(str(path))==0
 report=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,seconds=time.monotonic()-began,quantized=False,adopted=False)
 (root/'conversion.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
finally:model.release()
'''


if __name__=='__main__':
    export=json.loads((OUT/'export.json').read_text());cfg=json.loads((ROOT/'runs/orangepi/connection_ready.json').read_text());stop()
    print(ssh("from pathlib import Path;r=Path("+repr(REMOTE)+");assert not r.exists();r.mkdir();(r/'convert.py').write_text("+repr(SOURCE)+");print('conversion prepared')"),flush=True)
    run(['scp','-i',cfg['private_key_path'],'-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes',
        '-o','UserKnownHostsFile='+cfg['known_hosts_path'],export['archive'],cfg['user']+'@'+cfg['host']+':'+REMOTE+'/inputs.tar.gz'])
    code='''from pathlib import Path
import hashlib,json,tarfile,subprocess,os
root=Path(REMOTE);archive=root/'inputs.tar.gz';assert hashlib.sha256(archive.read_bytes()).hexdigest()==DIGEST
with tarfile.open(archive) as tf:
 for m in tf.getmembers():assert m.isfile() and root.resolve() in (root/m.name).resolve().parents
 tf.extractall(root)
p=subprocess.run(['python3','-u',str(root/'convert.py')],cwd=root,capture_output=True,text=True,timeout=75,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
(root/'conversion.log').write_text(p.stdout+chr(10)+p.stderr)
assert p.returncode==0,p.stdout+p.stderr
print((root/'conversion.json').read_text())
'''.replace('REMOTE',repr(REMOTE)).replace('DIGEST',repr(export['archive_sha256']))
    result=json.loads(ssh(code));(OUT/'conversion.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result),flush=True)
