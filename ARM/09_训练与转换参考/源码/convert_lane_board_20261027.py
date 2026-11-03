"""Convert the retained boundary model to FP on the existing Orange Pi."""
import json
from pathlib import Path
from deploy_arm_a06_20261027 import ssh,run
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/basic_finish70_20261027/lane_keys/model'
REMOTE='/home/orangepi/adas2026/lane_boundary_fp_20261027'
SOURCE=r'''import hashlib,json,os,time
from pathlib import Path
os.nice(5)
from rknn.api import RKNN
root=Path(__file__).parent;model=RKNN(verbose=False);started=time.monotonic()
try:
 model.config(mean_values=[[0,0,0]],std_values=[[255,255,255]],target_platform='rk3566',optimization_level=3)
 assert model.load_onnx(model=str(root/'boundary.onnx'))==0
 assert model.build(do_quantization=False)==0
 output=root/'boundary_fp.rknn';assert model.export_rknn(str(output))==0
 result=dict(path=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),bytes=output.stat().st_size,
   seconds=time.monotonic()-started,quantization=False,new_training=False,accepted=False)
 (root/'conversion.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
finally:model.release()
'''
if __name__=='__main__':
    report=json.loads((OUT/'export.json').read_text());cfg=json.loads((ROOT/'runs/orangepi/connection_ready.json').read_text())
    print(ssh("from pathlib import Path;r=Path(REMOTE);assert not r.exists();r.mkdir();(r/'convert.py').write_text(SOURCE);print('prepared')".replace('REMOTE',repr(REMOTE)).replace('SOURCE',repr(SOURCE))).strip(),flush=True)
    run(['scp','-i',cfg['private_key_path'],'-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes',
        '-o','UserKnownHostsFile='+cfg['known_hosts_path'],report['archive'],cfg['user']+'@'+cfg['host']+':'+REMOTE+'/inputs.tar.gz'])
    code='''from pathlib import Path
import hashlib,json,os,subprocess,tarfile
root=Path(REMOTE);archive=root/'inputs.tar.gz';assert hashlib.sha256(archive.read_bytes()).hexdigest()==DIGEST
with tarfile.open(archive) as tf:
 for m in tf.getmembers():assert m.isfile() and root.resolve() in (root/m.name).resolve().parents
 tf.extractall(root)
with (root/'convert.log').open('ab') as stream:
 p=subprocess.Popen(['python3','-u',str(root/'convert.py')],cwd=root,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),
  stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid,remote=str(root))))
'''.replace('REMOTE',repr(REMOTE)).replace('DIGEST',repr(report['archive_sha256']))
    result=json.loads(ssh(code));(OUT/'conversion_launch.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result),flush=True)
