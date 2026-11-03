"""Move frozen color and type CNNs to FP NPU execution, with separate models."""
import hashlib,json,tarfile
from pathlib import Path
from deploy_arm_a06_20261027 import ssh,run
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/basic_finish70_20261027/s3/classifiers_npu'
REMOTE='/home/orangepi/adas2026/signal_classifiers_npu_20261027'
SOURCE=r'''import hashlib,json,os,time
from pathlib import Path
os.nice(5)
from rknn.api import RKNN
root=Path(__file__).parent;records=[]
for name,input_name in [('color_guard','images'),('lamp_types','whole')]:
 model=RKNN(verbose=False);started=time.monotonic()
 try:
  model.config(mean_values=[[0,0,0]],std_values=[[255,255,255]],target_platform='rk3566',optimization_level=3)
  assert model.load_onnx(model=str(root/(name+'.onnx')),inputs=[input_name],input_size_list=[[1,3,128,128]])==0
  assert model.build(do_quantization=False)==0
  output=root/(name+'_fp.rknn');assert model.export_rknn(str(output))==0
  records.append(dict(name=name,path=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
    onnx_sha256=hashlib.sha256((root/(name+'.onnx')).read_bytes()).hexdigest(),bytes=output.stat().st_size,seconds=time.monotonic()-started))
 finally:model.release()
(root/'conversion.json').write_text(json.dumps(dict(records=records,new_training=False,quantization=False,accepted=False),indent=2))
print(json.dumps(records),flush=True)
'''


def main():
    OUT.mkdir(exist_ok=True)
    paths=[('color_guard',ROOT/'arm_screen_demo_a02_20261024/models/color_guard.onnx'),
           ('lamp_types',ROOT/'runs/basic_finish70_20261027/s3/fast_types/lamp_types_whole_only.onnx')]
    archive=OUT/'inputs.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:
        for name,path in paths:tf.add(path,arcname=name+'.onnx')
    code="from pathlib import Path;r=Path(REMOTE);assert not r.exists();r.mkdir();(r/'convert.py').write_text(SOURCE);print('prepared')"
    print(ssh(code.replace('REMOTE',repr(REMOTE)).replace('SOURCE',repr(SOURCE))).strip(),flush=True)
    cfg=json.loads((ROOT/'runs/orangepi/connection_ready.json').read_text(encoding='utf8'))
    run(['scp','-i',cfg['private_key_path'],'-P',str(cfg['port']),'-o','BatchMode=yes','-o','IdentitiesOnly=yes',
        '-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+cfg['known_hosts_path'],str(archive),
        cfg['user']+'@'+cfg['host']+':'+REMOTE+'/inputs.tar.gz'])
    code='''import hashlib,json,os,subprocess,tarfile
from pathlib import Path
root=Path(REMOTE);archive=root/'inputs.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest()==DIGEST
with tarfile.open(archive) as tf:
 for m in tf.getmembers():assert m.isfile() and root.resolve() in (root/m.name).resolve().parents
 tf.extractall(root)
with (root/'convert.log').open('ab') as stream:
 p=subprocess.Popen(['python3','-u',str(root/'convert.py')],cwd=root,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),
  stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
(root/'convert.pid').write_text(str(p.pid));print(json.dumps(dict(pid=p.pid,remote=str(root),state='converting')))
'''.replace('REMOTE',repr(REMOTE)).replace('DIGEST',repr(hashlib.sha256(archive.read_bytes()).hexdigest()))
    result=json.loads(ssh(code));(OUT/'conversion_launch.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
