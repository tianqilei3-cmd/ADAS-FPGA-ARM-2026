"""TRAIN-only calibration of a separate road raw-head INT8 candidate."""
import hashlib,json,tarfile
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/basic_finish70_20261028/road_int8'
REMOTE='/home/orangepi/adas2026/road_int8_20261028'

def main():
    from deploy_unified_arm_20261028 import ssh,stop
    from deploy_arm_a06_20261027 import run
    import sys
    sys.path.insert(0,str(ROOT/'arm_unified_20261028'))
    from vision_common import preprocess
    OUT.mkdir(exist_ok=True);assert not (OUT/'export.json').exists();cal=OUT/'calibration';cal.mkdir();records=[];cv2.setNumThreads(1)
    bdd=ROOT/'datasets/adas/prepared/bdd100k_4class_5000_1000/images/train';paths=sorted(bdd.glob('*.jpg'));assert len(paths)>=96
    selected=[dict(source=str(paths[int(i)]),role='bdd_train') for i in np.linspace(0,len(paths)-1,96,dtype=int)]
    old=json.loads((ROOT/'runs/fast_lamp_rect_20261024/dataset_manifest.json').read_text());official=[];seen=set()
    for r in old['records']:
        if r['split']!='train' or r.get('domain')!='official' or r['source'] in seen:continue
        assert not any(c in (r.get('video') or '') for c in ['112454','113421','113539']);seen.add(r['source']);official.append(r)
    assert len(official)>=32
    selected.extend(dict(source=official[int(i)]['source'],pixel_sha256=official[int(i)]['source_pixel_sha256'],role='official_train') for i in np.linspace(0,len(official)-1,32,dtype=int))
    for i,r in enumerate(selected):
        p=Path(r['source']);im=cv2.imdecode(np.fromfile(p,dtype=np.uint8),cv2.IMREAD_COLOR);assert im is not None
        if r.get('pixel_sha256'):assert hashlib.sha256(im.tobytes()).hexdigest()==r['pixel_sha256']
        rgb,_=preprocess(im,[384,640]);name=f'{i:03d}.png';assert cv2.imwrite(str(cal/name),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        records.append(dict(r,image='calibration/'+name,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    (OUT/'calibration.json').write_text(json.dumps(dict(records=records,train_only=True,validation_images=0,acceptance_images=0),indent=2),encoding='utf8')
    source=ROOT/'runs/basic_finish70_20261028/road_sparse_heads/road_heads.onnx';archive=OUT/'inputs.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:
        tf.add(source,arcname=source.name)
        for p in cal.iterdir():tf.add(p,arcname='calibration/'+p.name)
    result=dict(archive=str(archive),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),onnx_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),calibration_samples=len(records),new_weights=False,quantized=True,adopted=False)
    (OUT/'export.json').write_text(json.dumps(result,indent=2),encoding='utf8');stop()
    code=r'''from pathlib import Path
import os,json,time,hashlib
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2';os.nice(5)
from rknn.api import RKNN
root=Path(__file__).parent;paths=sorted((root/'calibration').glob('*.png'));assert len(paths)==128
(root/'calibration.txt').write_text(chr(10).join(str(p) for p in paths))
model=RKNN(verbose=False);started=time.monotonic()
try:
 model.config(mean_values=[[0,0,0]],std_values=[[255,255,255]],target_platform='rk3566',optimization_level=3)
 assert model.load_onnx(model=str(root/'road_heads.onnx'))==0
 assert model.build(do_quantization=True,dataset=str(root/'calibration.txt'))==0
 output=root/'road_int8_heads.rknn';assert model.export_rknn(str(output))==0
 result=dict(path=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),bytes=output.stat().st_size,seconds=time.monotonic()-started,quantized=True,calibration_samples=128,new_weights=False,adopted=False)
 (root/'conversion.json').write_text(json.dumps(result,indent=2))
finally:model.release()
'''
    print(ssh("from pathlib import Path;r=Path("+repr(REMOTE)+");assert not r.exists();r.mkdir();(r/'convert.py').write_text("+repr(code)+");print('prepared')"),flush=True)
    cfg=json.loads((ROOT/'runs/orangepi/connection_ready.json').read_text());run(['scp','-i',cfg['private_key_path'],'-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+cfg['known_hosts_path'],str(archive),cfg['user']+'@'+cfg['host']+':'+REMOTE+'/inputs.tar.gz'])
    code='''from pathlib import Path
import os,json,hashlib,tarfile,subprocess
root=Path(REMOTE);archive=root/'inputs.tar.gz';assert hashlib.sha256(archive.read_bytes()).hexdigest()==DIGEST
with tarfile.open(archive) as tf:
 for m in tf.getmembers():assert m.isfile() and root.resolve() in (root/m.name).resolve().parents
 tf.extractall(root)
with (root/'convert.log').open('ab') as log:
 p=subprocess.Popen(['python3','-u',str(root/'convert.py')],cwd=root,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True)
(root/'convert.pid').write_text(str(p.pid));print(json.dumps(dict(pid=p.pid,remote=str(root),state='converting')))
'''.replace('REMOTE',repr(REMOTE)).replace('DIGEST',repr(result['archive_sha256']))
    result=json.loads(ssh(code));(OUT/'launch.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
