"""Hash-verified source-only deployment; models reused via checked symlinks."""
import argparse,ast,hashlib,json,subprocess,tarfile
from pathlib import Path
from deploy_arm_a06_20261027 import ssh as legacy_ssh,run
ROOT=Path(__file__).resolve().parent
PACKAGE=ROOT/'arm_unified_20261028'
REMOTE='/home/orangepi/adas2026/unified_20261028'
OUT=ROOT/'runs/basic_finish70_20261028/unified_arm'
CHECKS=['verify_lane_keys.py','verify_compact_boundary.py','verify_capture_quality.py','verify_row_boundary.py','verify_boundary_softmax.py','verify_boundary_cpu.py','verify_unified_policy.py','verify_sparse_road.py','verify_lamp_cpu_protocol.py','verify_transport_age.py','verify_current_housing.py','verify_geometry_proposals.py','verify_patch_quality.py','verify_geometry_worker.py','verify_immutable_identity.py']


def ssh(code):
    ast.parse(code)
    return legacy_ssh(code)


def seal():
    models=[]
    old=json.loads((ROOT/'arm_lane_keys_20261027/manifest.json').read_text())
    for m in old['models']:models.append(dict(path=m['path'],source=m['source'],sha256=m['sha256']))
    source=json.loads((ROOT/'arm_screen_demo_a20_20261027/manifest.json').read_text())
    for item in source['reused_models']:
        if Path(item['path']).name in ['color_guard_fp.rknn','lamp_types_fp.rknn','lamp_attributes.onnx']:
            models.append(dict(path=item['path'],source=item['remote_source'],sha256=item['sha256']))
    m=json.loads((ROOT/'runs/basic_finish70_20261027/s3/lamp_int8_heads/fp_conversion.json').read_text())['records'][0]
    models.append(dict(path='models/lamp_rect_fp_heads.rknn',source=m['path'],sha256=m['sha256']))
    candidate=ROOT/'runs/basic_finish70_20261028/lamp_capture_aligned/export/conversion.json'
    parity=candidate.with_name('native_parity.json')
    if candidate.exists() and parity.exists() and json.loads(parity.read_text())['eligible_for_bounded_trial']:
        m=json.loads(candidate.read_text())
        models.append(dict(path='models/lamp_aligned_fp_heads.rknn',source=m['path'],sha256=m['sha256']))
    road=ROOT/'runs/basic_finish70_20261028/road_sparse_heads/conversion.json'
    if road.exists() and road.with_name('native_parity.json').exists() and json.loads(road.with_name('native_parity.json').read_text())['eligible_for_bounded_trial']:
        m=json.loads(road.read_text());models.append(dict(path='models/road_heads_fp.rknn',source=m['path'],sha256=m['sha256']))
    road_int8=ROOT/'runs/basic_finish70_20261028/road_int8/conversion.json'
    if road_int8.exists() and road_int8.with_name('native_parity.json').exists() and json.loads(road_int8.with_name('native_parity.json').read_text())['eligible_for_bounded_trial']:
        m=json.loads(road_int8.read_text());models.append(dict(path='models/road_int8_heads.rknn',source=m['path'],sha256=m['sha256']))
    for name,source,remote in [
        ('color_guard.onnx',ROOT/'arm_screen_demo_a02_20261024/models/color_guard.onnx','/home/orangepi/adas2026/screen_demo_a02_20261024/models/color_guard.onnx'),
        ('lamp_types_whole.onnx',ROOT/'runs/basic_finish70_20261027/s3/fast_types/lamp_types_whole_only.onnx','/home/orangepi/adas2026/signal_classifiers_npu_20261027/lamp_types.onnx')]:
        models.append(dict(path='models/'+name,source=remote,sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    batch=ROOT/'runs/basic_finish70_20261028/batched_classifiers/conversion.json'
    if batch.exists() and batch.with_name('native_parity.json').exists() and json.loads(batch.with_name('native_parity.json').read_text())['eligible_for_bounded_trial']:
        for m in json.loads(batch.read_text())['records']:models.append(dict(path='models/'+Path(m['path']).name,source=m['path'],sha256=m['sha256']))
    batch2=ROOT/'runs/basic_finish70_20261028/batch2_classifiers/conversion.json'
    if batch2.exists() and batch2.with_name('native_parity.json').exists() and json.loads(batch2.with_name('native_parity.json').read_text())['eligible_for_bounded_trial']:
        for m in json.loads(batch2.read_text())['records']:models.append(dict(path='models/'+Path(m['path']).name,source=m['path'],sha256=m['sha256']))
    files=[]
    for p in sorted(PACKAGE.iterdir()):
        if not p.is_file() or p.name=='manifest.json':continue
        if p.suffix not in ['.py','.json']:continue
        s=p.read_text(encoding='utf8');p.write_text(s,encoding='utf8',newline='\n')
        if p.suffix=='.py':ast.parse(s)
        files.append(dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
    result=dict(profile=40,supported_profiles=[40,41],full_resolution_requires_profile=41,files=files,models=models,new_weights=any(m['path'].endswith('lamp_aligned_fp_heads.rknn') for m in models),accepted=False,
        default_lamp_profile='retained',default_road_profile='decoded',candidate_profiles_are_not_automatically_adopted=True,
        scope='Physical key/test driver controlled virtual signal/lane/risk; no physical calibration claim',
        TTL_ms=500,key_epoch_required=True,recovery='Discard stale capture; park before source recovery; preserve hazard tracks',
        memory='One NPU owner; one latest capture; one bounded shared CPU worker; one evidence thread/job',checks=CHECKS)
    (PACKAGE/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf8',newline='\n');return result


def stop():
    code='''from pathlib import Path
import json,os,signal,time
allowed=ALLOWED;found=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:cmd=(p/'cmdline').read_bytes().split(b'\\0')
 except OSError:continue
 if not any(v.endswith(b'/run_lane_demo.py') or v.endswith(b'/run_screen_demo.py') or v.endswith(b'/run_unified_demo.py') for v in cmd):continue
 assert any(v.encode() in cmd for v in allowed),'Unrecognized runtime '+p.name
 found.append(int(p.name))
for pid in found:
 try:os.kill(pid,signal.SIGTERM)
 except ProcessLookupError:pass
def alive(pid):
 try:return (Path('/proc')/str(pid)/'stat').read_text().split()[2]!='Z'
 except FileNotFoundError:return False
for _ in range(150):
 live=[pid for pid in found if alive(pid)]
 if not live:break
 time.sleep(.1)
else:raise RuntimeError('Runtime did not release resources '+repr(live))
print(json.dumps(dict(stopped=found)))
'''.replace('ALLOWED',repr([REMOTE+'/run_unified_demo.py','/home/orangepi/adas2026/unified_20261028/run_unified_demo.py','/home/orangepi/adas2026/phase_trial_20261028/run_unified_demo.py','/home/orangepi/adas2026/tx_reinit_trial_20261028/run_unified_demo.py','/home/orangepi/adas2026/signal_frozen_20261028/run_unified_demo.py','/home/orangepi/adas2026/signal_recovered_20261028/run_unified_demo.py','/home/orangepi/adas2026/lane_keys_20261027/run_lane_demo.py','/home/orangepi/adas2026/screen_demo_a06_20261027/run_screen_demo.py']))
    result=json.loads(ssh(code));OUT.mkdir(parents=True,exist_ok=True);(OUT/'last_stop.json').write_text(json.dumps(result,indent=2),encoding='utf8');return result


def install_or_update():
    OUT.mkdir(parents=True,exist_ok=True);manifest=seal();archive=OUT/'sources.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:
        for item in manifest['files']:tf.add(PACKAGE/item['path'],arcname=item['path'])
        tf.add(PACKAGE/'manifest.json',arcname='manifest.json')
    cfg=json.loads((ROOT/'runs/orangepi/connection_ready.json').read_text())
    run(['scp','-i',cfg['private_key_path'],'-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes',
        '-o','UserKnownHostsFile='+cfg['known_hosts_path'],str(archive),cfg['user']+'@'+cfg['host']+':/home/orangepi/adas2026/unified_sources_upload.tar.gz'])
    stop()
    code='''from pathlib import Path
import hashlib,json,tarfile,shutil,subprocess,os,time
root=Path(REMOTE);archive=root.parent/'unified_sources_upload.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest()==DIGEST
staging=root.parent/('unified_staging_'+time.strftime('%Y%m%d_%H%M%S'));staging.mkdir()
with tarfile.open(archive) as tf:
 for m in tf.getmembers():assert m.isfile() and staging.resolve() in (staging/m.name).resolve().parents
 tf.extractall(staging)
manifest=json.loads((staging/'manifest.json').read_text());history=None
for item in manifest['files']:assert hashlib.sha256((staging/item['path']).read_bytes()).hexdigest()==item['sha256']
if root.exists():
 old=json.loads((root/'manifest.json').read_text());history=root/'source_repairs'/time.strftime('%Y%m%d_%H%M%S');history.mkdir(parents=True)
 for item in old['files']:
  p=root/item['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'];shutil.copy2(p,history/p.name)
 shutil.copy2(root/'manifest.json',history/'manifest.json')
else:root.mkdir()
for item in manifest['files']:shutil.copy2(staging/item['path'],root/item['path'])
shutil.copy2(staging/'manifest.json',root/'manifest.json');(root/'models').mkdir(exist_ok=True)
for item in manifest['models']:
 source=Path(item['source']);assert hashlib.sha256(source.read_bytes()).hexdigest()==item['sha256']
 target=root/item['path']
 if target.is_symlink():target.unlink()
 else:assert not target.exists(),'Never overwrite model file'
 target.symlink_to(source)
checks=[]
for name in manifest['checks']:
 p=subprocess.run(['python3',str(root/name)],cwd=root,capture_output=True,text=True,timeout=30,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
 assert p.returncode==0,p.stdout+p.stderr
 checks.append(dict(suite=name,**json.loads(p.stdout)))
print(json.dumps(dict(installed=True,remote=str(root),history=str(history),checks=checks)))
'''.replace('REMOTE',repr(REMOTE)).replace('DIGEST',repr(hashlib.sha256(archive.read_bytes()).hexdigest()))
    result=json.loads(ssh(code));(OUT/'installation.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result),flush=True)


def start(duration=21600,capture_mode='720',headless=False,lamp_profile='retained',road_profile='decoded',classifiers='npu',video_layout='tiles',windowed=False,housing_refinement=False,proposal_mode='neural',capture_fps=30):
    stop();command=['python3',REMOTE+'/run_unified_demo.py','--duration',str(duration),'--capture-mode',capture_mode]
    command+=['--lamp-profile',lamp_profile]
    command+=['--road-profile',road_profile]
    command+=['--classifiers',classifiers]
    command+=['--video-layout',video_layout]
    if windowed:command.append('--windowed')
    if housing_refinement:command.append('--housing-refinement')
    command+=['--proposal-mode',proposal_mode]
    command+=['--capture-fps',str(capture_fps)]
    if headless:command.append('--headless')
    code='''from pathlib import Path
import subprocess,os,json,time
root=Path(REMOTE);log=root/('launch_'+time.strftime('%Y%m%d_%H%M%S')+'.log')
with log.open('ab') as f:
 p=subprocess.Popen(COMMAND,cwd=root,env=dict(os.environ,DISPLAY=':0',XAUTHORITY='/home/orangepi/.Xauthority',XDG_RUNTIME_DIR='/run/user/1000',QT_QPA_PLATFORM='xcb',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(root/'demo.pid').write_text(str(p.pid));print(json.dumps(dict(pid=p.pid,command=COMMAND,log=str(log))))
'''.replace('REMOTE',repr(REMOTE)).replace('COMMAND',repr(command))
    result=json.loads(ssh(code));(OUT/'latest_launch.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result),flush=True);return result


def check():
    code='''from pathlib import Path
import json
root=Path(REMOTE);s=Path((root/'last_session.txt').read_text());p=s/'status.json'
print(json.dumps(dict(session=str(s),status=json.loads(p.read_text()) if p.exists() else None)))
'''.replace('REMOTE',repr(REMOTE))
    result=json.loads(ssh(code));print(json.dumps(result),flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['install','launch','stop','check']:p.add_argument('--'+name,action='store_true')
    p.add_argument('--duration',type=float,default=21600);p.add_argument('--capture-mode',choices=['720','1080'],default='720');p.add_argument('--headless',action='store_true')
    p.add_argument('--capture-fps',type=int,choices=[10,20,30],default=30)
    p.add_argument('--lamp-profile',choices=['retained','capture_ft'],default='retained')
    p.add_argument('--road-profile',choices=['decoded','sparse','int8'],default='decoded')
    p.add_argument('--classifiers',choices=['npu','batch2','batch3','cpu_process'],default='npu')
    p.add_argument('--video-layout',choices=['tiles','full'],default='tiles');p.add_argument('--windowed',action='store_true');p.add_argument('--housing-refinement',action='store_true')
    p.add_argument('--proposal-mode',choices=['neural','geometry'],default='neural');a=p.parse_args()
    if a.install:install_or_update()
    if a.stop:print(json.dumps(stop()))
    if a.launch:start(a.duration,a.capture_mode,a.headless,a.lamp_profile,a.road_profile,a.classifiers,a.video_layout,a.windowed,a.housing_refinement,a.proposal_mode,a.capture_fps)
    if a.check:check()
