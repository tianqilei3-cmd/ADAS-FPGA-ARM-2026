"""Deploy one retained-model lane prototype; never synthesize or train variants."""
import argparse,hashlib,json,subprocess,tarfile
from pathlib import Path
from deploy_arm_a06_20261027 import ssh,run,launch as restore
ROOT=Path(__file__).resolve().parent
PACKAGE=ROOT/'arm_lane_keys_20261027'
REMOTE='/home/orangepi/adas2026/lane_keys_20261027'
OUT=ROOT/'runs/basic_finish70_20261027/lane_keys/arm'

def save(name,data):(OUT/name).write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf8')

def install():
    conversion=json.loads((OUT.parent/'model/conversion.json').read_text())
    models=[dict(path='models/boundary_fp.rknn',source=conversion['path'],sha256=conversion['sha256']),
        dict(path='models/road_rect_fp.rknn',source='/home/orangepi/adas2026/signal_rect_models_20261027/road_rect_fp.rknn',
            sha256='2db828d4bd489873a84be52cf6dd93f04750c96b6fc084bcfe45802a439344b0')]
    manifest=dict(files=[dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in sorted(PACKAGE.glob('*.py'))],
        models=models,source='one retained boundary model and FP road model',new_training=False,physical_lane_association_verified=False,
        key_protocol='55 03 02 STATE SEQ SUM',control_sequence_echo_in_context=False,accepted=False)
    (PACKAGE/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    archive=OUT/'sources.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:
        for item in manifest['files']:tf.add(PACKAGE/item['path'],arcname=item['path'])
        tf.add(PACKAGE/'manifest.json',arcname='manifest.json')
    cfg=json.loads((ROOT/'runs/orangepi/connection_ready.json').read_text())
    run(['scp','-i',cfg['private_key_path'],'-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes',
        '-o','UserKnownHostsFile='+cfg['known_hosts_path'],str(archive),cfg['user']+'@'+cfg['host']+':/home/orangepi/adas2026/lane_keys_upload.tar.gz'])
    code='''from pathlib import Path
import hashlib,json,subprocess,tarfile
root=Path(REMOTE);assert not root.exists()
archive=root.parent/'lane_keys_upload.tar.gz';assert hashlib.sha256(archive.read_bytes()).hexdigest()==DIGEST;root.mkdir()
with tarfile.open(archive) as tf:
 for member in tf.getmembers():assert member.isfile() and root.resolve() in (root/member.name).resolve().parents
 tf.extractall(root)
m=json.loads((root/'manifest.json').read_text())
for i in m['files']:assert hashlib.sha256((root/i['path']).read_bytes()).hexdigest()==i['sha256']
(root/'models').mkdir()
for i in m['models']:
 p=Path(i['source']);assert hashlib.sha256(p.read_bytes()).hexdigest()==i['sha256'];(root/i['path']).symlink_to(p)
p=subprocess.run(['python3',str(root/'verify_lane_keys.py')],cwd=root,capture_output=True,text=True,timeout=30)
assert p.returncode==0,p.stdout+p.stderr
report=dict(installed=True,remote=str(root),tests=json.loads(p.stdout),accepted=False)
(root/'installation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
'''.replace('REMOTE',repr(REMOTE)).replace('DIGEST',repr(hashlib.sha256(archive.read_bytes()).hexdigest()))
    report=json.loads(ssh(code));save('installation.json',report);print(json.dumps(report),flush=True)

def stop():
    code='''import json,os,signal,time
from pathlib import Path
allowed=['/home/orangepi/adas2026/screen_demo_a06_20261027/run_screen_demo.py',REMOTE+'/run_lane_demo.py']
pids=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:cmd=(p/'cmdline').read_bytes().split(b'\\0')
 except OSError:continue
 if not any(v.endswith(b'run_screen_demo.py') or v.endswith(b'run_lane_demo.py') for v in cmd):continue
 assert any(v.encode() in cmd for v in allowed),'Unexpected live runtime owner '+p.name
 pids.append(int(p.name))
for pid in pids:os.kill(pid,signal.SIGTERM)
for _ in range(120):
 alive=[]
 for pid in pids:
  p=Path('/proc')/str(pid)/'stat'
  if p.exists() and p.read_text().split()[2]!='Z':alive.append(pid)
 if not alive:break
 time.sleep(.1)
else:raise RuntimeError('Runtime failed to stop '+repr(alive))
print(json.dumps(dict(stopped=pids)))
'''.replace('REMOTE',repr(REMOTE))
    report=json.loads(ssh(code));save('last_stop.json',report);return report

def start(args):
    stop()
    code='''from pathlib import Path
import json,subprocess,os,time
root=Path(REMOTE);p=root/('launch_'+time.strftime('%Y%m%d_%H%M%S')+'.log')
with p.open('ab') as stream:
 proc=subprocess.Popen(COMMAND,cwd=root,env=dict(os.environ,DISPLAY=':0',XAUTHORITY='/home/orangepi/.Xauthority',XDG_RUNTIME_DIR='/run/user/1000',
  QT_QPA_PLATFORM='xcb',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
(root/'demo.pid').write_text(str(proc.pid));print(json.dumps(dict(pid=proc.pid,command=COMMAND,log=str(p))))
'''.replace('REMOTE',repr(REMOTE))
    command=['python3',REMOTE+'/run_lane_demo.py','--duration',str(args.duration)]
    if args.bench_boundary:command+=['--bench-boundary']
    report=json.loads(ssh(code.replace('COMMAND',repr(command))));save('latest_launch.json',report);print(json.dumps(report),flush=True)

def check():
    code='''from pathlib import Path
import json,time
root=Path(REMOTE);session=Path((root/'last_session.txt').read_text());print(json.dumps(dict(session=str(session),status=json.loads((session/'status.json').read_text()),
 summary=json.loads((session/'summary.json').read_text()) if (session/'summary.json').exists() else None)))
'''.replace('REMOTE',repr(REMOTE))
    report=json.loads(ssh(code));save('last_check.json',report);print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for n in ['install','launch','stop','check']:parser.add_argument('--'+n,action='store_true')
    parser.add_argument('--duration',type=float,default=3600);parser.add_argument('--bench-boundary',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.install:install()
    if args.launch:start(args)
    if args.stop:print(json.dumps(stop()))
    if args.check:check()
