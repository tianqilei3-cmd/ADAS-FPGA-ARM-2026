"""Verified source-only A04 installation, optional launch, and bounded inspection."""
import argparse
import hashlib
import json
import py_compile
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / 'arm_screen_demo_a06_20261027'
OUT = ROOT / 'runs/basic_finish70_20261027/a06'
REMOTE = '/home/orangepi/adas2026/screen_demo_a06_20261027'


def run(command, **kwargs):
    return subprocess.run(command, check=True, text=True, encoding='utf8',
                          capture_output=True, timeout=90, **kwargs)


def ssh(code):
    cfg = json.loads((ROOT / 'runs/orangepi/connection_ready.json').read_text(encoding='utf8'))
    command = ['ssh', '-i', cfg['private_key_path'], '-p', str(cfg['port']),
               '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'ConnectTimeout=8',
               '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile=' + cfg['known_hosts_path'],
               cfg['user'] + '@' + cfg['host'], 'python3 -']
    try:
        return run(command, input=code).stdout
    except subprocess.CalledProcessError as error:
        raise RuntimeError('Board command failed:\n'+error.stdout+'\n'+error.stderr) from None


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')


def install():
    manifest = json.loads((PACKAGE / 'manifest.json').read_text(encoding='utf8'))
    for item in manifest['files']:
        path = PACKAGE / item['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], item['path']
        if path.suffix == '.py':
            py_compile.compile(str(path), doraise=True)
    test = run(['D:/anacade/envs/yolo/python.exe', '-X', 'utf8',
                str(ROOT / 'verify_arm_a04_scene_20261027.py')])
    save('local_scene_checks.json', json.loads(test.stdout))
    archive = OUT / 'a06_sources.tar.gz'
    with tarfile.open(archive, 'w:gz') as tf:
        for item in manifest['files']:
            tf.add(PACKAGE / item['path'], arcname=item['path'])
        tf.add(PACKAGE / 'manifest.json', arcname='manifest.json')
        tf.add(ROOT / 'verify_arm_a04_scene_20261027.py', arcname='verify_arm_a04_scene_20261027.py')
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    cfg = json.loads((ROOT / 'runs/orangepi/connection_ready.json').read_text(encoding='utf8'))
    command = ['scp', '-i', cfg['private_key_path'], '-P', str(cfg['port']), '-o', 'BatchMode=yes',
               '-o', 'IdentitiesOnly=yes', '-o', 'ConnectTimeout=8', '-o', 'StrictHostKeyChecking=yes',
               '-o', 'UserKnownHostsFile=' + cfg['known_hosts_path'], str(archive),
               cfg['user'] + '@' + cfg['host'] + ':/home/orangepi/adas2026/a06_sources_upload.tar.gz']
    run(command)
    code = '''import hashlib,json,subprocess,tarfile
from pathlib import Path
root=Path(REMOTE)
archive=root.parent/'a06_sources_upload.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest()==DIGEST
assert not root.exists(), 'Existing A04 preserved; use --check instead'
root.mkdir()
with tarfile.open(archive,'r:gz') as tf:
 for member in tf.getmembers():
  assert member.isfile() and root.resolve() in (root/member.name).resolve().parents
 tf.extractall(root)
manifest=json.loads((root/'manifest.json').read_text())
for item in manifest['files']:
 assert hashlib.sha256((root/item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
base=Path(manifest['model_base'])
for item in manifest['reused_models']:
 assert hashlib.sha256((base/item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
(root/'models').symlink_to(base/'models',target_is_directory=True)
checks=[]
for name in ['verify_uart_ack.py','verify_arm_a04_scene_20261027.py','verify_live_capture_20261027.py','verify_basic70_wire_20261027.py','verify_basic70_uart_20261027.py']:
 p=subprocess.run(['python3',str(root/name)],capture_output=True,text=True,timeout=40,cwd=root)
 assert p.returncode==0,p.stdout+p.stderr
 checks.append(dict(name=name,report=json.loads(p.stdout)))
result=dict(state='installed_checked_not_launched',root=str(root),checks=checks,
 models_reused=True,new_model_bytes=0,archive_sha256=DIGEST)
(root/'installation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
'''.replace('REMOTE', repr(REMOTE)).replace('DIGEST', repr(digest))
    result = json.loads(ssh(code))
    save('installation.json', result)
    print(json.dumps(result, ensure_ascii=False))


def launch(duration, headless, frames, go):
    code = '''import json,os,subprocess,time
from pathlib import Path
root=Path(REMOTE)
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:command=(proc/'cmdline').read_bytes()
 except OSError:continue
 if b'run_screen_demo.py' in command and b'python' in command and b'/adas2026/' in command:
  raise RuntimeError('Another demo is running: '+proc.name)
logs=root/'launcher_logs';logs.mkdir(exist_ok=True)
log=logs/(time.strftime('%Y%m%d_%H%M%S')+'.log')
env=dict(os.environ,DISPLAY=':0',XAUTHORITY='/home/orangepi/.Xauthority',
 XDG_RUNTIME_DIR='/run/user/1000',QT_QPA_PLATFORM='xcb',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
command=['python3',str(root/'run_screen_demo.py'),'--road-only','--stride','1','--duration',str(DURATION)]
command+=['--device','/dev/video0','--road-roi','0','0','.5','.5']
if HEADLESS:command+=['--headless']
if FRAMES:command+=['--max-frames',str(FRAMES)]
if GO:command+=['--go']
with log.open('ab') as stream:
 p=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
(root/'demo.pid').write_text(str(p.pid))
print(json.dumps(dict(pid=p.pid,command=command,log=str(log),duration=DURATION,live_capture=True)))
'''.replace('REMOTE', repr(REMOTE)).replace('DURATION', repr(duration)).replace('HEADLESS', repr(headless)).replace('FRAMES', repr(frames)).replace('GO', repr(go))
    result = json.loads(ssh(code))
    save('latest_launch.json', result)
    print(json.dumps(result, ensure_ascii=False))


def check():
    code = '''import json,time
from pathlib import Path
root=Path(REMOTE);session=Path((root/'last_session.txt').read_text())
result=dict(checked_unix=time.time(),session=str(session))
for name in ['status.json','summary.json']:
 if (session/name).exists():result[name]=json.loads((session/name).read_text())
if (session/'detections.jsonl').exists():
 rows=[]
 with (session/'detections.jsonl').open() as f:
  for line in f:
   try:row=json.loads(line)
   except json.JSONDecodeError:continue
   rows.append(row)
 result['records']=len(rows)
 result['last_record']=rows[-1] if rows else None
 result['status_ack']=sum(r.get('uart_status_acknowledged',False) for r in rows)
 result['all_ack']=sum(r.get('uart_all_acknowledged',False) for r in rows)
 result['hazard_entries']=sum(bool(r['hazard_result']['active_tracks']) for r in rows)
 result['outside_only_person_frames']=sum(bool(r['hazard_observation']['objects']) and not any(o['in_corridor'] for o in r['hazard_observation']['objects']) for r in rows)
 result['action_counts']={a:sum(r['arm_decision']['action']==a for r in rows) for a in ['STOP','RISK_CLEAR']}
 for key in ['total_pipeline_ms','road_age_since_file_read_ms','result_interval_ms']:
  values=sorted(r[key] for r in rows[1:] if r.get(key) is not None)
  if values:result[key]=dict(mean=sum(values)/len(values),p95=values[min(len(values)-1,int(.95*len(values)))],maximum=max(values))
print(json.dumps(result))
'''.replace('REMOTE', repr(REMOTE))
    result = json.loads(ssh(code))
    save('latest_check.json', result)
    compact = dict(result)
    compact.pop('last_record', None)
    print(json.dumps(compact, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--install', action='store_true')
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--duration', type=float, default=600.)
    parser.add_argument('--frames', type=int, default=0)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--go', action='store_true')
    args = parser.parse_args()
    if not 0 < args.duration <= 14400 or args.frames < 0:
        parser.error('Invalid run bounds')
    OUT.mkdir(parents=True, exist_ok=True)
    if args.install:
        install()
    if args.launch:
        launch(args.duration, args.headless, args.frames, args.go)
    if args.check:
        check()


if __name__ == '__main__':
    main()
