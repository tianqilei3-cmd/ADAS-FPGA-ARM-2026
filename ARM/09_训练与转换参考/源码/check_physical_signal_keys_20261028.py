"""Read-only physical-key evidence collector; never sends driver/UART commands."""
import argparse
import ast
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from deploy_unified_arm_20261028 import ROOT, ssh
from run_unified_acceptance_20261028 import score_signal, standard

REMOTE = '/home/orangepi/adas2026/signal_recovered_20261028'


def observer_ssh(code, timeout):
    """Only this read-only observer may wait for a human beyond the normal90s RPC."""
    ast.parse(code)
    cfg=json.loads((ROOT/'runs/orangepi/connection_ready.json').read_text())
    command=['ssh','-i',cfg['private_key_path'],'-p',str(cfg['port']),
             '-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','ConnectTimeout=8',
             '-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+cfg['known_hosts_path'],
             cfg['user']+'@'+cfg['host'],'python3 -']
    result=subprocess.run(command,input=code,capture_output=True,text=True,encoding='utf8',timeout=timeout)
    if result.returncode:raise RuntimeError('Read-only key observer failed: '+result.stderr)
    return result.stdout


def score(case, row):
    key = row['key_snapshot']
    physical = key.get('source') == 'physical_keys' and not key['state'] & 64
    if case['id'] != 'physical_stop':
        result = score_signal(case, row)
    else:
        feedback = row.get('feedback') or {}
        receipt = row.get('receipt') or {}
        identity = row['identity']
        checks = dict(input_valid=row['capture_quality']['valid'], fresh=0 <= row['profile']['final_age_ms'] <= 500,
            unchanged=row['key_unchanged'], driver_stopped=not key['state'] & 16,
            arm_stopped=row['decision']['action'] == 'STOP',
            identity=feedback.get('frame_id') == (identity['frame_id'] & 65535)
                     and feedback.get('epoch') == key['sequence'] and feedback.get('scene') == 1
                     and feedback.get('camera_id') == 0,
            acknowledged=bool(receipt.get('status_acknowledged')),
            feedback_stopped=feedback.get('action') == 1 and feedback.get('decision_valid', False)
                             and feedback.get('reject_reason') == 0 and not feedback.get('link_timeout', True))
        grant = row['decision']['action'] == 'GO' or feedback.get('decision_valid') and feedback.get('action') in [2, 3, 4, 5, 6]
        result = dict(passed=all(checks.values()), checks=checks, unsafe=['grant_after_physical_stop'] if grant else [])
    result['checks']['actual_physical_key_source'] = physical
    result['passed'] = result['passed'] and physical
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case', choices=['green_go', 'red_stop', 'forced_red', 'physical_stop'], required=True)
    p.add_argument('--seconds', type=int, default=60)
    p.add_argument('--wait-key-seconds', type=int, default=300)
    a = p.parse_args()
    assert 30 <= a.seconds <= 60
    assert 30 <= a.wait_key_seconds <= 600
    case = next((c for c in standard()['cases'] if c['id'] == a.case), dict(id='physical_stop'))
    desired = 0 if a.case == 'physical_stop' else 48 if a.case == 'forced_red' else 16
    expected_previous = [16, 48] if a.case == 'physical_stop' else [16] if a.case == 'forced_red' else [0]
    fixture = None
    if a.case != 'physical_stop':
        fixture = json.loads((ROOT / 'runs/basic_finish70_20261027/player_status.json').read_text())
        import time
        assert fixture['state'] == 'playing' and time.time() - fixture['updated_unix'] < 4
        assert Path(fixture['video']).stem.endswith(case['clip'])
        assert fixture['start_frame'] == case['start'] and fixture['end_frame_exclusive'] == case['end']
    initial = json.loads(ssh('''from pathlib import Path
import json,time
root=Path(REMOTE);session=Path((root/'last_session.txt').read_text());deadline=time.monotonic()+12
while time.monotonic()<deadline:
 s=json.loads((session/'status.json').read_text());feedback=s.get('feedback') or {}
 if s.get('state') in ('failed','stopped'):raise RuntimeError('Runtime not running: '+str(s))
 if feedback.get('build_tag') not in (None,41):raise RuntimeError('Unexpected FPGA profile: '+str(feedback))
 if s.get('state')=='running' and time.time()-s.get('updated_unix',0)<3 and feedback.get('build_tag')==41 and s.get('uart_confirmed'):break
 time.sleep(.1)
else:raise RuntimeError('No fresh confirmed0x29 feedback before observer arming')
print(json.dumps(dict(session=str(session),status=s)))
'''.replace('REMOTE', repr(REMOTE))))
    out = ROOT / 'runs/basic_finish70_20261028/physical_signal_keys' / datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC')
    out.mkdir(parents=True)
    rule = dict(case=case, required_frames=90, minimum_ratio=.7, desired_state=desired,
                expected_previous_states=expected_previous, initial=initial,
                actual_PC_fixture=fixture, collector_read_only=True,
                requires_new_physical_key_epoch=True, physical_lane_association_verified=False,
                wait_key_seconds=a.wait_key_seconds, sample_timeout_seconds=a.seconds,
                scope='Physical key state and pixel/ARM/FPGA feedback; LED/buzzer observation remains manual')
    (out / 'standard.json').write_text(json.dumps(rule, indent=2), encoding='utf8')
    print('READY: press the physical key for ' + a.case + '; source or remote-driver changes interrupt this check.', flush=True)
    code = '''from pathlib import Path
import json,time
session=Path(SESSION);first=FIRST;deadline=time.monotonic()+WAIT_KEY;rows=[];target_epoch=None;interruption=None
previous_state=PREVIOUS_STATE;previous_epoch=PREVIOUS_EPOCH;stream=None;current_path=None;pending=''
progress=session/'physical_observer_progress.json';last_progress=0
try:
 while time.monotonic()<deadline and len(rows)<90:
  if time.monotonic()-last_progress>=1:
   progress.write_text(json.dumps(dict(case=CASE,phase='waiting_for_key' if target_epoch is None else 'sampling',rows=len(rows),physical_epoch=target_epoch,updated_unix=time.time(),observer_read_only=True)))
   last_progress=time.monotonic()
  status=json.loads((session/'status.json').read_text())
  if status.get('state')!='running':interruption='runtime_not_running';break
  path=Path(status['detections_path'])
  if path!=current_path:
   if stream:stream.close()
   current_path=path;stream=path.open();pending=''
  part=stream.readline()
  if not part:time.sleep(.02);continue
  pending+=part
  if not pending.endswith(chr(10)):continue
  r=json.loads(pending);pending=''
  if r['sequence']<first:continue
  key=r.get('key_snapshot') or {};state=key.get('state',128);epoch=key.get('sequence');physical=key.get('source')=='physical_keys' and not state&64
  if not isinstance(epoch,int):interruption='key_snapshot_unavailable';break
  logical=state&63
  if target_epoch is None:
   if physical and logical==DESIRED and previous_state in EXPECTED and epoch!=previous_epoch and (epoch-previous_epoch)&255==1:
    target_epoch=epoch;deadline=time.monotonic()+SECONDS
   previous_state=logical;previous_epoch=epoch
  if target_epoch is None:continue
  if not physical or epoch!=target_epoch or logical!=DESIRED:
   interruption='key_or_remote_driver_changed_during_sample';break
  rows.append(r)
finally:
 if stream:stream.close()
progress.write_text(json.dumps(dict(case=CASE,phase='finished',rows=len(rows),physical_epoch=target_epoch,interruption=interruption,updated_unix=time.time(),observer_read_only=True)))
print(json.dumps(dict(rows=rows,physical_epoch=target_epoch,interruption=interruption)))
'''.replace('SESSION', repr(initial['session'])).replace('FIRST', repr(initial['status']['frames'])).replace('WAIT_KEY', repr(a.wait_key_seconds)).replace('SECONDS', repr(a.seconds)).replace('CASE',repr(a.case)).replace('PREVIOUS_STATE', repr(initial['status']['key_state'] & 63)).replace('PREVIOUS_EPOCH', repr(initial['status']['key_sequence'])).replace('DESIRED', repr(desired)).replace('EXPECTED', repr(expected_previous))
    data = json.loads(observer_ssh(code,a.wait_key_seconds+a.seconds+30))
    rows = data.pop('rows')
    scores = [score(case, row) for row in rows]
    with (out / 'observations.jsonl').open('w', encoding='utf8') as stream:
        for row, value in zip(rows, scores):
            stream.write(json.dumps(dict(observation=row, score=value)) + '\n')
    successes = sum(s['passed'] for s in scores)
    unsafe = [dict(sequence=r['sequence'], violations=s['unsafe']) for r, s in zip(rows, scores) if s['unsafe']]
    if fixture is not None:
        current = json.loads((ROOT / 'runs/basic_finish70_20261027/player_status.json').read_text())
        if current.get('state') != 'playing' or any(current.get(k) != fixture.get(k) for k in
                ['pid', 'video', 'start_frame', 'end_frame_exclusive', 'width', 'height', 'frequency']):
            data['interruption'] = 'PC_fixture_changed_during_sample'
    result = dict(case=a.case, frames=len(rows), successes=successes, ratio=successes / 90,
                  passed=len(rows) == 90 and successes / 90 >= .7 and not unsafe and not data['interruption'],
                  unsafe=unsafe, collector_read_only=True, physical_lane_association_verified=False,
                  manual_LED_buzzer_verification=False, folder=str(out), **data)
    (out / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result), flush=True)
    print('If the screen still requests driving, use KEY6 to park after observing the result.', flush=True)
    raise SystemExit(0 if result['passed'] else 2)


if __name__ == '__main__':
    main()
