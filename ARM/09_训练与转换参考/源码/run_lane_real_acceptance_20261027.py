"""Score fixed, actual HDMI->FPGA->capture->NPU->UART test cases.

Only video pixels and explicitly labelled test driver intent reach the ARM.
Expected line/action labels remain on this scoring computer.
"""
import argparse,hashlib,json,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
from deploy_arm_a06_20261027 import ssh
from deploy_lane_keys_20261027 import stop,start,REMOTE
from request_lane_driver_test_20261027 import request

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'runs/basic_finish70_20261027/lane_keys'
STANDARD=BASE/'acceptance_standard_v1.json'


def save(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')


def collect(sequence,state,count):
    code='''from pathlib import Path
import json,time
root=Path(REMOTE);session=Path((root/'last_session.txt').read_text());path=session/'detections.jsonl'
rows=[];deadline=time.monotonic()+55;interruption=None;began=False
evidence=session/('acceptance_key_'+str(SEQUENCE)+'.jsonl')
with path.open() as stream,evidence.open('w') as output:
 while len(rows)<COUNT and time.monotonic()<deadline:
  line=stream.readline()
  if not line:
   time.sleep(.02);continue
  row=json.loads(line);key=row.get('key_snapshot') or {}
  if began and key.get('sequence')!=SEQUENCE and key.get('source')=='physical_keys':
   interruption=dict(sequence=key.get('sequence'),state=key.get('state'),reason='physical_key_override');break
  if key.get('sequence')==SEQUENCE and key.get('state')==STATE:
   began=True
   output.write(line)
   compact={k:row[k] for k in ['sequence','identity','controls','key_snapshot','key_unchanged','capture_quality','boundaries','decision','context','feedback','profile','control_sequence_acknowledged_by_FPGA']}
   compact['receipt']={k:row['receipt'][k] for k in ['status_acknowledged','all_acknowledged','failure_reason']} if row.get('receipt') else None
   compact['key_snapshot']={k:v for k,v in key.items() if k!='query'}
   rows.append(compact)
print(json.dumps(dict(session=str(session),remote_full_evidence=str(evidence),rows=rows,expected_count=COUNT,interruption=interruption)))
'''.replace('REMOTE',repr(REMOTE)).replace('COUNT',repr(count)).replace('SEQUENCE',repr(sequence)).replace('STATE',repr(state))
    return json.loads(ssh(code))


def score(case,row):
    feedback=row.get('feedback') or {};decision=row['decision'];context=row['context']
    key=row['key_snapshot'];boundary=(row.get('boundaries') or {}).get(row['controls']['requested_turn'],{})
    checks=dict(
        input_valid=(row.get('capture_quality') or {}).get('valid',False),
        fresh=0<=row['profile']['final_age_ms']<=500,
        key_unchanged=row['key_unchanged'],
        arm_valid=decision['decision_valid'],arm_action=decision['action']==case['arm'],
        status_ack=bool(row.get('receipt') and row['receipt']['status_acknowledged']),
        feedback_valid=bool(feedback.get('decision_valid')),
        identity=(feedback.get('frame_id')==row['identity']['frame_id']&65535 and
                  feedback.get('camera_id')==0 and feedback.get('epoch')==key['sequence'] and
                  feedback.get('scene')==2 and feedback.get('reject_reason')==0 and not feedback.get('link_timeout',True)),
        fpga_action=feedback.get('action')==case['fpga_action'],alarm=feedback.get('alarm_code')==case['alarm'],
        epoch_ack=row['control_sequence_acknowledged_by_FPGA'])
    if case['line']:
        checks['line']=bool(boundary.get('state_valid') and boundary.get('simulation_association_valid') and boundary.get('line_type')==case['line'])
    unsafe=[]
    change=decision['action'] in ('CHANGE_LEFT','CHANGE_RIGHT')
    if change and (not boundary.get('state_valid') or boundary.get('line_type')!='dashed'):unsafe.append('change_without_current_dashed_boundary')
    if (change or feedback.get('action') in (5,6)) and not checks['fresh']:unsafe.append('expired_change_grant')
    if key['state']&16==0 and feedback.get('action') not in (None,0,1):unsafe.append('stop_override_failed')
    if feedback.get('action') in (2,5,6) and feedback.get('epoch')!=key['sequence']:unsafe.append('old_key_epoch_grant')
    return dict(passed=all(checks.values()),checks=checks,unsafe=unsafe)


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',action='append');args=p.parse_args()
    standard=json.loads(STANDARD.read_text());cases=[c for c in standard['cases'] if not args.case or c['id'] in args.case]
    if args.case:assert set(args.case)=={c['id'] for c in cases}
    folder=BASE/'acceptance'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC');folder.mkdir(parents=True)
    manifest=json.loads(ssh("from pathlib import Path;print((Path("+repr(REMOTE)+")/'manifest.json').read_text())"))
    save(folder/'arm_manifest.json',manifest)
    save(folder/'standard.json',standard);results=[]
    try:
        for case in cases:
            print(json.dumps(dict(event='case_start',case=case['id'])),flush=True)
            request('stop');stop()
            source=subprocess.run([sys.executable,'-X','utf8','-u',str(ROOT/'set_signal_video_source_20261027.py'),
                '--clip',case['clip'],'--start',str(case['start']),'--end',str(case['end']),'--refresh','60'],
                capture_output=True,text=True,encoding='utf8',timeout=30,check=True)
            start(SimpleNamespace(duration=14400,bench_boundary=True))
            receipt=request(case['intent']);key_state=receipt['status']['key_state'];sequence=receipt['status']['key_sequence']
            observed=collect(sequence,key_state,standard['frames_per_case']);rows=observed.pop('rows')
            scores=[score(case,row) for row in rows]
            with (folder/(case['id']+'.jsonl')).open('w',encoding='utf8') as f:
                for row,s in zip(rows,scores):f.write(json.dumps(dict(observation=row,score=s))+'\n')
            denominator=standard['frames_per_case'];success=sum(s['passed'] for s in scores)
            failures={k:sum(not s['checks'].get(k,False) for s in scores) for k in scores[0]['checks']} if scores else {}
            unsafe=[dict(frame=row['identity']['frame_id'],violations=s['unsafe']) for row,s in zip(rows,scores) if s['unsafe']]
            gate=receipt['receipt'].get('immediate_gate_feedback') or {}
            gate_safe=gate.get('action')==1 and not gate.get('decision_valid',True)
            result=dict(case=case,**observed,frames=len(rows),successes=success,ratio=success/denominator,
                passed=len(rows)==denominator and success/denominator>=standard['minimum_success_ratio'] and not unsafe and gate_safe,
                failed_check_counts=failures,unsafe_events=unsafe,key_epoch_immediate_stop=gate_safe,driver_receipt=receipt,
                actual_source=json.loads(source.stdout),software_driver_is_not_physical_button_press=True)
            save(folder/(case['id']+'_summary.json'),result);results.append(result)
            save(folder/'summary.json',dict(completed_cases=results,all_requested_passed=all(r['passed'] for r in results),
                all_mandatory_tested=set(standard['mandatory_cases']).issubset(r['case']['id'] for r in results),
                physical_lane_association_verified=False,standard_sha256=hashlib.sha256(STANDARD.read_bytes()).hexdigest()))
            print(json.dumps(dict(event='case_result',case=case['id'],frames=len(rows),successes=success,ratio=result['ratio'],passed=result['passed'],failures=failures,unsafe=len(unsafe),folder=str(folder))),flush=True)
            if result.get('interruption'):raise RuntimeError('Physical key override interrupted automatic bench test; result retained')
    except BaseException as exc:
        save(folder/'error.json',dict(error=repr(exc),completed_cases=len(results)))
        raise
    finally:
        try:save(folder/'parked_exit.json',request('exit'))
        except Exception as exc:save(folder/'parked_exit_error.json',dict(error=repr(exc)));raise


if __name__=='__main__':main()
