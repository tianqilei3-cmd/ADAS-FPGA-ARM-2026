"""Frozen real-pixel acceptance; expected answers stay on this scorer only."""
import argparse,hashlib,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
from deploy_unified_arm_20261028 import ROOT,REMOTE,OUT,stop,start,ssh
from request_unified_driver_20261028 import request
from run_lane_real_acceptance_20261027 import score as score_lane
BASE=ROOT/'runs/basic_finish70_20261028'


def standard():
    p=BASE/'unified_acceptance_v1.json'
    if p.exists():return json.loads(p.read_text())
    old=json.loads((ROOT/'runs/basic_finish70_20261027/s3/core_signal_acceptance_v1.json').read_text())
    cases=[]
    for c in old['cases']:
        cases.append(dict(id=c['id'],kind='signal',clip=c['clip'],start=c['start_frame'],end=c['end_frame_exclusive'],
            intent='signal_force' if c['driver_force'] else 'signal',arm=c['arm_action'],fpga_action=c['fpga_action'],alarm=c['fpga_alarm'],
            light={1:'red',3:'green'}[c['light']],line=None,frames=90))
    lanes=json.loads((ROOT/'runs/basic_finish70_20261027/lane_keys/acceptance_standard_v1.json').read_text())
    cases.extend(dict(c,kind='boundary',frames=60) for c in lanes['cases'])
    value=dict(scope='Actual HDMI-FPGA-capture-NPU-UART, virtual driver-simulated lane; no physical calibration claim',
        minimum_success_ratio=.7,maximum_age_ms=500,minimum_green_confirmations=3,cases=cases,
        mandatory_signal_cases=[c['id'] for c in cases if c['kind']=='signal'],mandatory_lane_cases=lanes['mandatory_cases'],
        semantic_rules_retained_from=str(ROOT/'runs/basic_finish70_20261027/s3/core_signal_acceptance_v1.json'))
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8');return value


def collect(sequence,state,count):
    code='''from pathlib import Path
import json,time
root=Path(REMOTE);session=Path((root/'last_session.txt').read_text());rows=[];began=False;interruption=None;pending=''
deadline=time.monotonic()+75;evidence=session/('acceptance_key_'+str(SEQUENCE)+'.jsonl')
with (session/'detections.jsonl').open() as stream,evidence.open('w') as output:
 while len(rows)<COUNT and time.monotonic()<deadline:
  line=stream.readline()
  if not line:time.sleep(.02);continue
  pending+=line
  if not pending.endswith(chr(10)):continue
  line=pending;pending='';row=json.loads(line);key=row.get('key_snapshot') or {}
  if began and key.get('sequence')!=SEQUENCE and key.get('source')=='physical_keys':
   interruption=dict(reason='physical_key_or_recovery_override',state=key.get('state'),sequence=key.get('sequence'));break
  if key.get('sequence')!=SEQUENCE or key.get('state')!=STATE:continue
  began=True;output.write(line)
  compact={k:row[k] for k in ['sequence','identity','controls','key_snapshot','key_unchanged','capture_quality','boundaries','decision','context','feedback','profile','control_sequence_acknowledged_by_FPGA','signal_selection','signal_metrics','capture_recoveries']}
  compact['receipt']={k:row['receipt'][k] for k in ['status_acknowledged','all_acknowledged','failure_reason']} if row.get('receipt') else None
  compact['key_snapshot']={k:v for k,v in key.items() if k!='query'}
  rows.append(compact)
print(json.dumps(dict(session=str(session),remote_full_evidence=str(evidence),rows=rows,interruption=interruption)))
'''.replace('REMOTE',repr(REMOTE)).replace('SEQUENCE',repr(sequence)).replace('STATE',repr(state)).replace('COUNT',repr(count))
    return json.loads(ssh(code))


def score_signal(case,row):
    feedback=row.get('feedback') or {};key=row['key_snapshot'];selected=row['signal_selection'];d=row['decision']
    matched=feedback.get('frame_id')==(row['identity']['frame_id']&65535) and feedback.get('camera_id')==0 and feedback.get('epoch')==key['sequence'] and feedback.get('scene')==1
    checks=dict(input_valid=row['capture_quality']['valid'],fresh=0<=row['profile']['final_age_ms']<=500,key_unchanged=row['key_unchanged'],
        arm_valid=d['decision_valid'],arm_action=d['action']==case['arm'],light=bool(selected['selected_valid'] and selected['simulation_association_valid'] and selected['selected_state']==case['light']),
        status_ack=bool(row.get('receipt') and row['receipt']['status_acknowledged']),identity=matched and feedback.get('reject_reason')==0 and not feedback.get('link_timeout',True),
        feedback_valid=feedback.get('decision_valid',False),fpga_action=feedback.get('action')==case['fpga_action'],alarm=feedback.get('alarm_code')==case['alarm'],epoch_ack=row['control_sequence_acknowledged_by_FPGA'])
    grant=d['action']=='GO' or (matched and feedback.get('decision_valid') and feedback.get('action') in [2,3,4,5,6])
    unsafe=[]
    if grant and selected['selected_state']=='red':unsafe.append('red_grant')
    if grant and not(selected['selected_valid'] and selected['simulation_association_valid'] and selected['selected_state']=='green'):unsafe.append('unknown_or_non_green_grant')
    if grant and not checks['fresh']:unsafe.append('stale_grant')
    if grant and not row['key_unchanged']:unsafe.append('unconfirmed_driver_grant')
    return dict(passed=all(checks.values()),checks=checks,unsafe=unsafe)


def main():
    p=argparse.ArgumentParser();p.add_argument('--group',choices=['signal','boundary','all'],default='signal');p.add_argument('--case',action='append')
    p.add_argument('--capture-mode',choices=['720','1080'],default='720')
    p.add_argument('--capture-fps',type=int,choices=[10,20,30],default=30)
    p.add_argument('--lamp-profile',choices=['retained','capture_ft'],default='retained')
    p.add_argument('--road-profile',choices=['decoded','sparse','int8'],default='decoded')
    p.add_argument('--classifiers',choices=['npu','batch2','batch3','cpu_process'],default='npu')
    p.add_argument('--video-layout',choices=['tiles','full'],default='tiles');p.add_argument('--windowed',action='store_true');p.add_argument('--housing-refinement',action='store_true')
    p.add_argument('--proposal-mode',choices=['neural','geometry'],default='neural');a=p.parse_args();std=standard()
    cases=[c for c in std['cases'] if (a.group=='all' or c['kind']==a.group) and (not a.case or c['id'] in a.case)]
    if a.case:assert set(a.case)=={c['id'] for c in cases}
    assert cases
    folder=BASE/'acceptance'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC');folder.mkdir(parents=True)
    def save(name,data):(folder/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
    save('standard.json',std)
    save('arm_manifest.json',json.loads(ssh("from pathlib import Path;print((Path("+repr(REMOTE)+")/'manifest.json').read_text())")))
    reports=[]
    try:
        # Conversion may have deliberately stopped inference. Establish the
        # runtime owner before asking it to apply a parked driver intent.
        start(21600,a.capture_mode,False,a.lamp_profile,a.road_profile,a.classifiers,a.video_layout,a.windowed,a.housing_refinement,a.proposal_mode,a.capture_fps)
        for case in cases:
            print(json.dumps(dict(event='case_start',case=case['id'],capture_mode=a.capture_mode,folder=str(folder))),flush=True)
            request('exit');stop()
            src=subprocess.run([sys.executable,'-X','utf8','-u',str(ROOT/'set_signal_video_source_20261027.py'),'--clip',case['clip'],'--start',str(case['start']),'--end',str(case['end']),'--refresh','60'],capture_output=True,text=True,encoding='utf8',timeout=30,check=True)
            start(21600,a.capture_mode,False,a.lamp_profile,a.road_profile,a.classifiers,a.video_layout,a.windowed,a.housing_refinement,a.proposal_mode,a.capture_fps);intent=request(case['intent']);sequence=intent['status']['key_sequence'];state=intent['status']['key_state']
            observed=collect(sequence,state,case['frames']);rows=observed.pop('rows');scores=[score_signal(case,r) if case['kind']=='signal' else score_lane(case,r) for r in rows]
            with (folder/(case['id']+'.jsonl')).open('w',encoding='utf8') as f:
                for row,s in zip(rows,scores):f.write(json.dumps(dict(observation=row,score=s))+'\n')
            success=sum(s['passed'] for s in scores);unsafe=[dict(frame=r['identity']['frame_id'],violations=s['unsafe']) for r,s in zip(rows,scores) if s['unsafe']]
            gate=intent['receipt'].get('immediate_gate_feedback') or {};gate_safe=gate.get('action')==1 and not gate.get('decision_valid',True)
            result=dict(case=case,**observed,frames=len(rows),successes=success,ratio=success/case['frames'],
                passed=len(rows)==case['frames'] and success/case['frames']>=.7 and gate_safe and not unsafe,
                failed_check_counts={k:sum(not s['checks'].get(k,False) for s in scores) for k in scores[0]['checks']} if scores else {},
                unsafe_events=unsafe,key_epoch_immediate_stop=gate_safe,driver_receipt=intent,actual_source=json.loads(src.stdout),capture_mode=a.capture_mode,capture_fps=a.capture_fps,lamp_profile=a.lamp_profile,road_profile=a.road_profile,classifiers=a.classifiers,video_layout=a.video_layout,windowed=a.windowed,housing_refinement=a.housing_refinement,proposal_mode=a.proposal_mode)
            save(case['id']+'_summary.json',result);reports.append(result)
            save('summary.json',dict(reports=reports,all_requested_passed=all(r['passed'] for r in reports),all_signal_tested=set(std['mandatory_signal_cases']).issubset(r['case']['id'] for r in reports),all_lane_tested=set(std['mandatory_lane_cases']).issubset(r['case']['id'] for r in reports),physical_lane_association_verified=False))
            print(json.dumps(dict(event='case_result',case=case['id'],frames=len(rows),successes=success,ratio=result['ratio'],passed=result['passed'],failures=result['failed_check_counts'],unsafe=len(unsafe),folder=str(folder))),flush=True)
            if result['interruption']:raise RuntimeError('Driver state overridden; incomplete test retained')
    except BaseException as exc:save('error.json',dict(error=repr(exc)));raise
    finally:
        try:save('parked_exit.json',request('exit'))
        except Exception as exc:save('parked_exit_error.json',dict(error=repr(exc)))


if __name__=='__main__':main()
