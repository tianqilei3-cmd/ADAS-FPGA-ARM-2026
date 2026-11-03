"""Labelled driver intent only; no lamp color or expected model output sent."""
import argparse,json,time,uuid
from deploy_unified_arm_20261028 import REMOTE,ssh
STATES={'signal_stop':0,'signal':0x10,'signal_force':0x30,'lane_stop':4,'left_stop':5,'right_stop':6,
    'straight':0x14,'left':0x15,'right':0x16,'force_left':0x35,'force_right':0x36,'risk_stop':12,'risk':0x1c,'exit':255}


def request(name):
    state=STATES[name];uid=uuid.uuid4().hex
    code='''from pathlib import Path
import json,time
root=Path(REMOTE);data=dict(uid=UID,state=STATE,created_unix=time.time(),deadline_unix=time.time()+25)
p=root/'remote_driver_request.tmp';p.write_text(json.dumps(data));p.replace(root/'remote_driver_request.json')
deadline=time.monotonic()+28
while time.monotonic()<deadline:
 r=root/'remote_driver_receipt.json';s=root/'last_session.txt'
 if r.exists() and s.exists():
  receipt=json.loads(r.read_text());p=Path(s.read_text())/'status.json'
  status=json.loads(p.read_text()) if p.exists() else {}
  if status.get('state') in ('failed','stopped') and status.get('updated_unix',0)>=data['created_unix']:raise RuntimeError('Runtime not active: '+str(status))
  if receipt.get('uid')==UID and not receipt.get('acknowledged'):raise RuntimeError(str(receipt))
  if receipt.get('uid')==UID and status.get('key_state')==EXPECTED:
   print(json.dumps(dict(receipt=receipt,status=status)));break
 time.sleep(.1)
else:raise TimeoutError('Test driver intent not consumed; no scene answer supplied')
'''.replace('REMOTE',repr(REMOTE)).replace('UID',repr(uid)).replace('STATE',repr(state)).replace('EXPECTED',repr(4 if state==255 else state|64))
    result=json.loads(ssh(code));return dict(name=name,state=state,**result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('intent',choices=STATES);args=p.parse_args();print(json.dumps(request(args.intent)),flush=True)
