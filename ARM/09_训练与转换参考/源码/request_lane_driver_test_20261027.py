"""Request labelled boundary-test driver intent; no image/classification input."""
import argparse,json,time,uuid
from pathlib import Path
from deploy_arm_a06_20261027 import ssh
REMOTE='/home/orangepi/adas2026/lane_keys_20261027'
STATES={'stop':4,'left_stop':5,'right_stop':6,'straight':0x14,'left':0x15,'right':0x16,
        'force_left':0x35,'force_right':0x36,'exit':255}
def request(name):
    state=STATES[name];uid=uuid.uuid4().hex
    code='''from pathlib import Path
import json,time
root=Path(REMOTE);data=dict(uid=UID,state=STATE,deadline_unix=time.time()+15)
p=root/'remote_driver_request.tmp';p.write_text(json.dumps(data));p.replace(root/'remote_driver_request.json')
print(json.dumps(data))
'''.replace('REMOTE',repr(REMOTE)).replace('UID',repr(uid)).replace('STATE',repr(state))
    wanted=json.loads(ssh(code))
    for _ in range(15):
        code='''from pathlib import Path
import json
r=Path(REMOTE);p=r/'remote_driver_receipt.json';s=Path((r/'last_session.txt').read_text())/'status.json'
print(json.dumps(dict(receipt=json.loads(p.read_text()) if p.exists() else None,status=json.loads(s.read_text()) if s.exists() else {})))
'''.replace('REMOTE',repr(REMOTE))
        result=json.loads(ssh(code))
        expected=(4 if state==255 else state|64)
        if result.get('receipt') and result['receipt'].get('uid')==uid and result.get('status',{}).get('key_state')==expected:
            return dict(name=name,request=wanted,**result)
        time.sleep(.3)
    raise TimeoutError('Driver test command not consumed')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('intent',choices=list(STATES));a=p.parse_args()
    print(json.dumps(request(a.intent),ensure_ascii=False),flush=True)
