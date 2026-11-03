"""FP batch2 graphs; reuse original weights, keep batch3 conversion untouched."""
import json
from deploy_unified_arm_20261028 import ROOT,ssh,stop
from prepare_batched_classifiers_20261028 import SOURCE
OUT=ROOT/'runs/basic_finish70_20261028/batch2_classifiers'
REMOTE='/home/orangepi/adas2026/batch2_classifiers_20261028'

if __name__=='__main__':
    OUT.mkdir(exist_ok=True);stop()
    source=SOURCE.replace('[[3,3,128,128]]','[[2,3,128,128]]').replace('_batch3_fp','_batch2_fp').replace('batch=3','batch=2')
    print(ssh("from pathlib import Path;r=Path("+repr(REMOTE)+");assert not r.exists();r.mkdir();(r/'convert.py').write_text("+repr(source)+");print('prepared')"),flush=True)
    code='''from pathlib import Path
import subprocess,os
root=Path(REMOTE)
p=subprocess.run(['python3','-u',str(root/'convert.py')],cwd=root,capture_output=True,text=True,timeout=75,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
(root/'conversion.log').write_text(p.stdout+chr(10)+p.stderr);assert p.returncode==0,p.stdout+p.stderr
print((root/'conversion.json').read_text())
'''.replace('REMOTE',repr(REMOTE))
    data=json.loads(ssh(code));(OUT/'conversion.json').write_text(json.dumps(data,indent=2),encoding='utf8');print(json.dumps(data),flush=True)
