"""Recovery launcher: bounded settling while parked;24 actual valid frames per mode."""
import argparse,json,os,sys,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm-dir',type=Path,default=Path('/home/orangepi/adas2026/signal_frozen_20261028'));p.add_argument('--check-link-only',action='store_true');a=p.parse_args()
    assert os.name=='posix','Run this preflight on Orange Pi Linux'
    root=a.arm_dir.resolve();assert root.parent==Path('/home/orangepi/adas2026').resolve() and root.name in ('signal_frozen_20261028','unified_20261028','signal_recovered_20261028')
    sys.path.insert(0,str(root))
    from uart_basic70_confirmed import ConfirmedUart
    from basic70_wire import packet
    from latest_live_capture_20261027 import LatestCaptureReader
    from capture_quality import inspect_native_patches
    uart=ConfirmedUart('/dev/ttyUSB0');stages=[];reader=None
    try:
        assert uart.latest_feedback['build_tag']==41,'Requires FPGA profile0x29'
        assert uart.exchange(packet(0x14,[0x54,255]))['acknowledged']
        for name,mode,roi in [('legacy_tiles',0,[0,0,.5,.5]),('full_raw',7,[0,0,1,1])]:
            assert uart.exchange(packet(0,[mode]))['acknowledged']
            reader=LatestCaptureReader('/dev/video0',width=1280,height=720,fps=10,roi=roi,read_timeout=3.)
            tick=time.monotonic();valid=0;qualities=[];timeouts=[]
            while valid<24 and time.monotonic()-tick<25:
                try:frame=reader.read()
                except TimeoutError as error:
                    valid=0;timeouts.append(dict(seconds=time.monotonic()-tick,error=repr(error)))
                    continue
                quality=inspect_native_patches(frame.image)
                if quality['valid'] and 0<=time.monotonic()-frame.decoded_at<=.5:valid+=1
                else:valid=0
                qualities.append(quality)
            assert valid>=24,'No24consecutive fresh valid frames at '+name
            stages.append(dict(stage=name,seconds=time.monotonic()-tick,valid_frames=valid,settling_timeouts=timeouts,negotiated=next(v for v in reader.info if v.get('event')=='opened'),last_quality=qualities[-1]))
            reader.close();reader=None
        assert uart.exchange(packet(0x14,[0x54,255]))['acknowledged']
        feedback,_=uart.query_state();assert feedback['action']==1 and not feedback['decision_valid']
        report=dict(link_preflight_passed=True,stages=stages,park_feedback=feedback,cold_power_cycle_proven_by_this_test=False)
        (root/'RECOVERY_LINK_BOOTSTRAP.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    finally:
        if reader:reader.close()
        try:uart.exchange(packet(0x14,[0x54,255]))
        finally:uart.close()
    if not a.check_link_only:
        os.chdir(root);args=[sys.executable,str(root/'run_unified_demo.py'),'--duration','21600','--capture-mode','720','--capture-fps','10','--lamp-profile','capture_ft','--road-profile','int8','--classifiers','batch2','--video-layout','full','--windowed','--housing-refinement','--proposal-mode','geometry']
        os.execve(sys.executable,args,dict(os.environ,DISPLAY=':0',XAUTHORITY='/home/orangepi/.Xauthority',XDG_RUNTIME_DIR='/run/user/1000',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))


if __name__=='__main__':main()
