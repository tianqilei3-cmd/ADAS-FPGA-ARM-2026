"""Physical-capture signal/lane/risk prototype with bounded recovery and logs."""
import argparse,json,math,os,signal,time,shutil,threading
from dataclasses import asdict,replace
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2,numpy as np
from vision_common import preprocess,decode,decoded_output,draw,write_image
from latest_live_capture_20261027 import LatestCaptureReader
from arm_scene_a04 import frame_identity
from competition_scene_logic import Controls,SceneController
from road_hazard_guard import HazardStopController,hazard_observation
from physical_keys import KeyTracker
from boundary_runtime import BoundaryRuntime
from boundary_cpu_worker import BoundaryCPU
from capture_quality import prepare as prepare_capture,inspect_native_patches
from signal_engine import SignalEngine,unknown_signal
from npu_signal_attributes import NpuColorGuard
from basic70_wire import packets,packet
from ack_protocol import exchange_frame
from lane_support import atomic,snapshot_json,save_evidence
from unified_policy import choose,context,legal_driver_state

ROOT=Path(__file__).resolve().parent
WINDOW='ADAS 2026 - signal / lane / physical keys'


def panel(image,roads,paths,lights,controls,selected,boundaries,decision,key,receipt,feedback,age):
    canvas=cv2.resize(draw(image,[r for r in roads if r['name']!='traffic_light']),(1280,720),interpolation=cv2.INTER_NEAREST)
    scale=np.asarray([1280/image.shape[1],720/image.shape[0]],np.float32)
    for path in paths:
        points=np.asarray(path['points_native'],np.float32)
        if len(points)<2:continue
        cuts=np.flatnonzero(np.diff(points[:,1])>image.shape[0]/288*1.5)+1
        strokes=[part.reshape(-1,1,2) for part in np.split(np.rint(points*scale).astype(np.int32),cuts) if len(part)>1]
        if strokes:cv2.polylines(canvas,strokes,False,(255,210,30) if path['line_type']=='dashed' else (30,200,255),3)
    for lamp in lights:
        a,b,c,d=np.rint(np.asarray(lamp['box'])*np.tile(scale,2)).astype(int)
        color={'red':(30,40,255),'green':(40,240,60),'yellow':(20,220,250)}.get(lamp['name'],(150,150,150))
        cv2.rectangle(canvas,(a,b),(c,d),color,2)
    cv2.rectangle(canvas,(0,0),(1280,130),(22,26,30),-1);cv2.rectangle(canvas,(0,620),(1280,720),(22,26,30),-1)
    color=(50,80,255) if decision['alarm'] else (60,230,90) if decision['action'] in ('GO','CHANGE_LEFT','CHANGE_RIGHT') else (90,190,250)
    source='REMOTE TEST DRIVER' if key.get('source')=='remote_test_driver' else 'PHYSICAL KEYS'
    lines=[f"ARM {decision['action']} | ALARM {int(decision['alarm'])} | {decision['reason']}",
        f"{source} seq={key.get('sequence','?')} | scene={controls.mode} | {controls.movement} | {controls.requested_motion} | force={int(controls.forced)}",
        f"LAMP {selected.get('selected_state','unknown')} | {selected.get('reason','not_active')} | age {age:.0f}ms"]
    if controls.mode=='boundary':lines[2]=f"IMAGE CENTER left={(boundaries or {}).get('left',{}).get('line_type')} right={(boundaries or {}).get('right',{}).get('line_type')} | age {age:.0f}ms"
    for i,line in enumerate(lines):cv2.putText(canvas,line[:145],(12,30+i*35),0,.58 if i==0 else .51,color if i==0 else (240,240,240),2 if i==0 else 1,cv2.LINE_AA)
    confirmed=receipt and receipt['status_acknowledged']
    state=f"action={feedback['action']} alarm={feedback['alarm_code']} fid={feedback['frame_id']} epoch={feedback['epoch']}" if feedback else 'current state unconfirmed'
    cv2.putText(canvas,('UART CONFIRMED' if confirmed else 'UART UNCONFIRMED / STOP')+' | FPGA '+state,(12,646),0,.56,(60,230,90) if confirmed else (60,80,250),1)
    cv2.putText(canvas,'KEY2 scene | KEY3 left | KEY4 straight | KEY5 right | KEY6 drive/stop | KEY7 force',(12,676),0,.54,(240,240,240),1)
    cv2.putText(canvas,'Virtual lane only. Physical lane calibration, side clearance and vehicle actuation unverified.',(12,705),0,.5,(210,210,210),1)
    return canvas


def stopped_canvas(reason):
    image=np.full((720,1280,3),(25,25,100),np.uint8)
    cv2.putText(image,'STOP | INPUT / INFERENCE NOT CURRENT',(30,280),0,1.,(255,255,255),2)
    cv2.putText(image,str(reason)[:120],(30,330),0,.6,(240,240,240),1)
    return image


def main():
    from uart_basic70_confirmed import ConfirmedUart
    p=argparse.ArgumentParser();p.add_argument('--duration',type=float,default=21600);p.add_argument('--frames',type=int,default=0)
    p.add_argument('--headless',action='store_true');p.add_argument('--capture-mode',choices=['720','1080'],default='720')
    p.add_argument('--capture-fps',type=int,choices=[10,20,30],default=30)
    p.add_argument('--lamp-profile',choices=['retained','capture_ft'],default='retained')
    p.add_argument('--road-profile',choices=['decoded','sparse','int8'],default='decoded')
    p.add_argument('--classifiers',choices=['npu','batch2','batch3','cpu_process'],default='npu')
    p.add_argument('--video-layout',choices=['tiles','full'],default='tiles');p.add_argument('--windowed',action='store_true')
    p.add_argument('--housing-refinement',action='store_true')
    p.add_argument('--proposal-mode',choices=['neural','geometry'],default='neural')
    p.add_argument('--uart',default='/dev/ttyUSB0');args=p.parse_args();cv2.setNumThreads(1)
    stop=threading.Event();lock=threading.Lock();shared=dict(canvas=stopped_canvas('Starting models'),captured=None,failed=False)
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.set())
    session=ROOT/'results'/time.strftime('%Y%m%d_%H%M%S');session.mkdir(parents=True);(ROOT/'last_session.txt').write_text(str(session))
    def publish(canvas,captured=None):
        with lock:shared.update(canvas=canvas,captured=captured)
    def worker():
        reader=road=boundary=boundary_cpu=uart=lamp=guard=stream=None
        pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='one_evidence_job');evidence_future=None
        count=0;errors=[];recoveries=0;evidence=set();started=time.monotonic();last_open=0.;wire_offset=None;log_index=0;must_park=False;video_initialized=False
        tracker=KeyTracker();controller=SceneController(max_age_ms=500.,green_confirmations=3);hazard=HazardStopController(max_age_ms=500.)
        controls=Controls(requested_motion='stop');key={};inbox=ROOT/'remote_driver_request.json';driver_receipt=ROOT/'remote_driver_receipt.json'
        last_request=json.loads(driver_receipt.read_text()).get('uid') if driver_receipt.exists() else None
        def note(phase,exc):
            errors.append(dict(time_unix=time.time(),phase=phase,error=repr(exc)))
            if len(errors)>200:del errors[:-200]
        def open_capture():
            return LatestCaptureReader('/dev/video0',width=1920 if args.capture_mode=='1080' else 1280,
                height=1080 if args.capture_mode=='1080' else 720,fps=args.capture_fps,roi=[0,0,1,1] if args.video_layout=='full' else [0,0,.5,.5],read_timeout=3.)
        try:
            from rknnlite.api import RKNNLite
            road_name={'sparse':'road_heads_fp.rknn','decoded':'road_rect_fp.rknn','int8':'road_int8_heads.rknn'}[args.road_profile]
            road=RKNNLite(verbose=False);assert road.load_rknn(str(ROOT/'models'/road_name))==0;assert road.init_runtime()==0
            road.inference(inputs=[np.zeros((1,384,640,3),np.uint8)],data_format=['nhwc'])
            if args.classifiers=='cpu_process':guard=None
            elif args.classifiers in ('batch2','batch3'):
                from batched_signal_attributes import BatchedColorGuard
                guard=BatchedColorGuard(ROOT/('models/color_guard_'+args.classifiers+'_fp.rknn'),batch_size=int(args.classifiers[-1]))
            else:guard=NpuColorGuard(ROOT/'models/color_guard_fp.rknn')
            meta=json.loads((ROOT/'light_metadata.json').read_text())
            model_name='lamp_aligned_fp_heads.rknn' if args.lamp_profile=='capture_ft' else 'lamp_rect_fp_heads.rknn'
            lamp=SignalEngine(ROOT,guard,dict(meta,image_size=[256,640],model_name=model_name,classifiers=args.classifiers,housing_refinement=args.housing_refinement,proposal_mode=args.proposal_mode))
            reader=open_capture();stream=(session/'detections.jsonl').open('w',buffering=1)
            while not stop.is_set() and time.monotonic()-started<args.duration and (not args.frames or count<args.frames):
                free=shutil.disk_usage(session).free
                if free<512*1024**2:raise OSError('Disk guard512MiB')
                if stream.tell()>64*1024**2:
                    stream.close();log_index+=1;stream=(session/f'detections_{log_index:04d}.jsonl').open('w',buffering=1)
                key_ok=False
                try:
                    if uart is None and time.monotonic()-last_open>1:
                        last_open=time.monotonic();uart=ConfirmedUart(args.uart)
                        if uart.latest_feedback['capabilities']&0x60!=0x60:raise OSError('Missing key epoch / remote intent profile')
                        if args.video_layout=='full' and uart.latest_feedback['build_tag']!=0x29:raise OSError('Full resolution capture requires FPGA profile0x29')
                        if uart.latest_feedback['build_tag']==0x29:
                            mode_ack=uart.exchange(packet(0,[7 if args.video_layout=='full' else 0]))
                            if not mode_ack['acknowledged']:raise OSError('Video layout command not acknowledged')
                            # A serial ACK timeout does not change video layout.
                            # Flush only initial layout or a detected FPGA reset.
                            if not video_initialized or not uart.latest_feedback['seen_extended']:
                                reader.close();reader=open_capture();controller.reset();lamp.controller.reset()
                            video_initialized=True
                        if wire_offset is None:wire_offset=uart.latest_feedback['frame_id']+1-count
                    if uart:
                        if inbox.exists():
                            request=json.loads(inbox.read_text())
                            if request.get('uid')!=last_request:
                                if not legal_driver_state(request.get('state')) or not time.time()<=request.get('deadline_unix',0)<=time.time()+30:
                                    last_request=request.get('uid');atomic(driver_receipt,dict(uid=last_request,acknowledged=False,error='invalid_or_expired_intent'))
                                else:
                                    tx=uart.exchange(packet(0x14,[0x54,request['state']]))
                                    if not tx['acknowledged']:raise OSError('Remote driver intent not acknowledged')
                                    gate,_=uart.query_state();last_request=request['uid']
                                    atomic(driver_receipt,dict(uid=last_request,state=request['state'],acknowledged=True,transaction=tx,immediate_gate_feedback=gate))
                        key=tracker.update(uart.query_keys());controls=key['controls'];key_ok=True
                        if must_park:
                            if controls.requested_motion=='stop':must_park=False
                            else:
                                uart.exchange(packet(0x14,[0x54,255]));key_ok=False
                                controls=replace(controls,requested_motion='stop',forced=False)
                        if key['changed']:
                            controller.reset();lamp.controller.reset()
                            with (session/'key_events.jsonl').open('a') as f:f.write(json.dumps(dict(time_unix=time.time(),**snapshot_json(key)))+'\n')
                except (OSError,ValueError) as exc:
                    note('key_before_image',exc)
                    if uart:uart.close();uart=None
                    controls=replace(controls,requested_motion='stop',forced=False)
                if wire_offset is None:
                    stop.wait(.1);continue
                try:frame=reader.read()
                except (TimeoutError,RuntimeError) as exc:
                    note('capture_recovery',exc);recoveries+=1;must_park=True;publish(stopped_canvas('Capture recovery; fresh physical GO required'))
                    input_event=dict(time_unix=time.time(),detected_monotonic=time.monotonic(),error=repr(exc),recovery=recoveries,park_transaction=None,park_feedback=None)
                    if uart:
                        try:
                            input_event['park_transaction']=uart.exchange(packet(0x14,[0x54,255]))
                            input_event['park_feedback'],input_event['park_query']=uart.query_state()
                        except (OSError,ValueError) as park_error:note('capture_park',park_error);uart.close();uart=None
                    with (session/'input_events.jsonl').open('a') as f:f.write(json.dumps(input_event)+'\n')
                    controls=replace(controls,requested_motion='stop',forced=False);controller.reset();lamp.controller.reset()
                    reader.close();stop.wait(.5)
                    if stop.is_set():break
                    reader=open_capture()
                    # Retain hazard tracks across recovery; missing images cannot clear them.
                    continue
                raw_image=frame.image;captured=frame.decoded_at
                capture_decode_age_ms=(time.monotonic()-captured)*1000;quality_tick=time.monotonic()
                if key.get('scene_index',0)==0:filtered=raw_image;quality=inspect_native_patches(raw_image)
                else:filtered,quality=prepare_capture(raw_image)
                quality_ms=(time.monotonic()-quality_tick)*1000
                image=filtered if controls.mode=='boundary' else raw_image
                frame_counter=count+(wire_offset or 0);identity=frame_identity(image,'/dev/video0',0,frame_counter,captured)
                identity['camera_id']='capture:/dev/video0:host-session'
                mode=key.get('scene_index',0);job=None
                if mode==0:job=lamp.begin(image,identity,controls)
                if mode==1:
                    if boundary is None:boundary=BoundaryRuntime(ROOT/'models/boundary_fp.rknn');boundary_cpu=BoundaryCPU()
                    logits,boundary_infer_ms=boundary.infer(image);token=(identity['frame_id'],identity['image_identity']);boundary_cpu.submit(logits,token)
                tick=time.monotonic();rgb,g=preprocess(image,[384,640]);raw=road.inference(inputs=[rgb[None]],data_format=['nhwc'])
                roads=decode(decoded_output(raw,[384,640],4,score_threshold=.25),g,['traffic_light','vehicle','two_wheeler','person'],.25,.45,100)
                road_ms=(time.monotonic()-tick)*1000
                result=dict(boundaries=None,paths=[],model_ms=0.,postprocess_ms=0.)
                if mode==1:
                    classes,normalize_ms=boundary_cpu.take(token)
                    result=boundary.finish_classes(classes,image,identity,roads,boundary_infer_ms+normalize_ms)
                selected=unknown_signal(identity,controls,'signal_not_active');lights=[];signal_metrics={}
                if job:lights,selected,signal_metrics=lamp.finish(job,image,roads,identity,controls)
                observation=hazard_observation(roads,identity,image.shape);observation['observation_valid']=quality['valid']
                risk=hazard.update(identity,observation,controls,now_ms=time.monotonic()*1000)
                decision=choose(identity,controls,key,result['boundaries'],risk,selected,lamp.controller if mode==0 else controller,time.monotonic()*1000)
                if not quality['valid']:decision.update(action='STOP',decision_valid=False,reason='invalid_capture:'+quality['reason'])
                unchanged=key_ok
                if uart:
                    try:
                        final=tracker.update(uart.query_keys());unchanged=key_ok and final['state']==key['state'] and final['sequence']==key['sequence']
                        if final['changed']:controller.reset();lamp.controller.reset()
                        if not unchanged:decision.update(action='STOP',decision_valid=False,reason='key_changed_wait_for_current_image')
                        key=final
                    except (OSError,ValueError) as exc:
                        note('key_before_tx',exc);unchanged=False;uart.close();uart=None
                if not unchanged:decision.update(action='STOP',decision_valid=False,reason='driver_key_unconfirmed_or_changed')
                current_context=context(controls,key,result['boundaries'],selected,decision,observation,risk,captured,unchanged,quality)
                receipt=feedback=None;tx_tick=time.monotonic()
                if uart:
                    receipt=exchange_frame(uart,packets(frame_counter&65535,roads,image.shape[1],image.shape[0],current_context,max_boxes=1),captured)
                    feedback=next((x.get('feedback') for x in receipt['transactions'] if x.get('feedback')),None)
                    if receipt['reopen_required']:uart.close();uart=None;last_open=time.monotonic()
                uart_ms=(time.monotonic()-tx_tick)*1000;age=(time.monotonic()-captured)*1000
                age_guard=next((t['transport_age_guard'] for t in (receipt or {}).get('transactions',[]) if t.get('transport_age_guard')),None)
                if age_guard:
                    current_context['age_units']=age_guard['encoded_age_units']
                    if age_guard['grant_suppressed_for_transport_budget']:
                        current_context.update(arm_action=1,valid_flags=current_context['valid_flags']&~64)
                        decision.update(action='STOP',decision_valid=False,reason='insufficient_transport_and_display_budget')
                if not 0<=age<=465:decision.update(action='STOP',decision_valid=False,reason='expired_before_display')
                render_tick=time.monotonic();canvas=panel(image,roads,result['paths'],lights,controls,selected,result['boundaries'],decision,key,receipt,feedback,age)
                final_age=(time.monotonic()-captured)*1000
                if final_age>500:
                    decision.update(action='STOP',decision_valid=False,reason='expired_before_display')
                    if uart:
                        try:feedback,_=uart.query_state()
                        except (OSError,ValueError) as exc:note('expired_feedback',exc);feedback=None;uart.close();uart=None
                    else:feedback=None
                    canvas=stopped_canvas('Current image expired; no current GO publication')
                row=dict(sequence=count,identity=identity,source_mode='live_capture',capture_mode=args.capture_mode,capture_fps=args.capture_fps,capture_negotiated=next((v for v in reader.info if v.get('event')=='opened'),None),video_layout=args.video_layout,proposal_mode=args.proposal_mode,housing_refinement=args.housing_refinement,lamp_profile=args.lamp_profile,road_profile=args.road_profile,classifiers=args.classifiers,capture_pid=reader.process.pid,
                    capture_parent_size=frame.source_parent_size,capture_roi_pixels=frame.capture_roi_pixels,capture_quality=quality,
                    controls=asdict(controls),key_snapshot=snapshot_json(key) if key else None,key_unchanged=unchanged,
                    roads=roads,lights=lights,signal_selection=selected,signal_metrics=signal_metrics,
                    boundaries=result['boundaries'],paths=result['paths'],hazard=risk,decision=decision,context=current_context,
                    receipt=receipt,feedback=feedback,profile=dict(capture_decode_age_ms=capture_decode_age_ms,quality_ms=quality_ms,road_ms=road_ms,boundary_model_ms=result['model_ms'],boundary_post_ms=result['postprocess_ms'],uart_ms=uart_ms,render_ms=(time.monotonic()-render_tick)*1000,final_age_ms=final_age),
                    capture_recoveries=recoveries,control_sequence_acknowledged_by_FPGA=bool(feedback and feedback['decision_valid'] and feedback['epoch']==key.get('sequence') and receipt['status_acknowledged']),physical_lane_association_verified=False)
                stream.write(json.dumps(row)+'\n');count+=1
                atomic(session/'status.json',dict(state='running',frames=count,updated_unix=time.time(),session=str(session),detections_path=str(stream.name),
                    key_sequence=key.get('sequence'),key_state=key.get('state'),controls=asdict(controls),decision=decision,
                    signal_selection=selected,boundaries=result['boundaries'],feedback=feedback,uart_confirmed=bool(receipt and receipt['status_acknowledged']),
                    age_ms=final_age,capture_pid=reader.process.pid,capture_recoveries=recoveries,lamp_profile=args.lamp_profile,road_profile=args.road_profile,classifiers=args.classifiers,errors=errors[-3:]))
                publish(canvas,captured)
                if evidence_future is not None and evidence_future.done():
                    try:evidence_future.result()
                    except Exception as exc:note('evidence_write',exc)
                    evidence_future=None
                signature=(key.get('state'),decision['action'],selected.get('selected_state'),(feedback or {}).get('action'),(feedback or {}).get('alarm_code'))
                if free>=1024**3 and signature not in evidence and len(evidence)<64 and receipt and receipt['status_acknowledged'] and evidence_future is None:
                    evidence.add(signature);evidence_future=pool.submit(save_evidence,session/'evidence',count,raw_image,image,canvas)
                if count==1 or count%60==0:write_image(session/'latest_preview.jpg',canvas)
        except BaseException as exc:
            note('fatal',exc);publish(stopped_canvas('Program fault: '+repr(exc)))
            with lock:shared['failed']=True
        finally:
            for resource,method in [(reader,'close'),(uart,'close'),(stream,'close'),(pool,'shutdown'),(boundary_cpu,'close'),(boundary,'close'),(lamp,'close'),(guard,'close'),(road,'release')]:
                if resource is not None:
                    try:getattr(resource,method)()
                    except Exception as exc:note('cleanup',exc)
            atomic(session/'summary.json',dict(frames=count,seconds=time.monotonic()-started,errors=errors,capture_recoveries=recoveries))
            atomic(session/'status.json',dict(state='failed' if shared['failed'] else 'stopped',frames=count,updated_unix=time.time(),errors=errors[-3:]))
            if args.headless or not shared['failed']:stop.set()
    if not args.headless:
        cv2.namedWindow(WINDOW,cv2.WINDOW_AUTOSIZE if args.windowed else cv2.WINDOW_NORMAL)
        if not args.windowed:cv2.setWindowProperty(WINDOW,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        cv2.imshow(WINDOW,shared['canvas']);cv2.waitKey(1)
    thread=threading.Thread(target=worker,name='one_NPU_owner',daemon=True);thread.start()
    last_canvas=None;last_stale=None
    try:
        while not stop.is_set():
            if args.headless:thread.join(.1);continue
            with lock:canvas=shared['canvas'];captured=shared['captured']
            stale=captured is not None and time.monotonic()-captured>.5
            if canvas is not last_canvas or stale!=last_stale:
                shown=canvas
                if stale:
                    shown=canvas.copy();cv2.rectangle(shown,(0,0),(1280,130),(20,20,140),-1);cv2.rectangle(shown,(0,620),(1280,720),(20,20,140),-1)
                    cv2.putText(shown,'STOP | displayed image expired',(12,45),0,.8,(255,255,255),2)
                    cv2.putText(shown,'Previous UART reply is a past snapshot; current action not confirmed here',(12,655),0,.58,(255,255,255),1)
                cv2.imshow(WINDOW,shown);last_canvas=canvas;last_stale=stale
            if cv2.waitKey(20)&255 in (27,ord('q')):stop.set()
    finally:
        stop.set();thread.join(12)
        if not args.headless:cv2.destroyAllWindows()


if __name__=='__main__':main()
