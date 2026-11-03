"""Loop official road video on the external HDMI display at 1080p60."""
import argparse
import ctypes
from ctypes import wintypes as w
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'runs/basic_finish70_20261027'


class DisplayDevice(ctypes.Structure):
    _fields_ = [('cb', w.DWORD), ('name', w.WCHAR * 32), ('description', w.WCHAR * 128),
                ('state', w.DWORD), ('id', w.WCHAR * 128), ('key', w.WCHAR * 128)]


class DevMode(ctypes.Structure):
    _fields_ = [('name', w.WCHAR * 32), ('spec', w.WORD), ('driver', w.WORD),
                ('size', w.WORD), ('extra', w.WORD), ('fields', w.DWORD),
                ('x', w.LONG), ('y', w.LONG), ('orientation', w.DWORD), ('fixed_output', w.DWORD),
                ('color', w.SHORT), ('duplex', w.SHORT), ('yresolution', w.SHORT),
                ('ttoption', w.SHORT), ('collate', w.SHORT), ('form', w.WCHAR * 32),
                ('logpixels', w.WORD), ('bits', w.DWORD), ('width', w.DWORD), ('height', w.DWORD),
                ('flags', w.DWORD), ('frequency', w.DWORD), ('icm_method', w.DWORD),
                ('icm_intent', w.DWORD), ('media_type', w.DWORD), ('dither_type', w.DWORD),
                ('reserved1', w.DWORD), ('reserved2', w.DWORD),
                ('panning_width', w.DWORD), ('panning_height', w.DWORD)]


def external_display():
    displays = []
    for i in range(16):
        device = DisplayDevice()
        device.cb = ctypes.sizeof(device)
        if not ctypes.windll.user32.EnumDisplayDevicesW(None, i, ctypes.byref(device), 0):
            break
        if device.state & 1 and not device.state & 4:
            mode = DevMode()
            mode.size = ctypes.sizeof(mode)
            if ctypes.windll.user32.EnumDisplaySettingsW(device.name, -1, ctypes.byref(mode)):
                displays.append((device.name, mode))
    if len(displays) != 1:
        raise RuntimeError('Expected one active external display; found ' + str(len(displays)))
    return displays[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default='D:/BaiduNetdiskDownload/紫光同创杯赛题三参考素材/video_20260929_113421.mp4')
    parser.add_argument('--start-frame',type=int,default=0)
    parser.add_argument('--end-frame',type=int,help='Exclusive source frame; loop only this interval')
    parser.add_argument('--refresh',type=int,choices=[30,60],default=60)
    args = parser.parse_args()
    video = Path(args.video)
    if not video.is_file():
        raise FileNotFoundError(video)
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except (AttributeError, OSError):
        ctypes.windll.user32.SetProcessDPIAware()
    device, mode = external_display()
    original = dict(device=device, x=mode.x, y=mode.y, width=mode.width,
                    height=mode.height, frequency=mode.frequency)
    reconfigured=(mode.width,mode.height,mode.frequency)!=(1920,1080,args.refresh)
    if reconfigured:
        mode.width, mode.height, mode.frequency = 1920, 1080, args.refresh
        mode.fields |= 0x00080000 | 0x00100000 | 0x00400000
        test = ctypes.windll.user32.ChangeDisplaySettingsExW(device, ctypes.byref(mode), None, 2, None)
        if test != 0:
            raise RuntimeError('Requested 1080p display mode test failed: ' + str(test))
        changed = ctypes.windll.user32.ChangeDisplaySettingsExW(device, ctypes.byref(mode), None, 0, None)
        if changed != 0:
            raise RuntimeError('Display mode change failed: ' + str(changed))
    _, mode = external_display()
    assert mode.width == 1920 and mode.height == 1080 and mode.frequency == args.refresh
    os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '0'
    os.environ['QT_SCALE_FACTOR'] = '1'
    import cv2
    cv2.setNumThreads(2)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError('Cannot decode official video')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not 1 <= fps <= 120:
        raise ValueError('Invalid source FPS')
    total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame=total_frames if args.end_frame is None else args.end_frame
    if not 0<=args.start_frame<end_frame<=total_frames:
        raise ValueError('Invalid source frame interval')
    if args.start_frame:cap.set(cv2.CAP_PROP_POS_FRAMES,args.start_frame)
    title = 'Official road video -> FPGA HDMI | Q / ESC to stop'
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1920, 1080)
    cv2.moveWindow(title, mode.x, mode.y)
    cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    report = dict(pid=os.getpid(), video=str(video), source_fps=fps, external_display=device,
                  x=mode.x,y=mode.y,width=mode.width,height=mode.height,frequency=mode.frequency,
                  original_mode=original,display_mode_reconfigured=reconfigured,frames=0,loops=0,state='starting',
                  start_frame=args.start_frame,end_frame_exclusive=end_frame)
    # Disk/CPU work and video display do not themselves prevent display idle.
    # Hold this request only for this player's lifetime; no power-plan changes.
    # https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate
    keep_awake=ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
    if not keep_awake:raise OSError('Cannot keep the HDMI source display active')
    report['display_sleep_inhibited_while_playing']=True
    (OUT / 'player.pid').write_text(str(os.getpid()))
    deadline = time.monotonic()
    try:
        while True:
            if cap.get(cv2.CAP_PROP_POS_FRAMES)>=end_frame:
                ok,frame=False,None
            else:ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError('Video loop rewind failed')
                report['loops'] += 1
                deadline = time.monotonic()
            cv2.imshow(title, cv2.resize(frame, (1920, 1080)))
            report['frames'] += 1
            report['source_frame']=int(cap.get(cv2.CAP_PROP_POS_FRAMES))-1
            if report['frames'] == 1 or report['frames'] % 30 == 0:
                _,observed_mode=external_display()
                observed=(observed_mode.width,observed_mode.height,observed_mode.frequency)
                report['observed_display_mode']=dict(width=observed[0],height=observed[1],frequency=observed[2])
                if observed!=(1920,1080,args.refresh):
                    report['source_fault']='HDMI display mode changed during playback'
                    raise RuntimeError(report['source_fault']+': '+str(observed))
                report.update(state='playing',updated_unix=time.time())
                temporary = OUT / 'player_status.json.tmp'
                temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
                temporary.replace(OUT / 'player_status.json')
            deadline += 1 / fps
            key = cv2.waitKey(max(1, round((deadline-time.monotonic())*1000))) & 255
            if key in (27, ord('q')):
                break
            if time.monotonic() - deadline > .25:
                deadline = time.monotonic()
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        cap.release()
        cv2.destroyAllWindows()
        report.update(state='stopped',updated_unix=time.time())
        (OUT / 'player_status.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')


if __name__ == '__main__':
    main()
