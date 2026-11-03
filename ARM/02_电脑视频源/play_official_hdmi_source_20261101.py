"""Play a real HDMI source, supporting a primary screen mirrored upstream.

Does not alter display resolution or supply perception labels to Orange Pi.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import time

from play_official_video_to_fpga_20261027 import DisplayDevice, DevMode


def active_displays():
    rows = []
    for index in range(16):
        device = DisplayDevice()
        device.cb = ctypes.sizeof(device)
        if not ctypes.windll.user32.EnumDisplayDevicesW(None, index, ctypes.byref(device), 0):
            break
        if not device.state & 1:
            continue
        mode = DevMode()
        mode.size = ctypes.sizeof(mode)
        if ctypes.windll.user32.EnumDisplaySettingsW(device.name, -1, ctypes.byref(mode)):
            rows.append((device.name, mode, bool(device.state & 4)))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=21600)
    args = parser.parse_args()
    if not args.video.is_file() or not 0 < args.duration <= 21600:
        parser.error("Existing video and duration in (0, 21600] required")
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except (AttributeError, OSError):
        ctypes.windll.user32.SetProcessDPIAware()
    displays = active_displays()
    external = [row for row in displays if not row[2]]
    if len(external) > 1:
        raise RuntimeError("Multiple external displays; select the HDMI target first")
    selected = external or [row for row in displays if row[2]]
    if len(selected) != 1:
        raise RuntimeError("No unambiguous source display")
    name, mode, primary = selected[0]
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
    os.environ["QT_SCALE_FACTOR"] = "1"
    import cv2
    import numpy as np
    cv2.setNumThreads(2)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError("Cannot decode source video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not 1 <= fps <= 120:
        cap.release()
        raise ValueError("Invalid source FPS")
    out = Path(__file__).resolve().parent / "runs/restore_hdmi_live_20261101"
    out.mkdir(exist_ok=True)
    status = out / "player_status.json"
    title = "Official road video - LIVE HDMI SOURCE - Q or ESC to stop"
    started = time.monotonic()
    deadline = started
    report = dict(pid=os.getpid(), video=str(args.video), source_fps=fps,
                  display=name, primary_screen_used=primary,
                  width=mode.width, height=mode.height, frequency=mode.frequency,
                  display_mode_changed=False, frames=0, loops=0,
                  maximum_duration_s=args.duration, perception_labels_supplied=False)
    try:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, mode.width, mode.height)
        cv2.moveWindow(title, mode.x, mode.y)
        cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setWindowProperty(title, cv2.WND_PROP_TOPMOST, 1)
        if not ctypes.windll.kernel32.SetThreadExecutionState(0x80000003):
            raise OSError("Cannot inhibit display sleep during playback")
        canvas = np.zeros((mode.height, mode.width, 3), np.uint8)
        while time.monotonic() - started < args.duration:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Video rewind failed")
                report["loops"] += 1
                deadline = time.monotonic()
            h, w = frame.shape[:2]
            scale = min(mode.width / w, mode.height / h)
            dw, dh = round(w * scale), round(h * scale)
            left, top = (mode.width - dw) // 2, (mode.height - dh) // 2
            canvas.fill(0)
            canvas[top:top + dh, left:left + dw] = cv2.resize(frame, (dw, dh))
            cv2.imshow(title, canvas)
            report["frames"] += 1
            report["source_frame"] = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if report["frames"] == 1 or report["frames"] % 30 == 0:
                report.update(state="playing", updated_unix=time.time())
                temporary = status.with_suffix(".tmp")
                temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
                temporary.replace(status)
            deadline += 1 / fps
            key = cv2.waitKey(max(1, round((deadline - time.monotonic()) * 1000))) & 255
            if key in (27, ord("q")):
                break
            if time.monotonic() - deadline > .25:
                deadline = time.monotonic()
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        cap.release()
        cv2.destroyAllWindows()
        report.update(state="stopped", updated_unix=time.time())
        status.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
