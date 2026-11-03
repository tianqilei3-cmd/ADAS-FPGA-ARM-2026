"""Desktop controls for the hash-frozen ARM runtime; no YOLO dependency."""
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

MANIFEST_SHA = "09c4795d9457f6aff117643234edb6c7fcdd12395a5d7268ec11c32aba0005dd"
BOOTSTRAP_SHA = "987c615172378032b5ce0e565b0bf66829b341b78c5b7ac55c3196ad02adc1fc"
RUNTIME_NAMES = {"run_unified_demo.py", "run_lane_demo.py", "run_screen_demo.py",
                 "start_recovered_signal_on_board_20261028.py"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_runtime(root):
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or sha(manifest_path) != MANIFEST_SHA:
        raise RuntimeError("运行目录不属于本快捷方式的冻结版本；请保留队友改动，核对版本后再使用。")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checked, bad = [], []
    for item in manifest["files"] + manifest["models"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid manifest path")
        path = root / relative
        if not path.is_file() or sha(path) != item["sha256"]:
            bad.append(item["path"])
        else:
            checked.append(item["path"])
    if bad:
        raise RuntimeError("源码或模型校验失败：" + ", ".join(bad))
    bootstrap = root / "start_recovered_signal_on_board_20261028.py"
    if not bootstrap.is_file() or sha(bootstrap) != BOOTSTRAP_SHA:
        raise RuntimeError("启动预检程序缺失或版本不同；请保留改动并核对版本。")
    return dict(verified=True, files=len(manifest["files"]), models=len(manifest["models"]),
                manifest_sha256=MANIFEST_SHA, runtime=str(root))


def runtime_processes(root, include_other_releases=False):
    root = Path(root).resolve()
    results = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            cwd = (entry / "cwd").resolve()
            args = (entry / "cmdline").read_bytes().split(b"\0")
            for argument in args:
                value = argument.decode(errors="replace")
                if Path(value).name not in RUNTIME_NAMES:
                    continue
                candidate = Path(value)
                candidate = (candidate if candidate.is_absolute() else cwd/candidate).resolve()
                own = candidate.parent == root
                other = include_other_releases and Path.home()/"adas2026" in candidate.parents
                if own or other:
                    results.append(dict(pid=int(entry.name), script=str(candidate), own_runtime=own))
                    break
        except (OSError, ValueError):
            continue
    return results


def dependencies():
    code = "import json,cv2,numpy,onnxruntime;from rknnlite.api import RKNNLite;print(json.dumps(dict(cv2=cv2.__version__,numpy=numpy.__version__,onnxruntime=onnxruntime.__version__,rknnlite_imported=True)))"
    try:
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           timeout=30, env=dict(os.environ, OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2"))
        return dict(ready=p.returncode == 0, versions=json.loads(p.stdout) if p.returncode == 0 else None,
                    error=p.stderr[-2000:] if p.returncode else None)
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        return dict(ready=False, error=repr(exc))


def check(root):
    result = verify_runtime(root)
    result.update(time=datetime.now().astimezone().isoformat(), dependencies=dependencies(),
                  running=runtime_processes(root, True),
                  disk_free_MiB=round(shutil.disk_usage(root).free/2**20, 1),
                  display_available=bool(os.environ.get("DISPLAY")) or Path("/tmp/.X11-unix/X0").exists(),
                  devices={p: dict(exists=Path(p).exists(), readable_writable=os.access(p, os.R_OK|os.W_OK))
                           for p in ("/dev/video0", "/dev/ttyUSB0")},
                  expected_fpga_profile=41, functional_test_performed=False,
                  note="仅诊断文件、依赖和设备可访问性；没有占用摄像头或UART，没有验证本次灯色/按键/视频源。")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def start(root):
    import fcntl
    # Keep the byte lock across both Python exec transitions of the old launcher.
    state = Path.home()/".local/state/adas2026"
    state.mkdir(parents=True, exist_ok=True)
    lock = (state/"desktop_runtime.lock").open("a+b")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("已有桌面启动正在进行。请勿重复启动；需要时先点停止快捷方式。")
    running = runtime_processes(root, True)
    if running:
        raise RuntimeError("已有演示程序占用设备，请先正常退出：" + repr(running))
    report = check(root)
    if not report["dependencies"]["ready"]:
        raise RuntimeError("香橙派运行依赖不完整，详见上方诊断。队友电脑无需安装YOLO。")
    if not all(v["readable_writable"] for v in report["devices"].values()):
        raise RuntimeError("请连接HDMI采集卡和USB-UART，并核对设备路径与读写权限。")
    if report["disk_free_MiB"] < 512:
        raise RuntimeError("香橙派可用空间不足512MiB，先归档旧日志。")
    if not report["display_available"]:
        raise RuntimeError("未发现Linux图形桌面；请在香橙派桌面或已登录的显示会话中启动。")
    print("请保持FPGA已配置0x29，并在HDMI输入端播放道路视频。启动后先停车；Q或Esc退出。", flush=True)
    (state/"last_desktop_start.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.set_inheritable(lock.fileno(), True)
    environment = dict(os.environ, DISPLAY=os.environ.get("DISPLAY") or ":0",
                       XAUTHORITY=os.environ.get("XAUTHORITY") or str(Path.home()/".Xauthority"),
                       XDG_RUNTIME_DIR=os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}",
                       OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    command = [sys.executable, "-u", str(root/"start_recovered_signal_on_board_20261028.py"),
               "--arm-dir", str(root)]
    os.chdir(root)
    os.execve(sys.executable, command, environment)


def stop(root):
    owned = runtime_processes(root)
    for item in owned:
        try:
            # SIGINT lets bootstrap finally blocks and the GUI's handlers release devices.
            os.kill(item["pid"], signal.SIGINT)
        except ProcessLookupError:
            pass
    deadline = time.monotonic()+15
    while owned and time.monotonic() < deadline:
        remaining = runtime_processes(root)
        if not remaining:
            break
        time.sleep(0.2)
    else:
        remaining = runtime_processes(root)
    if remaining:
        raise RuntimeError("程序尚未退出；未强制终止其他进程，请查看屏幕/终端：" + repr(remaining))
    print(json.dumps(dict(stop_requested=[r["pid"] for r in owned], own_runtime_stopped=True,
                          unrelated_processes_terminated=False, hardware_feedback_queried=False), ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "stop", "check"])
    parser.add_argument("--runtime-root", type=Path,
                        default=Path.home()/"adas2026/signal_recovered_20261028")
    args = parser.parse_args()
    if os.name != "posix":
        parser.error("请在香橙派Linux运行本入口，电脑只用于SSH和文件传输。")
    root = args.runtime_root.expanduser().resolve()
    if root != Path("/home/orangepi/adas2026/signal_recovered_20261028"):
        raise RuntimeError("仅支持当前已验证的香橙派固定目录，其他版本请另建适配入口。")
    {"start": start, "stop": stop, "check": check}[args.action](root)
    if args.action == "check" and sys.stdin.isatty():
        input("诊断完成，按回车关闭。")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nADAS启动/诊断错误："+str(error), file=sys.stderr, flush=True)
        if sys.stdin.isatty():
            input("按回车关闭窗口，保留错误文字便于排查。")
        raise SystemExit(1)
