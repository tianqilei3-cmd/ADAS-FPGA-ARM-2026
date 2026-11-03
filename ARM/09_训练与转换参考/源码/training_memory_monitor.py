"""记录 Windows 提交额度及当前训练进程内存，帮助定位训练 OOM。"""
import csv
import ctypes
from ctypes import wintypes
from datetime import datetime
import os
from pathlib import Path
from threading import Event, Thread


class MemoryStatus(ctypes.Structure):
    _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
        (name, ctypes.c_ulonglong) for name in (
            "physical_total", "physical_free", "commit_limit", "commit_free",
            "virtual_total", "virtual_free", "extended",
        )
    ]


def memory_snapshot():
    import psutil
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    process = psutil.Process(os.getpid()).memory_info()
    return dict(
        time=datetime.now().isoformat(timespec="seconds"),
        physical_free_GB=round(status.physical_free / 2**30, 3),
        commit_free_GB=round(status.commit_free / 2**30, 3),
        commit_limit_GB=round(status.commit_limit / 2**30, 3),
        process_private_GB=round(getattr(process, "private", process.vms) / 2**30, 3),
        process_RAM_GB=round(process.rss / 2**30, 3),
    )


def start_monitor(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / datetime.now().strftime("%Y%m%d_%H%M%S_memory.csv")
    stop = Event()
    def run():
        try:
            with target.open("w", newline="", encoding="utf-8") as output:
                writer = None
                while not stop.is_set():
                    snapshot = memory_snapshot()
                    if writer is None:
                        writer = csv.DictWriter(output, fieldnames=list(snapshot))
                        writer.writeheader()
                    writer.writerow(snapshot)
                    output.flush()
                    stop.wait(10)
        except Exception as error:
            print(f"内存监测失败：{error}", flush=True)
    thread = Thread(target=run, daemon=True)
    thread.start()
    print(f"内存记录：{target}", flush=True)
    return stop, thread


if __name__ == "__main__":
    print(memory_snapshot())
