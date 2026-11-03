"""Install small desktop controls while preserving the verified runtime."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from adas_launcher import verify_runtime, MANIFEST_SHA


def desktop_quote(value):
    value = str(value).replace("%", "%%")
    for character in ("\\", '"', "`", "$"):
        value = value.replace(character, "\\"+character)
    return '"'+value+'"'


def write_owned(path, data, mode=0o644):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != data:
        backup = path.with_name(path.name+".before_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
        shutil.copy2(path, backup)
    path.write_bytes(data)
    path.chmod(mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path,
                        default=Path("/home/orangepi/adas2026/signal_recovered_20261028"))
    parser.add_argument("--restore-if-missing", action="store_true")
    args = parser.parse_args()
    if os.name != "posix" or Path.home().resolve() != Path("/home/orangepi"):
        parser.error("请以orangepi用户在这块香橙派运行，不使用sudo。")
    root = args.runtime_root.resolve()
    if root != Path("/home/orangepi/adas2026/signal_recovered_20261028"):
        parser.error("仅支持当前固定恢复目录")
    source = Path(__file__).resolve().parent
    restored = False
    if not root.exists():
        payload = source.parent/"runtime"
        if not args.restore_if_missing or not payload.is_dir():
            raise RuntimeError("运行目录不存在。完整运行包可加 --restore-if-missing 恢复；现有目录不会被覆盖。")
        verify_runtime(payload)
        root.parent.mkdir(parents=True, exist_ok=True)
        needed = sum(p.stat().st_size for p in payload.rglob("*") if p.is_file())
        if shutil.disk_usage(root.parent).free < needed+512*2**20:
            raise RuntimeError("空间不足：需要运行文件体积加512MiB余量。")
        shutil.copytree(payload, root)
        restored = True
    verified = verify_runtime(root)
    target = Path.home()/".local/share/adas2026/desktop_20261101"
    expected_parent = Path.home()/".local/share/adas2026"
    expected_parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.resolve() != expected_parent.resolve()/target.name:
        raise RuntimeError("快捷方式目录为意外符号链接，停止安装。")
    target.mkdir(parents=True, exist_ok=True)
    helper = target/"adas_launcher.py"
    write_owned(helper, (source/"adas_launcher.py").read_bytes(), 0o755)
    # Expose installer beside the helper for future re-installations.
    if source != target:
        write_owned(target/"install_shortcuts.py", Path(__file__).read_bytes())
    try:
        desktop = Path(subprocess.check_output(["xdg-user-dir", "DESKTOP"], text=True).strip()).resolve()
    except (OSError, subprocess.CalledProcessError):
        desktop = Path.home()/"Desktop"
    if desktop != Path.home().resolve() and Path.home().resolve() not in desktop.parents:
        raise RuntimeError("桌面目录不在当前用户家目录内。")
    if desktop == Path.home().resolve():
        desktop = Path.home()/"Desktop"
    desktop.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    bus = Path(f"/run/user/{os.getuid()}/bus")
    if bus.exists():
        environment.setdefault("DBUS_SESSION_BUS_ADDRESS", "unix:path="+str(bus))
    entries = []
    for action, title, icon in [("start", "ADAS 道路演示", "video-display"),
                                ("stop", "ADAS 停止演示", "process-stop"),
                                ("check", "ADAS 运行诊断", "utilities-terminal")]:
        path = desktop/("ADAS_"+action+".desktop")
        if path.exists() and "X-ADAS-Launcher-Version=" not in path.read_text(encoding="utf-8"):
            raise RuntimeError("桌面存在同名的其他快捷方式，未覆盖："+str(path))
        command = " ".join(map(desktop_quote, [sys.executable, helper, action, "--runtime-root", root]))
        text = "\n".join(["[Desktop Entry]", "Version=1.0", "Type=Application", "Name="+title,
                           "Comment=Verified ARM runtime; uses existing RKNN environment",
                           "Exec="+command, "Icon="+icon, "Terminal=true", "StartupNotify=false",
                           "Categories=Education;Science;", "X-ADAS-Launcher-Version=20261101", ""])
        write_owned(path, text.encode("utf-8"), 0o755)
        validator = shutil.which("desktop-file-validate")
        if validator:
            subprocess.run([validator, str(path)], check=True, capture_output=True, text=True)
        trusted = subprocess.run(["gio", "set", str(path), "metadata::trusted", "true"],
                                 env=environment, capture_output=True, text=True) if shutil.which("gio") else None
        entries.append(dict(path=str(path), executable=True, desktop_syntax_valid=bool(validator),
                            trusted=trusted is not None and trusted.returncode == 0,
                            trust_note=trusted.stderr.strip() if trusted is not None and trusted.returncode else None))
    result = dict(installed=True, restored_runtime=restored, runtime=verified, shortcuts=entries,
                  helper=str(helper), helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
                  started_demo=False, runtime_source_or_models_changed=False,
                  created=datetime.now().astimezone().isoformat())
    (target/"installation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
