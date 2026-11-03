"""Assemble the verified ARM submission materials; never contacts hardware."""
import ast
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'ARM_最终交接'
OLD = ROOT / 'runs/arm_delivery_20261101/package/ARM_run_20261101'
SOURCE_MAP = []


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def copy_file(source, relative, required=True):
    source = Path(source)
    if not source.is_file():
        if required:
            raise FileNotFoundError(source)
        return None
    target = DEST / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    digest = sha(source)
    assert sha(target) == digest
    SOURCE_MAP.append(dict(path=target.relative_to(DEST).as_posix(),
                           original_path=str(source), sha256=digest))
    return target


def copy_tree(source, relative, suffixes=None):
    for path in sorted(Path(source).rglob('*')):
        if not path.is_file() or any(p in {'__pycache__', '.git', '.pytest_cache'} for p in path.parts):
            continue
        if suffixes is None or path.suffix.lower() in suffixes:
            copy_file(path, Path(relative) / path.relative_to(source))


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def collect_reference_code():
    seeds = [
        'train_lamp_capture_aligned_20261028.py', 'evaluate_export_lamp_aligned_20261028.py',
        'convert_lamp_aligned_20261028.py', 'export_lane_board_20261027.py',
        'convert_lane_board_20261027.py', 'export_road_heads_20261028.py',
        'convert_road_heads_20261028.py', 'prepare_road_int8_20261028.py',
        'convert_signal_classifiers_npu_20261027.py', 'prepare_batched_classifiers_20261028.py',
        'prepare_batch2_classifiers_20261028.py', 'export_arm_lamp_attrs_20261024.py',
        'export_fast_signal_types_20261027.py', 'export_signal_int8_heads_20261027.py',
        'export_signal_rect_models_20261027.py', 'lane_boundary_segmentation.py',
        'train_lane_boundary_segmentation_pycharm.py', 'train_s2tld_pycharm.py',
        'train_bdd100k_pycharm.py', 'train_official_lights.py',
        'check_physical_signal_keys_20261028.py', 'run_unified_acceptance_20261028.py',
        'write_engineering_acceptance_report_20261028.py',
        'train_signal_navigation_20261029.py', 'summarize_signal_navigation_20261029.py',
    ]
    visited, pending, absent = set(), list(seeds), []
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        path = ROOT / name
        if not path.is_file():
            absent.append(name)
            continue
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        visited.add(name)
        copy_file(path, Path('09_训练与转换参考/源码') / name)
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [x.name.split('.')[0] for x in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module.split('.')[0]]
            for module in modules:
                if (ROOT / (module + '.py')).is_file():
                    pending.append(module + '.py')
    # Include the local upstream source that supplies YOLO architectures/export.
    copy_tree(ROOT / 'ultralytics', '09_训练与转换参考/ultralytics',
              {'.py', '.yaml', '.yml', '.json', '.txt', '.md'})
    for name in ['pyproject.toml', 'LICENSE', 'README.md', 'README.zh-CN.md']:
        copy_file(ROOT / name, Path('09_训练与转换参考') / name)
    dump(DEST / '09_训练与转换参考/源码依赖索引.json',
         dict(python_files=sorted(visited), missing_seed_files=sorted(set(absent)),
              uses_original_project_relative_data_paths=True,
              direct_training_from_this_folder_guaranteed=False,
              archived_scripts_may_contact_board_if_manually_executed=True))


def main():
    if DEST.exists():
        raise RuntimeError('Existing handoff preserved; choose a new directory or inspect it first')
    DEST.mkdir()
    # Preserve the already independently checked installable runtime byte-for-byte.
    copy_tree(OLD, '01_ARM实机程序')
    frozen = read('runs/arm_delivery_20261101/package/ARM_run_20261101/runtime/manifest.json')
    for item in frozen['files'] + frozen['models']:
        assert sha(DEST / '01_ARM实机程序/runtime' / item['path']) == item['sha256']

    for name in ['play_official_hdmi_source_20261101.py', 'play_official_video_to_fpga_20261027.py']:
        copy_file(ROOT / name, Path('02_电脑视频源') / name)
    copy_file(Path('D:/BaiduNetdiskDownload/紫光同创杯赛题三参考素材/video_20260929_113343.mp4'),
              '02_电脑视频源/示例输入/video_20260929_113343.mp4')

    engineering = 'runs/engineering_acceptance_20261028/20261028_083842_UTC'
    copy_tree(ROOT / engineering, '06_测试结果与统计/原始证据/工程验收_20261028',
              {'.json', '.jsonl', '.log', '.tcl', '.jpg'})
    copy_tree(ROOT / 'runs/basic_finish70_20261028/physical_signal_keys',
              '06_测试结果与统计/原始证据/实体按键_20261028', {'.json', '.jsonl'})
    for name in ['initial_audit.json', 'stop_local_and_devices.json', 'link_preflight.json',
                 'live_launch.json', 'live_check.json', 'steady_check.json',
                 'light_screen_check.json', 'final_state.json', 'user_acceptance.json',
                 'marker_probe.json']:
        copy_file(ROOT / 'runs/restore_hdmi_live_20261101' / name,
                  Path('06_测试结果与统计/原始证据/HDMI恢复_20261101') / name)

    for name in ['evaluation_baseline.json', 'evaluation_candidate.json', 'comparison.json', 'plan.json', 'status.json']:
        copy_file(ROOT / 'runs/arm_signal_navigation_20261029/signal' / name,
                  Path('06_测试结果与统计/原始证据/灯色PC离线_20261029') / name)
        copy_file(ROOT / 'runs/arm_signal_navigation_20261029/navigation' / name,
                  Path('06_测试结果与统计/原始证据/导航PC离线_20261029') / name)
    copy_file(ROOT / 'runs/adas/bdd100k_4class_stage42/report/metrics.json',
              '06_测试结果与统计/原始证据/道路检测PC离线/metrics.json')

    exports = [
        'runs/basic_finish70_20261027/lane_keys/model/export.json',
        'runs/basic_finish70_20261027/lane_keys/model/conversion.json',
        'runs/basic_finish70_20261027/lane_keys/model/native_parity.json',
        'runs/basic_finish70_20261028/road_sparse_heads/export.json',
        'runs/basic_finish70_20261028/road_int8/export.json',
        'runs/basic_finish70_20261028/road_int8/conversion.json',
        'runs/basic_finish70_20261028/road_int8/native_parity.json',
        'runs/basic_finish70_20261028/lamp_capture_aligned/plan.json',
        'runs/basic_finish70_20261028/lamp_capture_aligned/status.json',
        'runs/basic_finish70_20261028/lamp_capture_aligned/validation.json',
        'runs/basic_finish70_20261028/lamp_capture_aligned/export/export.json',
        'runs/basic_finish70_20261028/lamp_capture_aligned/export/conversion.json',
        'runs/basic_finish70_20261028/batch2_classifiers/conversion.json',
        'runs/arm_stage_lamp_20261024/export_verification.json',
        'runs/lane_segmentation_20261017/pc_geometry_candidate.json',
    ]
    for relative in exports:
        copy_file(ROOT / relative, Path('08_来源与许可证/导出和版本追溯') / Path(relative).relative_to('runs'))

    source_weights = [
        'runs/adas/bdd100k_4class_stage42/weights/best.pt',
        'runs/adas/lane_boundary_seg_f721cc098f9b/weights/best.pt',
        'runs/adas/s2tld_official_tiles_stage2/weights/deployment_best.pt',
        'runs/basic_finish70_20261028/lamp_capture_aligned/trial/weights/last.pt',
        'runs/adas/signal_emission_guard_20261017_172936_118725/weights/best.pt',
    ]
    attrs = read('runs/arm_stage_lamp_20261024/export_verification.json')
    for path, digest in attrs['source_hashes'].items():
        candidate = Path(path)
        if candidate.is_file():
            assert sha(candidate) == digest
            source_weights.append(candidate.relative_to(ROOT).as_posix())
    for relative in source_weights:
        copy_file(ROOT / relative, Path('09_训练与转换参考/源训练权重') / Path(relative).relative_to('runs'))

    # Dataset manifests contain paths and annotations, not whole third-party datasets.
    for relative in [
        'runs/basic_finish70_20261028/lamp_capture_aligned/dataset_manifest.json',
        'runs/arm_signal_navigation_20261029/signal/dataset_manifest.json',
        'runs/arm_signal_navigation_20261029/navigation/dataset_manifest.json',
        'datasets/adas/prepared/official_light_annotation_v1/manifest.json',
    ]:
        copy_file(ROOT / relative, Path('08_来源与许可证/数据划分记录') / relative, required=False)
    copy_file(ROOT / 'runs/navigation_runtime_20261019/exports/780f699d83e9/manifest.json',
              '08_来源与许可证/导出和版本追溯/color_guard_export_manifest.json')
    for filename in ['adas_basic70_control.v', 'adas_key_control.v', 'adas_key_epoch_gate.v']:
        copy_file(Path('D:/PDS_WORK/adas_fullres_20261028/source') / filename,
                  Path('05_通信协议与真实日志/对应FPGA接口摘录') / filename)

    for name in ['ARM运行交接与快捷方式_2026-11-01.md', 'HDMI实时采集恢复记录_2026-11-01.md',
                 '红绿灯与导航模型补训结果_2026-10-29.md', '训练数据集交接文档_2026-10-06.md',
                 'ARM端进度与FPGA协作_AI交接文档_2026-10-10.md']:
        copy_file(ROOT / name, Path('08_来源与许可证/历史说明') / name)
    copy_file(ROOT / 'LICENSE', '08_来源与许可证/许可证原文/Ultralytics_AGPL-3.0.txt')
    for package in ['torch', 'numpy', 'opencv-python', 'onnxruntime', 'PyYAML']:
        dist = importlib.metadata.distribution(package)
        candidates = [p for p in dist.files if str(p).lower().endswith(('license', 'license.txt'))]
        if candidates:
            chosen = min(candidates, key=lambda p: len(str(p)))
            copy_file(dist.locate_file(chosen), Path('08_来源与许可证/许可证原文') / (package + '_本机依赖_LICENSE.txt'))

    collect_reference_code()
    from arm_final_handoff_content_20261102 import write_materials
    write_materials(DEST, ROOT, engineering, copy_file)
    copy_file(Path(__file__), '10_整理与校验工具/build_arm_final_handoff_20261102.py')
    copy_file(ROOT / 'arm_final_handoff_content_20261102.py', '10_整理与校验工具/arm_final_handoff_content_20261102.py')
    copy_file(ROOT / 'verify_arm_final_handoff_20261102.py', 'verify_arm_final_handoff.py')
    copy_file(ROOT / 'recompute_arm_handoff_metrics_20261102.py', '06_测试结果与统计/recompute_metrics.py')
    dump(DEST / '08_来源与许可证/复制文件来源与SHA256.json', SOURCE_MAP)
    finalize()


def finalize():
    # Check actual runtime modules have their local import dependencies included.
    runtime = DEST / '01_ARM实机程序/runtime'
    local_names = {p.stem for p in ROOT.glob('*.py')}
    missing = []
    for p in runtime.glob('*.py'):
        tree = ast.parse(p.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [x.name.split('.')[0] for x in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split('.')[0]]
            for name in names:
                if name in local_names and not (runtime / (name + '.py')).is_file():
                    missing.append(dict(file=p.name, module=name))
    assert not missing, missing
    # Text-only credential check. Never emit any matched content.
    patterns = [rb'hf_[A-Za-z0-9]{20,}', rb'sk-[A-Za-z0-9_-]{25,}',
                rb'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----']
    bad = []
    for p in DEST.rglob('*'):
        if p.is_file() and p.suffix.lower() in {'.py', '.json', '.jsonl', '.md', '.txt', '.ini', '.toml', '.yaml', '.yml', '.html', '.sh', '.log', '.tcl', '.csv'}:
            raw = p.read_bytes()
            if any(re.search(pattern, raw) for pattern in patterns):
                bad.append(p.relative_to(DEST).as_posix())
    assert not bad, 'Credential-like data found; inspect these filenames: ' + repr(bad)
    for path in DEST.rglob('*.py'):
        ast.parse(path.read_text(encoding='utf-8-sig'))
    files = [dict(path=p.relative_to(DEST).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
             for p in sorted(DEST.rglob('*')) if p.is_file() and p.name != 'HANDOFF_MANIFEST.json']
    dump(DEST / 'HANDOFF_MANIFEST.json', dict(
        schema='ARM_final_handoff_20261102_v1',
        created=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        demo_baseline='signal_recovered_20261028',
        latest_live_recovery='2026-11-01', models=15, frozen_source_files=64,
        full_competition_accepted=False, physical_ego_lane_association_verified=False,
        credential_scan_passed=True, runtime_local_imports_complete=True, files=files))
    archive = ROOT / 'ARM_最终交接_2026-11-02.zip'
    if archive.exists():
        checksum = archive.with_suffix('.zip.sha256')
        assert checksum.exists() and checksum.read_text(encoding='utf-8').split()[0] == sha(archive), 'Unknown ZIP preserved'
    temporary_archive = archive.with_suffix('.zip.part')
    assert not temporary_archive.exists(), 'Unfinished archive preserved'
    with zipfile.ZipFile(temporary_archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(DEST.rglob('*')):
            if not p.is_file():
                continue
            info = zipfile.ZipInfo('ARM_最终交接/' + p.relative_to(DEST).as_posix())
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100000 | (0o755 if p.suffix == '.sh' else 0o644)) << 16
            z.writestr(info, p.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    with zipfile.ZipFile(temporary_archive) as archive_check:
        assert archive_check.testzip() is None
    temporary_archive.replace(archive)
    archive.with_suffix('.zip.sha256').write_text(sha(archive) + '  ' + archive.name + '\n', encoding='utf-8')
    result = dict(folder=str(DEST), zip=str(archive), bytes=archive.stat().st_size,
                  files=len(files) + 1, sha256=sha(archive), hardware_contacted=False)
    dump(ROOT / 'runs/arm_final_handoff_20261102/build_result.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
