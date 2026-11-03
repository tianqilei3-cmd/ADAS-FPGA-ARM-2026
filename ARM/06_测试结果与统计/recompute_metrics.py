"""Recompute descriptive results from the bundled records without hardware/AI."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


CASES = [
    ('实体KEY6绿行', '20261028_084737_UTC', 'green_go', 65),
    ('实体KEY6停车', '20261028_085102_UTC', 'physical_stop', 90),
    ('实体KEY6红停', '20261028_085727_UTC', 'red_stop', 89),
    ('实体KEY7强行红灯报警', '20261028_090840_UTC', 'forced_red', 88),
    ('实体KEY6停车清除报警', '20261028_091018_UTC', 'alarm_clear', 89),
    ('辅助冷恢复后实体KEY6绿行', '20261028_094032_UTC', 'green_go', 66),
]


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def check_row(row, case):
    key, fb = row['key_snapshot'], row.get('feedback') or {}
    dec, identity = row['decision'], row['identity']
    identity_ok = (fb.get('camera_id') == 0 and
                   fb.get('frame_id') == identity['frame_id'] & 65535 and
                   fb.get('epoch') == key['sequence'] and fb.get('scene') == 1)
    checks = dict(input_valid=row['capture_quality']['valid'],
                  fresh=0 <= row['profile']['final_age_ms'] <= 500,
                  unchanged=row['key_unchanged'],
                  identity=identity_ok,
                  acknowledged=bool(row.get('receipt') and row['receipt']['status_acknowledged']),
                  physical=key.get('source') == 'physical_keys' and not key['state'] & 64,
                  feedback_valid=fb.get('decision_valid', False),
                  no_reject=fb.get('reject_reason') == 0 and not fb.get('link_timeout', True))
    if case in ['physical_stop', 'alarm_clear']:
        checks.update(driver_stopped=not key['state'] & 16,
                      arm_stopped=dec['action'] == 'STOP', feedback_stopped=fb.get('action') == 1)
        if case == 'alarm_clear':
            checks.update(alarm_cleared=not dec['alarm'],
                          feedback_alarm_cleared=fb.get('alarm_code') == 0,
                          forced_cleared=not row['controls']['forced'])
    else:
        selected = row['signal_selection']
        green = case == 'green_go'
        checks.update(arm_valid=dec['decision_valid'],
                      arm_action=dec['action'] == ('GO' if green else 'STOP'),
                      light=bool(selected['selected_valid'] and selected['simulation_association_valid']
                                 and selected['selected_state'] == ('green' if green else 'red')),
                      fpga_action=fb.get('action') == (2 if green else 1),
                      alarm=fb.get('alarm_code') == (2 if case == 'forced_red' else 0),
                      epoch_ack=row['control_sequence_acknowledged_by_FPGA'])
    return all(checks.values())


def percentile(values, q):
    """Nearest-rank empirical percentile, without interpolation."""
    a = sorted(values)
    return a[max(0, math.ceil(q * len(a)) - 1)]


def compute(root):
    root = Path(root)
    data = root / '06_测试结果与统计/原始证据'
    cases = []
    for label, folder, case, expected in CASES:
        p = data / '实体按键_20261028' / folder / 'observations.jsonl'
        records = [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s]
        scores = [check_row(r['observation'], case) for r in records]
        successes = sum(scores)
        assert len(records) == 90 and successes == expected, (label, len(records), successes, expected)
        unsafe = sum(bool(r['score']['unsafe']) for r in records)
        cases.append(dict(case=label, frames=len(records), successes=successes,
                          ratio=successes / len(records), unsafe_frames=unsafe,
                          observations_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                          path=p.relative_to(root).as_posix(),
                          criterion='all checks per frame; first transition/failures retained'))
    perf = load(data / 'HDMI恢复_20261101/steady_check.json')
    rows = perf['rows']
    ages = [r['profile']['final_age_ms'] for r in rows]
    fpstime = perf['end_status']['updated_unix'] - perf['start_status']['updated_unix']
    n = perf['end_status']['frames'] - perf['start_status']['frames']
    performance = dict(frames=n, elapsed_seconds=fpstime, inference_pipeline_fps=n / fpstime,
                       rows=len(rows), state_acknowledged=sum(bool(r['receipt'] and r['receipt']['status_acknowledged']) for r in rows),
                       input_quality_valid=sum(r['capture_quality']['valid'] for r in rows),
                       final_age_ms=dict(mean=statistics.mean(ages), median=statistics.median(ages),
                                         p95=percentile(ages, .95), minimum=min(ages), maximum=max(ages)),
                       profile_mean_ms={k: statistics.mean(r['profile'][k] for r in rows) for k in rows[0]['profile']},
                       capture_recoveries=perf['end_status']['capture_recoveries'],
                       source_modes=sorted({r['source_mode'] for r in rows}),
                       scene_modes=sorted({r['controls']['mode'] for r in rows}),
                       sensor_to_display_latency_measured=False,
                       freshness_clock='userspace decode completion')
    baseline = load(data / '灯色PC离线_20261029/evaluation_baseline.json')
    signals = []
    for view, total in baseline['totals'].items():
        tp, fp, fn = total['tp'], total['fp'], total['fn']
        signals.append(dict(view=view, tp=tp, fp=fp, fn=fn,
                            precision=tp / (tp + fp), recall=tp / (tp + fn),
                            f1=2 * tp / (2 * tp + fp + fn),
                            truth=total['truth'], green_tp=total['green_tp'], green_truth=total['green_truth']))
    return dict(physical_cases=cases, current_hdmi_short_test=performance,
                lamp_PC_baseline=signals, independent_samples_claimed=False,
                model_training_or_hardware_access_performed=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = compute(args.root)
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
