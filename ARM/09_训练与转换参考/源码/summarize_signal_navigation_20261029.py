"""Read-only training audit; writes a compact report without board access."""
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/arm_signal_navigation_20261029"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def percent(number):
    return f"{100*number:.2f}%"


def main():
    lines = ["# 红绿灯与导航模型补训结果", "",
             "更新时间：" + datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"), "",
             "本轮只在笔记本CPU训练，未访问板卡、未部署候选。下面的P/R是固定阈值检测指标，不是实机闭环成功率。", ""]
    audit = dict(tasks={}, board_access=False, automatic_deployment=False)
    for task, title in [("signal", "红绿灯"), ("navigation", "导航提示卡片")]:
        folder = OUT / task
        lines += ["## " + title, ""]
        if not (folder / "status.json").exists():
            lines += ["尚未开始。", ""]
            continue
        status = load(folder / "status.json")
        lines += [f"状态：{status['state']}；完成{status.get('completed_epochs', 0)}轮，优化器更新{status.get('updates', 0)}次。", ""]
        plan = load(folder / "plan.json")
        unchanged = digest(plan["initial"]) == plan["initial_sha256"]
        assert unchanged, f"Source checkpoint changed: {task}"
        samples = []
        for path in (folder / "memory").glob("*.csv"):
            with path.open(encoding="utf-8") as stream:
                samples.extend(csv.DictReader(stream))
        memory_peak = max((float(row["process_RAM_GB"]) for row in samples), default=0.0)
        commit_min = min((float(row["commit_free_GB"]) for row in samples), default=0.0)
        audit["tasks"][task] = dict(state=status["state"], completed_epochs=status.get("completed_epochs", 0),
                                   updates=status.get("updates", 0), initial_checkpoint_unchanged=unchanged,
                                   peak_process_RAM_GiB=memory_peak, min_commit_free_GiB=commit_min)
        lines += [f"抽样进程内存峰值{memory_peak:.3f}GiB；Windows提交额度最少剩余{commit_min:.3f}GiB。起始权重SHA核对一致。", ""]
        comparison_path = folder / "comparison.json"
        if not comparison_path.exists():
            lines += ["候选尚未完成离线比较。", ""]
            continue
        comparison = load(comparison_path)
        lines += ["| 验证输入 | 模型 | 精确率P | 召回率R | F1 |", "|---|---|---:|---:|---:|"]
        for view, caption in [("clean", "原验证画面"), ("capture_hard", "相关退化压力画面")]:
            for key, label in [("baseline", "原模型"), ("candidate", f"第{plan['epochs']}轮候选")]:
                scores = comparison[key][view]
                lines.append(f"| {caption} | {label} | {percent(scores['precision'])} | {percent(scores['recall'])} | {percent(scores['f1'])} |")
        lines += [""]
        if task == "signal":
            for view, caption in [("clean", "原验证画面"), ("capture_hard", "退化画面")]:
                b, c = (comparison[k][view] for k in ("baseline", "candidate"))
                lines += [f"{caption}绿灯正确检出：{b['green_tp']}/{b['green_truth']}→{c['green_tp']}/{c['green_truth']}。"
                          f"与红灯真值重叠的绿灯预测计数：{b.get('red_as_green_overlaps', 0)}→{c.get('red_as_green_overlaps', 0)}。", ""]
            lines += ["上述颜色重叠是检测器诊断项；后续灯型/方向、灯头选择和仲裁会继续处理，不能据此直接判定硬件发生错误放行。", ""]
        lines += ["离线候选门槛：" + ("通过，仍需完整回放和板端验收。" if comparison["offline_gate_passed"]
                                        else "未通过全部条件，继续保留稳定模型。"), "",
                  "候选权重：`" + status["weights"] + "`。", "",
                  "权重SHA256：`" + status["weights_sha256"] + "`。", ""]
        assert digest(status["weights"]) == status["weights_sha256"]
        audit["tasks"][task].update(offline_gate_passed=comparison["offline_gate_passed"],
                                   comparison=comparison, weights=status["weights"],
                                   weights_sha256=status["weights_sha256"])
        if task == "navigation" and (folder / "glyph_regression.json").exists():
            regression = load(folder / "glyph_regression.json")
            audit["tasks"][task]["glyph_regression"] = regression
            lines += ["卡片到现有箭头分类器的同图回归（未包含OCR和动作仲裁）：", "",
                      "| 模型/输入 | 已知方向正确接受 | 错误接受 | 未知输出 |",
                      "|---|---:|---:|---:|"]
            for key, counts in regression["totals"].items():
                lines.append(f"| {key} | {counts['accepted_correct']}/{counts['known']} | "
                             f"{counts['accepted_wrong']} | {counts['unknown_output']} |")
            lines += [""]
    diagnosis = OUT / "historical_green_diagnosis.json"
    lines += ["## 解释与下一阶段", "",
              "红绿灯旧实机90帧通过65帧，25个失败帧中5帧因灯头关系未确认、20帧在重新累计绿灯确认。补训结果必须进一步检查灯头关系和连续确认，不能只用检测P/R宣称闭环问题已解决。", "",
              "导航本轮训练的是提示卡片定位，复用既有人工标签；未训练OCR或新增真实主导航直行来源。下一步还需接入箭头分类、导航观察和统一仲裁。", "",
              "若本轮未通过候选门槛，优先用稳定权重推进代码：回放灯头关系失效的5帧，检查当前图像验证与灯头连续关联；将已有效的导航定位、箭头分类和OCR迁移到隔离的ARM候选。新增训练应针对新标注的实际失败来源。", "",
              "训练/验证保持原角色。增强没有增加独立来源；导航仅9个训练来源帧组、6个验证帧组且同视频相关，无法据此宣称泛化验收。", "",
              "训练入口：`D:/deeplearning/ultralytics-8.3.163/run_signal_navigation_training_pycharm.py`。",
              "详细计划与逐图预测：`" + str(OUT) + "`。", ""]
    if diagnosis.exists():
        audit["historical_green_diagnosis"] = load(diagnosis)
    audit["pipeline_status"] = load(OUT / "pipeline_status.json")
    audit["output_bytes"] = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    (OUT / "training_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    report = ROOT / "红绿灯与导航模型补训结果_2026-10-29.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(report)
    print(json.dumps({k: {x: v for x, v in value.items() if x not in ("comparison", "glyph_regression")}
                      for k, value in audit["tasks"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
