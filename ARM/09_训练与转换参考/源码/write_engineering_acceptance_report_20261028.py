"""Render current engineering evidence to two local Markdown handoff files."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    batch = Path((ROOT / 'runs/engineering_acceptance_20261028/CURRENT.txt').read_text().strip())
    standard = json.loads((batch / 'standard.json').read_text())
    stamp = datetime.now(timezone(timedelta(hours=8))).isoformat()
    rows = []
    for label, name in [('实体KEY6绿行', 'physical_green_result.json'),
                        ('实体KEY6停车', 'physical_stop_result.json'),
                        ('实体KEY6红停', 'physical_red_result.json'),
                        ('实体KEY7强行红灯报警', 'physical_forced_red_result.json'),
                        ('实体KEY6停车并清除报警', 'physical_alarm_clear_result.json'),
                        ('冷启动后实体KEY6绿行', 'cold_physical_green_result.json')]:
        path = batch / name
        if path.exists():
            result = json.loads(path.read_text(encoding='utf8'))
            passed = result.get('passed', result.get('status') in ('PASS', 'PASS_INTERNAL_FUNCTIONAL_GATE'))
            ratio = result.get('ratio', result['successes'] / result['frames'])
            rows.append(f"| {label} | {result['successes']}/{result['frames']} | {ratio:.2%} | "
                        f"{'达到内部门槛' if passed else '未通过'} | {name} |")
        else:
            rows.append(f'| {label} | — | — | 待记录 | {name} |')
    gates = '\n'.join(f"| {g['id']} | {g['name']} | {g['status']} |" for g in standard['gates'])
    cold_path = batch / 'cold_park_smoke_result.json'
    cold = json.loads(cold_path.read_text()) if cold_path.exists() else {}
    report = f'''# 工程验收记录

更新时间：{stamp}。证据目录：`{batch.as_posix()}`。

当前整体工程验收尚未通过。实体按键直行红停、绿行、停车、强行红灯报警和解除报警已达到内部功能门槛；人工辅助冷启动已恢复画面及串口。FPGA完整时序、全链路时延等仍需单独完成。

## 固定版本与评分范围

- FPGA现用0x29（反馈41），PGL50H / ID00521899，bit SHA256 `{standard['bitstream_sha256']}`。
- ARM目录 `/home/orangepi/adas2026/signal_recovered_20261028`；manifest SHA256 `{standard['ARM_manifest_sha256']}`。
- 采集1280×720 / YUYV / 10FPS；HDMI源1920×1080@60；lamp=capture_ft、road=int8、classifiers=batch2、layout=full、geometry proposals。ARM模型和场景阈值保持冻结。
- 每个实体功能用例取按键转换后的连续90帧，内部门槛≥70%，错误放行0。该70%门槛由团队设定，不是赛题官方通过线，也不代表未知场景泛化准确率。
- 实体按键记录要求物理来源、模拟标志清除、新epoch及正确转换；电脑测试报文不充当实体按键或图像识别证明。当前车道是按键选定的虚拟车道。

## 本轮实体功能成绩

| 用例 | 完整检查成功帧 | 比例 | 结果 | 批次内证据 |
|---|---:|---:|---|---|
{chr(10).join(rows)}

各用例保留第一帧按键切换、过期及未确认帧的失败记录，没有剔除失败帧。上述用例均无错误放行。屏幕确认单独保留本人答复；没有把LED4作为独立报警证据。LED4为停车/报警综合灯。

## 程序重启、采集故障和通信保护

- `recovery_20261028_091325_UTC`：同一Linux开机下，ARM PID43956→46999，79项文件校验一致，tiles/raw两模式各连续24帧有效，重启后停车及UART正常。
- 真实终止所属采集子进程：主程序恢复采集并保持停车；故障500ms后观察3帧，无放行；明确发送新行驶请求后取得3帧新的确认GO。该项是软件故障注入；本批次未再次进行物理USB拔插。此前同版真实拔插记录另存。
- `lease_20261028_092520_UTC`：FPGA独占UART台架测试，分别注入age=0/200/400ms的合成有效报文，预期剩余租约500/300/100ms。第一次查询观察到超时的时刻为511.6/309.8/119.9ms，包含查询间隔及串口开销，不是精确硬件边沿测量。超时均STOP、决策无效、报警码4；重放及≥500ms过期报文均被拒绝。
- 合成报文测试只证明协议保护，不计入红绿灯模型成绩或实体按键成绩。

## 实际两板断电后的恢复

- 本人确认香橙派与FPGA均已断电再上电。Linux boot_id由8ef6af96-06a8-4703-a14c-05a1ac8afbf6变为77a7ee75-4fd1-4bf0-82bd-0f33f20be0ad。
- 使用匹配bit重新下载SRAM，核对反馈41；未写Flash。重新校验79项ARM文件。
- 第一次全幅预检失败，保留 `cold_input_preflight_attempt1_failed.json`。随后确认电脑播放器已停止，记录 `cold_source_was_stopped.json`。
- 重新播放原固定绿灯视频后，同次开机中tiles/raw各连续24帧通过，记录 `cold_input_preflight_attempt2.json`。没有改变权重或判定门槛。
- 冷恢复停车检查：{cold.get('successes', '待记录')}/{cold.get('frames', '待记录')}，证据 `cold_park_smoke_result.json`。冷启动后的实体绿行见上表。
- 结果属于人工辅助冷恢复：需要电脑恢复SRAM、重开HDMI视频源并启动ARM。尚未验证脱离电脑自动开机进入演示；不能按上电自启动交付。

## 工程门槛状态

| 编号 | 项目 | 状态 |
|---|---|---|
{gates}

1. 当前0x29完整时序仍有setup/hold/removal违例。离线0x2d仅修复像素域setup，未上板、其余时序未闭合，不计作本版本验收通过。
2. 本批次物理USB再次拔插、其余转道/导航/风险场景在当前统一版本上的回归仍需按各自场景留证。旧0x27转道成绩不能转给0x29。
3. 已有解码帧龄不等于摄像/HDMI曝光至显示的端到端延迟。720p YUYV10协商格式对应原始像素负载约147.456Mb/s，这是计算值，不是实测总线吞吐。
4. 后续整改版本必须单独保存版本、bit/模型哈希、时序报告及实机成绩。当前各用例及首次失败记录全部保留。

## 复现顺序

正常关机并按板卡规范断电→两板上电→电脑恢复匹配的0x29 SRAM→重新全屏播放HDMI固定视频并核实playing→ARM文件校验及两种采集模式各24帧预检→启动ARM并停车→按键功能复验。电脑视频源停止时先恢复视频源，再检查采集，不能使用旧status.json判断程序已运行。

详细机器记录以同目录 `standard.json`、各用例 `observations.jsonl`、恢复及冷启动记录为准。内部阶段通过和整个赛题完成分别记录。
'''
    Path('D:/工程验收记录_2026-10-28.md').write_text(report, encoding='utf8')
    handoff = f'''# 发给队友：工程验收结果与FPGA待办

更新：{stamp}。完整记录：`D:/工程验收记录_2026-10-28.md`。

当前实机仍为FPGA0x29/41。ARM模型冻结，实体直行红停、绿行、KEY6停车、KEY7强行报警及KEY6清除报警已达到内部70%门槛，错误放行0。本人已逐项确认屏幕和UART。冷启动视频源曾停止，重新播放后恢复；不是上电自动演示。

## FPGA优先工作

1. 处理完整时序违例并补齐实际接口约束。现用工程 `D:/PDS_WORK/adas_fullres_20261028`；最终报告 `report_timing/test_ddr.rtr`。当前像素setup WNS=-1.181ns，TNS=-14.048ns；另有ref/DDR/CMOS跨域hold和ref removal问题。
2. 已保留0x2d流水修复候选：`D:/PDS_WORK/adas_mean_retimed_20261028/adas_competition.pds`，像素setup +0.409ns/TNS0，但仍有hold/removal，尚未下载。候选包 `D:/FPGA均值滤波时序修复_0x2d_待实机验收.zip`；说明 `D:/FPGA下一轮_时序修复与实体按键验收_2026-10-28.md`。其中旧的“实体按键待测”进度以本记录为准。
3. 核对匹配板卡的时钟、复位释放与CDC结构，以及ref_clk重复IO标准定义。按真实路径整改，保留有效跨域时序，不能任意加false path消除报表。
4. 新固件版本需同步ARM版本白名单、校验新反馈及重新验证所有功能。现用ARM要求反馈41，0x2d=45不能直接替换。
5. 确认Flash型号/容量、板卡修订和启动拨码，再制定掉电启动方案。当前只下载SRAM，未写Flash，冷启动仍需要电脑。

## 共同联调与交接

- ARM负责保持当前模型/阈值/用例冻结，补充当前统一版本转道、导航及风险场景，测量端到端延迟；FPGA负责时序闭合、匹配的约束及掉电恢复方案。
- 修复版交接需含完整工程、最终setup/hold/recovery/removal报告、bit及SHA256、版本号、接口变更和烧录方式。每个版本独立验收，保留0x29回退。
- LED4是STOP或ALARM综合指示灯，普通红停ALARM0与强行红停ALARM1都可能亮。报警以屏幕数字和FPGA报警码区分：强行红灯=2，通信超时=4。
- 本次新增了真实实体按键、软件采集故障恢复、独立租约/重放/过期保护和实际断电后的辅助恢复证据。全工程验收仍未通过；70%为团队功能门槛，不能覆盖时序或未知场景问题。

证据目录：`{batch.as_posix()}`。
'''
    Path('D:/发给队友_工程验收结果与FPGA待办_2026-10-28.md').write_text(handoff, encoding='utf8')
    print(json.dumps(dict(report='D:/工程验收记录_2026-10-28.md', handoff='D:/发给队友_工程验收结果与FPGA待办_2026-10-28.md'), ensure_ascii=False))


if __name__ == '__main__':
    main()
