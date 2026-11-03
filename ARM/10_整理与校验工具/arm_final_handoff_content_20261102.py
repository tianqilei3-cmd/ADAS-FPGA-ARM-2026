"""Evidence-grounded documents, editable diagrams and scenario catalog."""
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import re
import textwrap


def put(root, name, text):
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(text).strip() + '\n', encoding='utf-8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(x).replace('|', '/') for x in row) + ' |' for row in rows])


def write_materials(dest, original, engineering_path, copy):
    from recompute_arm_handoff_metrics_20261102 import compute
    results = compute(dest)
    metrics_dir = dest / '06_测试结果与统计'
    (metrics_dir / '指标复算结果.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    perf = results['current_hdmi_short_test']
    cases = results['physical_cases']
    with (metrics_dir / '实体用例成绩.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
        writer.writeheader(); writer.writerows(cases)
    case_table = table(['实体功能用例', '成功帧/总帧', '比例', '错误放行记录'],
                       [[r['case'], f"{r['successes']}/{r['frames']}", f"{r['ratio']:.2%}", r['unsafe_frames']] for r in cases])
    age = perf['final_age_ms']
    observations = read(original / 'runs/restore_hdmi_live_20261101/steady_check.json')['rows']
    real_wire = next(r for r in observations if r.get('receipt') and
                     any(t.get('wire_hex') and t.get('feedback') for t in r['receipt']['transactions']))
    wire_record = dict(source='HDMI恢复_20261101/steady_check.json',
                       sequence=real_wire['sequence'], identity=real_wire['identity'],
                       source_mode=real_wire['source_mode'], key_snapshot=real_wire['key_snapshot'],
                       context=real_wire['context'], receipt=real_wire['receipt'],
                       feedback=real_wire['feedback'])
    wire_dir = dest / '05_通信协议与真实日志';wire_dir.mkdir(exist_ok=True)
    (wire_dir / '真实收发记录_摘自HDMI实机.json').write_text(json.dumps(wire_record, ensure_ascii=False, indent=2), encoding='utf-8')
    tx = next(t for t in real_wire['receipt']['transactions'] if t.get('wire_hex'))
    raw = ['# 2026-11-01 HDMI实时实机记录摘录',
           '# 时钟为香橙派 time.monotonic 秒；不是北京时间或传感器时间。',
           '# 原始JSON和SHA可追溯；没有用脚本构造的测试包代替实测。',
           f"# sequence={real_wire['sequence']} / source_mode={real_wire['source_mode']}",
           f"TX t={tx['tx_started_monotonic']:.9f}  {tx['wire_hex']}"]
    for chunk in tx['reply_reads']:
        raw.append(f"RX t={chunk['observed_monotonic']:.9f}  {chunk['hex']}")
    raw += ['# 0x84状态由0x13 ACK之后的0x04查询取得；查询写出时刻未单独记录，不伪造其时间。',
            '# decoded_feedback=' + json.dumps(tx['feedback'], ensure_ascii=False),
            '# acknowledged=' + str(tx['acknowledged'])]
    put(dest, '05_通信协议与真实日志/真实收发日志.txt', '\n'.join(raw))
    add_model_inventory(dest)
    add_evidence(dest, original, copy)
    draw_diagrams(dest)

    put(dest, 'README_先看.md', '''
    # ARM 最终交接

    整理日期：2026年11月2日。给负责合并比赛技术报告、源码及演示材料的队友。

    交付基线是实际演示使用的 `signal_recovered_20261028`：完整ARM源码、15个RKNN/ONNX模型、原配置和桌面启动器均已收入，保留原文件哈希。2026年11月1日恢复HDMI实时采集，现场效果得到本人确认。本次整理没有连接、改写板卡或训练模型；历史记录不表示板卡此刻仍在运行。

    ## 队友先看这四份材料

    1. `03_技术文档/ARM技术说明_供合并比赛报告.md`：可按官方模板拆入方案、软件设计、测试与结论章节。
    2. `03_技术文档/功能验证矩阵.md`：区分实板联动、ARM独立测试、PC测试及未验证功能。
    3. `README_从开机到运行.md`：接线、冷启动、逐条命令、停止和排错。
    4. `08_来源与许可证/数据权重第三方与师兄代码来源.md`：来源、已知授权信息及待队友补齐项。

    可直接打开 `index.html` 查看目录和图表，不需要YOLO环境。Markdown正文、CSV、JSON和SVG都可以编辑。当前没有已确认的比赛Word模板，因此未把材料伪称为已完成官方模板排版的最终报告。

    ## 目录与七项需求

    | 需求 | 位置 |
    | --- | --- |
    | 实际演示源码、模型和配置 | `01_ARM实机程序/runtime`；桌面工具位于 `01_ARM实机程序/launcher` |
    | 从开机到运行 | 本目录 `README_从开机到运行.md` |
    | ARM模块图与视频/检测数据流 | `04_模块图与数据流`，含PNG、SVG和Mermaid源文件 |
    | 通信协议和真实日志 | `05_通信协议与真实日志` |
    | 性能、识别指标与统计方法 | `06_测试结果与统计`，含原始记录、复算脚本及CSV |
    | 各场景图片与证据 | `07_场景证据`，逐图标注版本、输入方式、验证范围 |
    | 数据、权重、第三方及师兄来源 | `08_来源与许可证`，模型来源清单及许可证原文 |

    `09_训练与转换参考`额外收录源训练权重、导出/转换/采样评分程序及静态发现的本项目依赖、Ultralytics源码。这些是追溯参考，含原工程路径，不能在缺少训练数据与原环境时直接复训；其中部分旧部署脚本手动运行会连接板卡。开机演示只使用 `01_ARM实机程序`，不会执行这些归档脚本。

    `01_ARM实机程序`内的旧README和reference是原运行包的一部分，为保持原校验清单未改写。当前总体状态以本目录README和 `03_技术文档`为准，不能把旧说明中的“未启动”或历史候选状态当作最新结论。

    ## 已完成范围和提交时的表述

    已有实板证据：真实HDMI输入到香橙派推理、UART回包；实体KEY6停车/绿行/红停；实体KEY7强行红灯报警；停车清警；人工辅助冷恢复。闭环场景使用按键指定的虚拟车道和驾驶意图，未连接实际车辆执行机构。

    不能写成已完成：纯图像自动定位真实本车道并关联灯头、导航接入当前ARM统一程序、当前版本全部变道/风险场景验收、脱离电脑自动冷启动、完整硬件时序收敛、传感器到显示器的端到端延迟验收。11月1日截图仍有画面过期提示；现场接受不替代全工程验收。

    ## 校验

    在有Python 3的电脑或香橙派上，仅校验文件，不启动模型、不打开设备：

    ```sh
    python3 verify_arm_final_handoff.py
    ```

    Windows若只有python命令，使用 `python verify_arm_final_handoff.py`。ZIP另附SHA256文件。解压后不要把生成的临时文件放入本包再按“无多余文件”模式校验。

    当前目录是ARM侧提交材料。匹配的FPGA参考bit已保留在原运行包reference；完整FPGA工程、引脚约束、时序报告、烧录与板端成果由队友合并，不要用这几个接口RTL摘录代替完整FPGA工程。
    ''')

    put(dest, 'README_从开机到运行.md', r'''
    # 从开机到运行

    本流程针对已安装运行环境的Orange Pi 3B RK3566和盘古PGL50H板卡，以orangepi普通用户操作。队友电脑不需要YOLO或训练环境。ARM推理需要板上的RKNNLite、NPU驱动、OpenCV、NumPy和ONNX Runtime；本包不是操作系统镜像。

    ## 一 接线与上电

    ```text
    电脑HDMI输出 → 盘古HDMI IN
    盘古HDMI OUT → MS2130 HDMI输入
    MS2130 USB → 香橙派USB3
    香橙派HDMI输出 → 独立显示器
    香橙派USB → 盘古CP2102 USB-UART
    电脑USB → USB Cable II下载器 → 盘古JTAG
    ```

    HDMI视频采集卡、CP2102串口和JTAG下载器是三类设备。JTAG下载器不能替代视频采集卡。两块板卡按各自电源要求独立供电，香橙派登录Linux桌面。

    ## 二 冷启动后的FPGA准备

    本ARM基线要求 `0x29` 固件，状态回传 `build_tag=41`。匹配bit在 `01_ARM实机程序/reference/FPGA_0x29_test_ddr.sbit`。

    ```text
    SHA256 84b4c9fd406f27e21aacd5959509004993b22e1d801c9e4f29f9db09e4ee6f06
    器件 PGL50H-6IFBG484
    JTAG ID 0x00521899
    ```

    若断电后SRAM配置丢失，由FPGA负责人在匹配PDS中打开Configuration、扫描并确认目标器件、分配该bit、下载到SRAM。等下载完成后再启动ARM。这里没有写Flash的命令；无人值守上电启动尚未交付。队友使用其他固件时先核对协议，不能只看LED亮就认为版本匹配。

    ## 三 电脑播放真实HDMI视频源

    解压后进入本包根目录。已有可用Python/OpenCV环境时可以直接播放；首次在普通Windows电脑准备独立播放环境，可逐条运行：

    ```powershell
    py -3 -m venv .playback_env
    .\.playback_env\Scripts\python.exe -m pip install numpy opencv-python
    .\.playback_env\Scripts\python.exe .\02_电脑视频源\play_official_hdmi_source_20261101.py --video .\02_电脑视频源\示例输入\video_20260929_113343.mp4
    ```

    这是电脑的视频播放环境，不安装或训练YOLO。虚拟环境会增加文件，应在原包外复制播放器和示例输入后创建，或在完成文件校验后再操作。

    本机原有yolo解释器也可以使用，先在PowerShell进入解压目录，再运行：

    ```powershell
    & 'D:/anacade/envs/yolo/python.exe' -X utf8 .\02_电脑视频源\play_official_hdmi_source_20261101.py --video .\02_电脑视频源\示例输入\video_20260929_113343.mp4
    ```

    播放器优先选择唯一活动外接屏；没有独立外接区域时使用主屏。11月1日采用后一种方式，已用计时标记实测画面传到了采集端。主屏会被视频占用，视频窗口按Q或Esc退出；退出后输入可能变为电脑桌面。不要重复启动播放器。

    后续正式时序验收建议由FPGA负责人确认真实HDMI输入1920×1080/60Hz，不能用电脑逻辑主屏的2560×1600/120Hz数字代替测量。原扩展屏播放器也一并保留，但在无独立活动外接屏时会拒绝启动。

    ## 四 香橙派已有程序时直接运行

    在香橙派Linux终端逐条运行：

    ```bash
    whoami
    ip -4 -br addr
    lsusb
    ls -l /dev/video0 /dev/ttyUSB0
    df -h /home/orangepi
    python3 /home/orangepi/.local/share/adas2026/desktop_20261101/adas_launcher.py check
    python3 /home/orangepi/.local/share/adas2026/desktop_20261101/adas_launcher.py start
    ```

    应看到MS2130 `345f:2130`和CP2102 `10c4:ea60`。`check`仅检查依赖、文件和设备可访问性，不打开采集/串口，也不能替代实际画面验证。

    `start`先校验源码与模型，然后停车并等待tiles/raw两个模式分别24个连续有效输入帧，再进入实时窗口。已有演示运行时先正常停止，不能重复启动。也可双击桌面 **ADAS 道路演示**。

    仅通过SSH启动时：电脑和香橙派连接同一网络，先用自己的账号建立正常登录。历史IP为172.20.10.6，换网络以板端当前IP为准：

    ```powershell
    ssh orangepi@172.20.10.6
    ```

    登录后执行上面的Linux命令。画面出现在香橙派独立显示器上；SSH终端本身不传回视频。本包没有密码、私钥或账号token。

    ## 五 运行目录缺失时从本包恢复

    优先将本包放U盘，避免重复占用香橙派存储。假设U盘解压目录为 `/media/orangepi/USB/ARM_最终交接`，按实际位置修改第一行：

    ```bash
    cd '/media/orangepi/USB/ARM_最终交接'
    python3 verify_arm_final_handoff.py
    cd 01_ARM实机程序
    python3 verify_package.py
    sh INSTALL_ON_ORANGEPI.sh
    python3 /home/orangepi/.local/share/adas2026/desktop_20261101/adas_launcher.py check
    python3 /home/orangepi/.local/share/adas2026/desktop_20261101/adas_launcher.py start
    ```

    不用sudo运行安装器。只有固定运行目录不存在时才会恢复；已存在但哈希不同则停止，保留队友修改，不自动覆盖。运行目录是 `/home/orangepi/adas2026/signal_recovered_20261028`。

    已测板端环境：Linux 5.10.160 rockchip、OpenCV 4.11.0、NumPy 1.26.4、ONNX Runtime 1.23.2、RKNNLite 2.3.2、NPU驱动0.9.6。Python的完整补丁版本未在本次整理中重新采集，不伪造锁定值。重刷系统需要先准备匹配镜像和NPU驱动，不建议临赛直接全量升级依赖。

    ## 六 屏幕和按键检查

    确认道路画面更新、UART确认、第二行驾驶请求与实际按键一致。先停车，避免把驾驶请求go与第一行判定GO混为一谈。

    1. KEY2切场景：signal、boundary、导航保留场景、pedestrian；导航在本ARM基线中未实现，应保持保守停车。
    2. KEY3/KEY4/KEY5选择左/直/右意图；这是按键选定的虚拟车道方向。
    3. KEY6切换行驶请求/停车；KEY7发出强行请求，停车应清除强行状态。
    4. 绿灯且当前图像、灯头与驾驶状态连续确认后允许GO；红灯请求go仍应STOP；红灯强行请求应STOP并ALARM1。
    5. 界面ALARM1是布尔报警，FPGA反馈的 `alarm_code` 是另一组枚举。LED4是停车或报警综合灯，不能单凭亮灭区分两者。

    无真实电机、舵机或车辆控制器接入，不能将GO字段当作车辆已安全通行。

    ## 七 停止和查看日志

    在香橙派窗口按Q/Esc，或执行：

    ```bash
    python3 /home/orangepi/.local/share/adas2026/desktop_20261101/adas_launcher.py stop
    cat /home/orangepi/adas2026/signal_recovered_20261028/last_session.txt
    ```

    将第二条输出的会话路径代入：

    ```bash
    cat '<会话路径>/status.json'
    cat '<会话路径>/summary.json'
    ```

    `stopped`且错误为空表示已停止，不能用旧status假装仍在运行。`state=running`也需结合updated_unix和帧号增长判断。

    11月1日运行时可用空间仅约577MiB；低于512MiB会保护退出。长期演示前先归档已结束会话的日志，不删模型或当前打开的日志。该包没有自动清理用户文件的操作。

    ## 八 常见问题定位顺序

    - 没有 `/dev/video0`：先核对真正的HDMI采集卡和USB数据线、USB3端口。JTAG下载器不是视频设备。
    - 有设备但全黑：先确认电脑在正确HDMI源上持续播放，再核对FPGA输入、输出和采集卡HDMI输入；黑帧不算识别失败样本。
    - UART无回包：检查CP2102是否为 `/dev/ttyUSB0`、固件0x29及是否有SSCOM等其他程序占用。运行中不要另开串口探测器争用。
    - `displayed image expired`：现有推理时效超过显示窗口时保守停车；保留原阈值，不用刷新时间戳或提高时效阈值掩盖。当前资料已记为待优化项。
    - 双击本地视频演示：那是历史备用入口，不能用于验收HDMI实时采集。本包默认入口只读真实 `/dev/video0`。
    ''')

    matrix = [
        ['HDMI→MS2130→ARM实时推理→显示', '实板视频链已验证', '2026-11-01用户接受；1160帧后停止；显示过期提示仍待优化', 'HDMI恢复_20261101'],
        ['实体KEY6绿行/停车/红停', 'ARM+FPGA已联动', '控制直行片段、虚拟车道；每项连续90帧', '工程验收_20261028及实体按键_20261028'],
        ['实体KEY7强行红灯报警和停车清警', 'ARM+FPGA已联动', 'ARM判定、带epoch的FPGA回包及人工屏幕确认；LED4为综合灯', 'physical_forced_red_result.json等'],
        ['人工辅助冷恢复', '两板断电恢复已验证', '需重下SRAM、重播PC视频并启动ARM；不是无人值守自启', 'cold_*记录及首次失败记录'],
        ['车道虚实线与转道', '旧版本联板示例；当前仅部分运行证据', '0x27受控转道示例不能转为0x29完整验收；11月1日boundary能推理回包', '旧版变道示例及HDMI恢复记录'],
        ['车辆/两轮车/行人检测', 'ARM真实输入推理和UART已运行', '不等于各危险场景均已统计验收；检测精度有独立PC报告', '实时画面、road PC指标'],
        ['道路模型INT8转换一致性', 'ARM单独推理测试', '与原模型输出对照，不是人工真值准确率或FPGA功能验收', 'road_int8/native_parity.json'],
        ['灯型/方向ONNX导出一致性', 'PC数值检查；运行模型被ARM调用', '60个已知样本图，不能代替灯头关联真值测试', 'export_verification.json'],
        ['导航卡片和方向', '仅PC测试', '6个验证来源帧组、4个卡片真值；当前ARM导航字段固定0，未接入统一程序', '导航PC离线_20261029'],
        ['纯图像自动真实本车道—灯头关联', '未达到原目标；未按成品交付', '当前为按键选择虚拟车道和图像候选灯规则，不是实路几何归属保证', 'runtime策略及历史说明'],
        ['物理车辆执行机构', '未接入/未验证', '无电机或舵机，STOP/GO为逻辑结果及板上指示', 'hardware_motion_control_connected=False'],
        ['FPGA完整时序及端到端延迟', '未通过/待测', '0x29有时序违例；没有从曝光/HDMI源到实际显示的同步测量', '工程验收standard.json E09/E10'],
    ]
    matrix_md = table(['功能', '验证层级', '范围及限制', '证据索引'], matrix)
    put(dest, '03_技术文档/功能验证矩阵.md', '# 功能验证矩阵\n\n证据冻结到2026年11月1日；本次仅整理资料，不进行新实测。\n\n' + matrix_md + '''

“实板联动”指真实板卡处理及串口确认，场景可能仍使用按键设定的虚拟驾驶意图；“ARM单独”是板上离线推理或数值对照；“PC测试”不能算香橙派成果。内部70%是团队阶段门槛，不是赛题官方统一合格线。阶段功能通过不能平均抵消FPGA时序或端到端工程项目的缺失。
''')
    with (dest / '03_技术文档/功能验证矩阵.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream);writer.writerow(['功能','验证层级','范围限制','证据']);writer.writerows(matrix)

    report = f'''
# 基于Orange Pi 3B的道路视觉识别与FPGA协同系统ARM技术说明

整理日期为2026年11月2日。本材料供队友合并紫光同创赛题三技术报告。实现说明以已演示的ARM冻结源码为依据，实验成绩均对应已有记录，不把后续候选、PC测试或人工观测扩展成整机验收结论。

## 一 目标与已实现范围

我们在Orange Pi 3B的RK3566处理器上完成道路视频取帧、目标识别、信号灯与边界场景判定、结果显示和USB-UART通信。FPGA提供视频链路、实体按键、结果接收与保护判定及板上指示。当前已留存的实体功能用例包括停车、直行绿行、红停、强行红灯报警与停车清警。

当前车道由按键选择为虚拟左转、直行或右转意图，图像模块再判断候选灯组、灯色/灯型和道路风险。本系统没有完成纯图像真实本车道几何定位与灯头物理归属验证，也没有接入实际车辆的电机或舵机。导航仅有PC阶段材料，未接入当前ARM统一入口。

## 二 硬件及系统结构

ARM平台为Orange Pi 3B RK3566，现场板载内存约8GiB；FPGA器件为PGL50H-6IFBG484。MS2130作为HDMI到USB3的视频采集设备，CP2102用于115200波特率串口。显示器接香橙派HDMI输出，电脑承担官方视频播放及FPGA下载，不承担本轮ARM图像推理。

![ARM模块图](../04_模块图与数据流/ARM模块图.png)

![硬件与数据流](../04_模块图与数据流/视频与检测结果数据流.png)

视频由电脑HDMI送入FPGA，再由FPGA HDMI输出进入MS2130，MS2130通过UVC向香橙派提供图像。ARM检测结果通过另一条USB-UART链路回到FPGA。两条路径独立：视频不经过串口，完整检测画面也不通过UART回传。当前识别窗口来自香橙派本机绘制，不能把它说成已经在FPGA端实现全部OSD叠加。

## 三 ARM软件模块与实现

### 3.1 当前实际版本

入口为 `run_unified_demo.py`，板上固定目录为 `/home/orangepi/adas2026/signal_recovered_20261028`。完整源码、15个模型和配置位于本包 `01_ARM实机程序/runtime`。冻结清单SHA256为 `09c4795d9457f6aff117643234edb6c7fcdd12395a5d7268ec11c32aba0005dd`；实际固件为0x29，反馈build_tag=41。运行参数为720p、YUYV、10 FPS，灯模型capture_ft、道路模型int8、分类器batch2、full布局、灯壳细化及几何提议。

`light_metadata.json`保留了原阶段来源；实际capture_ft参数会覆盖灯模型文件名，使用 `lamp_aligned_fp_heads.rknn`。不能仅按历史metadata或导出时的adopted=false字段推断运行权重；本包以运行源码、实测参数和文件哈希共同锁定版本。未用10月29日新候选覆盖原稳定模型。

### 3.2 采集与图像身份

`LatestCaptureReader`独立子进程持续读取 `/dev/video0`，使用有界共享缓冲保留最新图像。最终推理取全幅1280×720输入，预检中的tiles模式仅用于重新协商和确认输入，不是最终两路摄像头。输入异常、无新帧或显著均匀画面不能建立通行许可。

每帧携带相机ID、主机帧号、图像哈希和解码完成时间。解码完成时间属于用户态时钟，不是相机曝光时间或FPGA像素采样时间；尚未证明FPGA视频帧与ARM帧号一一对应。程序保留这项限制，不通过重设旧帧时间戳规避过期检查。

### 3.3 道路目标与场景识别

道路四分类模型输入为384×640，类别为交通灯、车辆、两轮车和行人。FP32训练权重经ONNX裸输出头导出和INT8 RKNN转换，在NPU运行；CPU负责解码和NMS。道路目标给风险模块提供当前图像中的候选框，不单凭“有行人”即宣称碰撞或所有场景已验收。

信号场景采用矩形灯检测、颜色发光检查、灯型/方向判断、几何灯组提议及连续性规则。当前有效绿灯需连续确认，未知灯、关联不明或时效不足采取保守停车。检测到红/绿颜色只是中间结果，不等于完成真实车道—灯头归属。

边界场景使用512×288的类型边界分割模型，类别为背景、实线边界和虚线边界。它预测的是连续的类型边界走廊，不能用掩码是否连续判断虚实；后处理再提取实例、图像中心左右边界和可用线段。当前版本对变道功能的实板完整回归仍待补齐。

### 3.4 按键与仲裁

实体KEY2选择场景，KEY3/4/5选择左/直/右，KEY6切换行驶请求与停车，KEY7提出强行请求。按键返回state和epoch，ARM在推理前后核对状态；发生按键变化时等待新图像重新判定，旧结果不能冒用新驾驶请求。

仲裁由当前图像结果、驾驶意图、图像质量、时效和风险状态共同决定。停车请求可限制运动，红灯强行请求保持STOP并报警。日志中的simulation_only、virtual_lane及hardware_motion_control_connected=False表示虚拟驾驶场景和无真实车辆执行机构；真实运行位置应以source_mode=live_capture、采集进程和板端模型记录判断。

### 3.5 通信与显示

串口逐包发送、校验ACK并查询带相机/帧号/epoch的FPGA状态，避免只凭三字节ACK认定当前动作已确认。图像时效在发送边界再次编码，FPGA按剩余租约判断过期/重放并停车。界面显示ARM动作、布尔报警、驾驶请求及UART确认状态；旧画面超过500 ms会覆盖过期提示。

实测显示提示仍可能出现；本次不提高阈值或刷新时间戳掩盖该问题。LED4为停车或报警综合灯，ALARM布尔值与FPGA alarm_code枚举分别说明。

## 四 实验设计与结果

### 4.1 实体闭环功能

以0x29固件及本包ARM哈希固定版本进行控制场景测试。每个用例从实体按键转换后连续取90帧，保留首帧切换、过期及未确认失败，不挑选成功段。每帧需同时满足有效输入、时效、按键一致、图像决策、对应相机/帧号/epoch的FPGA确认及预期动作/报警。团队阶段门槛为成功比例至少70%且错误放行记录为0，不是官方统一通过线。

{case_table}

这些是受控片段的闭环检查比例，不是神经网络精确率或实路安全保证。相邻帧有强相关性，没有据此计算独立道路成功概率。实体按键和人工屏幕答复另有记录，不把远程测试驾驶报文当作实体按键证明。

### 4.2 最新HDMI实时恢复短测

2026年11月1日重新识别MS2130，用电脑跳动计时标记核实输入确在更新，再通过原停车预检恢复实时运行。约{perf['elapsed_seconds']:.2f}秒处理{perf['frames']}帧，流水线吞吐为{perf['inference_pipeline_fps']:.2f}帧/秒；状态发送确认{perf['state_acknowledged']}/{perf['rows']}，输入质量有效{perf['input_quality_valid']}/{perf['rows']}，采集重连为0。该段场景为{', '.join(perf['scene_modes'])}，不作为所有模型/场景的统一帧率。

从解码完成到结果准备的图像年龄：均值{age['mean']:.1f} ms、中位数{age['median']:.1f} ms、P95 {age['p95']:.1f} ms，范围{age['minimum']:.1f}–{age['maximum']:.1f} ms。这不是曝光/HDMI源到实际显示的端到端延迟。采集协商为10 FPS，也不等于模型推理10 FPS。

该次会话最后已处理1160帧后进入stopped，错误列表为空；用户接受现场采集效果。没有把正常停止误报为崩溃，也没有声明已证明停止触发方式。截图保留了displayed image expired提示，因此不能宣称显示时效已全部验收。

### 4.3 离线识别指标

对应当前灯模型源权重的PC评估，在已知开发验证画面上以conf=0.25、NMS IoU=0.7、匹配IoU=0.5统计：200个真值、164个预测，TP=148、FP=16、FN=52，P=90.24%、R=74.00%、F1=81.32%。绿灯正确检出75/96；这是检测诊断，不是ARM闭环绿行比例。相关退化视图P=88.41%、R=72.50%，增强视图不增加独立道路来源。

道路四分类源模型在1000张BDD开发验证图上的历史PC报告：P=42.88%、R=33.28%、mAP50=32.65%、mAP50–95=15.74%。不能略去较弱类别或把转换输出一致率当真值准确率；当前INT8部署模型未另做完整人工真值精度复测。

RKNN道路INT8与原输出对照的42个样例，匹配输出325/参考340，匹配输出325/候选335，说明转换一致性；没有独立人工真值，不能写成“准确率97%”。导航4/4卡片结果仅为PC极小相关验证集，未算入ARM或整机成绩。

## 五 资源与工程限制

采集1280×720 YUYV10的名义像素负载为147.456 Mb/s，不包含USB开销，也不是实测总线吞吐。串口为115200、8N1，无硬件流控，传送结构化检测/状态而非图像。单个上下文20字节、单个框20字节，当前入口限制每帧最多1个框；坐标归一到1920×1080协议画布。

已用独立采集进程、有界最新帧缓冲、单NPU所有者、单后处理任务和日志轮换控制资源。11月1日主进程快照RSS约234 MiB，但不包括所有子进程、驱动或NPU保留内存，不是总内存峰值；板端可用存储约577MiB，低于512MiB会保护停止。

人工辅助冷恢复已完成，仍需电脑恢复SRAM和视频源。FPGA0x29完整setup/hold/removal时序仍有违例，传感器到屏幕端到端延迟未实测。上述未完成项必须由最终合并报告保留，不能用受控功能成绩抵消。

## 六 来源与复现

道路与类型边界使用BDD100K相关数据；灯色使用S²TLD及赛题官方参考视频裁剪，包含AI辅助复核标注。YOLO架构及预训练初始化来自Ultralytics；RKNN转换/运行来自Rockchip SDK；OpenCV、NumPy和ONNX Runtime等提供处理与推理支撑。训练权重、运行模型、导出哈希、许可证和来源链接已另附。

往年师兄资料作为项目架构与接线参考，原2025资料目录当前不可读，本轮不能完成逐文件复用比例比对；不能据此把代码整体宣称完全原创或全部归为师兄代码。请队友补齐具体复用文件、修改人及授权说明。

从开机到运行见根目录README；统计可在无YOLO、无板卡条件下运行 `06_测试结果与统计/recompute_metrics.py` 重算。完整模型、配置、图像证据和真实串口收发日志随包交付。比赛模板拿到后由队友将本说明分别放入总体方案、软件实现、接口、测试和参考来源章节。
'''
    put(dest, '03_技术文档/ARM技术说明_供合并比赛报告.md', report)

    put(dest, '03_技术文档/队友合并清单.md', '''
    # 队友合并清单

    本次没有已确认的官方Word模板。正文、图、表、代码和证据按内容准备，下面是合并方向，不冒充比赛模板规定的章节名称。

    | 报告内容 | 可直接取用的材料 |
    | --- | --- |
    | 作品目标和系统方案 | ARM技术说明第一、二节，两个模块图 |
    | ARM软件和算法 | ARM技术说明第三节，实际源码和模型清单 |
    | ARM与FPGA接口 | 通信协议、完整字段表、真实收发JSON与TXT |
    | 性能和功能实验 | 技术说明第四、五节，指标CSV、统计方法、原始JSONL |
    | 演示截图 | 场景目录及逐图说明；保留远程测试、版本和过期提示标签 |
    | 参考资料和复用说明 | 来源文档、模型追溯表、许可文本、师兄复用待补项 |
    | 源码附件 | 01运行程序、02视频源、09训练转换参考、校验清单 |

    队友需要补：官方模板与封面信息、队名/成员和明确分工、FPGA完整工程及约束和时序结论、引脚/电源与硬件实物照片、正式演示录像、具体师兄复用文件及授权、尚缺场景的当前版本实测。已有截图中有些使用远程驾驶请求；实体按键通过由独立JSONL与人工答复支持，不要混称。

    合并时不要写“全场景100%”“完全自主车道关联”“上电自动运行”“已闭合全部FPGA时序”或“端到端延迟约370ms”。对应证据尚不支持这些表述。
    ''')

    performance_doc = f'''# 性能指标与统计方法

数据冻结到2026年11月1日，11月2日仅依据随包原始记录复算。全流程不连接串口、采集卡或运行神经网络。

## 分辨率与帧率

| 项目 | 数值 | 统计口径 |
| --- | --- | --- |
| UVC采集 | 1280×720、YUYV、协商10 FPS | 驱动协商结果；未作为独立硬件帧率计数 |
| 道路网络输入 | NCHW 1×3×384×640 | 实际导出网络形状 |
| 矩形灯网络输入 | NCHW 1×3×256×640 | capture_ft矩形头 |
| 类型边界输入 | 宽512×高288 | 转换导出记录 |
| 颜色/灯型批分类 | batch2、128×128 | 固定批量NPU模型 |
| 显示画布 | 1280×720 | GUI内部绘图尺寸，不是显示器物理时序 |
| 协议框坐标 | 1920×1080 | 按当前输入宽高缩放到协议画布 |
| 最新短测吞吐 | {perf['inference_pipeline_fps']:.4f} 帧/秒 | {perf['frames']}个完成结果 / {perf['elapsed_seconds']:.6f}秒，boundary场景 |

11月1日电脑逻辑显示为2560×1600/120Hz，只有一个活动主屏；新播放器没有改显示模式。计时标记证明画面到达采集端，但没有测量实际HDMI电气时序，因此不把电脑逻辑数字填写为FPGA已通过的输入时序。10月28日工程记录另有1920×1080/60Hz源设置，两次实验分开记录。

## 时延

从用户态解码完成的单调时钟 `decoded_at` 到结果准备/绘制前后的 `final_age_ms`：均值{age['mean']:.3f} ms、中位数{age['median']:.3f} ms、P95 {age['p95']:.3f} ms、最小{age['minimum']:.3f} ms、最大{age['maximum']:.3f} ms。P95采用升序第ceil(0.95N)个样本，不做插值。

这里包含已记录的图像年龄及处理开销，不含已知的传感器曝光、完整HDMI传输、采集卡缓存、显示扫描时延。没有同步硬件标记，不给出端到端延迟。摄像源帧号也未与FPGA像素帧号完成硬件对应证明。

分项均值见 `指标复算结果.json` 的profile_mean_ms。若模块有重叠执行，分项不能随意求和作为另一项端到端测量。

## 短测通信与图像有效率

{perf['state_acknowledged']}/{perf['rows']}次状态发送收到当前身份的确认；{perf['input_quality_valid']}/{perf['rows']}帧通过输入质量检查，采集重连{perf['capture_recoveries']}。输入质量检查是空间内容和噪声门槛，不是真值识别正确率。ACK比例也不是目标检测精度。

## 实体闭环功能

{case_table}

每项取按键转换后的连续90帧，首帧转换和未确认均保留。成功需要全部检查通过；功能分母为固定90，而非只计算接受输出的帧数。报警清除还要求ARM报警false、FPGA alarm_code=0及forced=false。没有剔除失败帧，没有将某项70%门槛当成官方统一合格线。

同一视频连续帧有相关性；这些比例仅描述固定片段。错误放行是评分器定义的违规集合为空，不能推导未知道路的误放行风险为零。

## 灯色检测与道路检测

灯色检测原阈值conf=0.25、NMS IoU=0.7、真值匹配IoU=0.5。一对一匹配后P=TP/(TP+FP)、R=TP/(TP+FN)、F1=2TP/(2TP+FP+FN)。clean为TP148、FP16、FN52；退化为TP145、FP19、FN55。原验证104个裁剪图、200个真值；图像存在同视频及增强相关，不能按104个独立路口计数。

道路模型历史PC验证为1000张BDD图，conf扫描低阈值0.001，NMS IoU0.7；P/R按Ultralytics验证程序选取工作点，mAP50和mAP50–95是PR曲线下面积，不能当逐图正确率。P42.88%、R33.28%、mAP50 32.65%、mAP50–95 15.74%。该报告属于导出前源模型，INT8版本另有输出一致性检查，未做同规模重新标注的部署精度评估。

导航卡片4个真值4个正确检出来自极小、相关的PC验证集，未在ARM统一入口联调。正文不把它写成导航100%泛化识别。

## 资源与缺测项

1280×720×2字节×10帧×8=147.456 Mb/s，仅为YUYV名义像素负载。115200、8N1理论每方向上限11520字节/秒，不代表实测有效吞吐。整机功耗、全部进程与NPU内存峰值、独立硬件帧率、曝光到屏幕延迟未完成测量。

## 复算命令

在本包根目录：

```sh
python3 06_测试结果与统计/recompute_metrics.py
```

脚本读取原始JSON/JSONL，重新检查图像、按键、FPGA身份与动作并复算比例、FPS、帧龄和P/R；预期计数不一致会报错。仅需Python标准库。另保留原评分器源码便于检查检查项定义。
'''
    put(dest, '06_测试结果与统计/测试指标与统计方法.md', performance_doc)
    write_protocol(dest, tx, real_wire)
    write_provenance(dest, original)
    write_html_views(dest)


def add_model_inventory(dest):
    root = dest / '01_ARM实机程序/runtime'
    manifest = read(root / 'manifest.json')
    roles = {
        'boundary_fp.rknn': ('boundary场景', '512×288类型边界网络'),
        'road_int8_heads.rknn': ('当前全场景', '384×640道路四分类INT8裸头'),
        'lamp_aligned_fp_heads.rknn': ('当前signal场景', '256×640灯色矩形检测器'),
        'color_guard_batch2_fp.rknn': ('当前signal场景', '颜色/发光门控batch2'),
        'lamp_types_batch2_fp.rknn': ('当前signal场景', '整灯灯型batch2'),
        'lamp_attributes.onnx': ('当前signal按候选需要加载', 'CPU ONNX类型/方向/残差路径'),
    }
    rows = []
    for model in manifest['models']:
        p = root / model['path']
        role, purpose = roles.get(p.name, ('保留兼容参数分支', '非默认分支模型，随完整源码一并保留'))
        rows.append(dict(file=model['path'], role=role, purpose=purpose,
                         bytes=p.stat().st_size, sha256=model['sha256'], original_board_path=model['source']))
    folder = dest / '08_来源与许可证';folder.mkdir(exist_ok=True)
    (folder / '全部运行模型清单.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    with (folder / '全部运行模型清单.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    put(dest, '01_ARM实机程序/配置阅读提示.md', '''
    # 配置阅读提示

    本目录原运行包保持字节一致；当前参数由launcher调用预检启动器后，传给run_unified_demo.py。实际入口读取全部源码中的模型分支，不能只看一个历史JSON或文件名。

    当前参数：`--duration 21600 --capture-mode 720 --capture-fps 10 --lamp-profile capture_ft --road-profile int8 --classifiers batch2 --video-layout full --windowed --housing-refinement --proposal-mode geometry`。

    这些参数以源码形式保存在start_recovered_signal_on_board_20261028.py；模型哈希见runtime/manifest.json。manifest的profile=40是原组合阶段元数据，同时明确supported_profiles包含40/41，full_resolution_requires_profile=41；本包运行full布局必须是0x29即41。不要把40误认为本轮实际下载的固件。

    所有15个部署模型都在runtime/models，不依赖软链接或队友电脑的.pt文件。默认路径及其他保留参数分支使用不同子集，清单见来源目录。训练源.pt只用于追溯，开机不加载PyTorch，也不要求Ultralytics。
    ''')


def write_protocol(dest, tx, row):
    protocol = '''
    # ARM与FPGA实际通信协议

    本说明针对当前0x29固件和本包ARM源码，依据basic70_wire.py、physical_keys.py、transport_age_guard.py以及对应FPGA接口RTL。不是未经收到的队友ARM_FPGA_PROTOCOL_V2.md的替代抄本。串口为CP2102 `/dev/ttyUSB0`，115200、8位数据、无奇偶、1停止位，无硬件流控；Linux独占打开。

    ## 一 通用帧与查询

    有载荷写包格式为 `55 CMD LEN PAYLOAD CHECKSUM`，校验为最后一字节之前全部字节求和mod256，多字节整数低字节先传。0x13和0x12均为LEN=16，总长20字节。

    查询与简短ACK是特殊短帧，不带LEN：键查询 `55 03 58`、状态查询 `55 04 59`；普通写包ACK为 `55 CMD (55+CMD)&FF`。例如0x13 ACK=`55 13 68`，0x12 ACK=`55 12 67`。不能把第三字节一律解释为长度。

    | CMD | 方向 | 用途 |
    | --- | --- | --- |
    | 00 | ARM→FPGA | 视频模式，当前full为payload 07，预检legacy tiles为00 |
    | 03 | ARM查询→FPGA回包 | 实体按键state及epoch，回包 `55 03 02 STATE EPOCH SUM` |
    | 04 / 84 | ARM查询 / FPGA回包 | 带相机、帧号、epoch的实际动作/报警状态 |
    | 12 | ARM→FPGA | V2检测框，载荷版本字段为2；当前最多1框/帧 |
    | 13 | ARM→FPGA | V3当前视觉上下文和决策，版本字段为3 |
    | 14 | ARM→FPGA | 远程驾驶测试/停车清除，payload前缀54；只提供驾驶意图，不提供图像答案 |

    保留旧V2类是代码兼容所需，当前基线使用V3上下文和0x84反馈。LED亮、JTAG DONE或三字节ACK均不能单独证明版本、当前帧或实际动作正确。

    ## 二 0x13上下文载荷

    下表offset从PAYLOAD第0字节开始；不是整包偏移。

    | offset | 字段 | 定义和本版实际使用 |
    | --- | --- | --- |
    | 0 | version | 3 |
    | 1 | camera_id | 0–3；当前仅0参与成品路径 |
    | 2–3 | frame_id | uint16 LE，主机结果编号，不是已证实的FPGA像素帧号 |
    | 4 | scene | 1信号灯、2边界、3导航保留、4风险/行人 |
    | 5 | valid_flags | 见下文，当前bit2/bit4不置位 |
    | 6 | object_flags | bit0车辆，bit1行人，bit2两轮车，bit5交通灯 |
    | 7 | risk_flags | bit0中心风险，bit1左侧风险，bit2右侧风险，bit3保留风险；当前ARM置bit0/bit3，侧向独立感知未验证 |
    | 8 | light_state | 0未知、1红、2黄、3绿、4灭/关闭；灯有效位不成立时填0 |
    | 9 | navigation | 0未知、1左、2直、3右；当前ARM固定0 |
    | 10 | control_flags | bit0请求行驶、bit1强行、bit2请求变道、bit3右侧、bit4强行变道、bit5实际运动；当前不置bit5 |
    | 11 | movement | 0未知、1左、2直、3右，来源为按键选择的意图 |
    | 12 | boundary | 0未知、1虚线、2实线，选择请求侧的图像边界 |
    | 13 | epoch | 当前按键事件序号uint8 |
    | 14 | age_units | 一单位5ms，向上取整；到发送前重新计算并加入传输余量 |
    | 15 | arm_action | 1限制停车、2允许请求动作；具体左/直/右由FPGA结合scene和movement判定 |

    valid_flags：bit0图像有效；bit1选灯有效；bit2导航有效；bit3边界有效；bit4实际运动测量有效；bit5按键/驾驶意图确认；bit6 ARM时效许可。当前ARM没有导航和真实运动测量，因此bit2/bit4为0，不能根据协议存在字段就宣称已实现这些功能。

    传输边界函数对age向上取整，含10ms传输余量；剩余GO预算不足50ms时撤销许可，不能用发送时刻重新给旧图像500ms完整租约。FPGA拒绝age>=100、重放或身份异常；租约超时返回STOP并使决策无效。完整控制原则见对应RTL摘录。

    ## 三 0x12检测框载荷

    | offset | 字段 |
    | --- | --- |
    | 0 | version=2 |
    | 1 | camera_id |
    | 2–3 | frame_id LE |
    | 4 | slot |
    | 5 | 当前为0 |
    | 6 | 类别：0交通灯、1车辆、2两轮车、3行人 |
    | 7 | round(confidence×255)，限制0–255 |
    | 8–9 / 10–11 | x1 / y1，uint16 LE |
    | 12–13 / 14–15 | x2 / y2，uint16 LE |

    原图坐标按输入宽高分别缩放到1920×1080协议画布，端点按代码裁剪。当前每帧最多发送1个框，优先行人/两轮车再按置信度排序。0x12 ACK仅证明该命令已接收，不代表所有检测框已在FPGA屏幕可见；ARM窗口绘制与FPGA视频OSD应分别验收。

    ## 四 按键与0x84回包

    state低2位：0直、1左、2右、3保留；bit2–3场景编号0信号、1边界、2导航、3行人；bit4请求行驶；bit5强行；bit6远程测试标记；bit7保留。物理按键成绩要求bit6清除且来源为physical_keys，新事件epoch必须改变。导航保留状态当前ARM不支持，会保守停车。

    0x84回复总长16字节：`55 84 0C`后为12字节载荷，再加校验。

    | offset | 字段 |
    | --- | --- |
    | 0 | version=3 |
    | 1 | camera_id |
    | 2–3 | accepted frame_id LE |
    | 4 | accepted epoch |
    | 5 | accepted scene |
    | 6 | action_code |
    | 7 | alarm_code |
    | 8 | bit0 decision_valid，bit1 link_timeout，bit2 seen_extended |
    | 9 | reject_reason：0无拒绝，1非扩展帧，2过期，3重放 |
    | 10 | build_tag，当前41即0x29 |
    | 11 | capabilities，当前127；具体能力按本版源码/RTL |

    action_code：1 STOP、2直行、3左转、4右转、5左变道、6右变道。枚举齐全不代表本轮已逐项完成场景验收。

    alarm_code：0无、1碰撞风险、2强行红灯、3强行实线变道、4链路超时、5输入无效、6过期、7重放。界面顶部ALARM是布尔值，不能把ALARM1误写成alarm_code=1。

    当前发送0x13后先取得命令ACK，再发0x04查询，核对相机、帧号、epoch、scene、版本、reject_reason、超时与时效才标记确认。不同机器的单调时间不可直接相减。真实TX/RX摘录和完整字段JSON见同目录，不用构造包假装实测。

    ## 五 实测摘录的解释

    本包摘录来自11月1日HDMI实时运行的boundary场景，不是红绿灯精度测试。原始文件是 `06_测试结果与统计/原始证据/HDMI恢复_20261101/steady_check.json`。日志既保留0x13的真实写出字节，也保留ACK和0x84分段读取；0x04查询写出时刻未单独记录，因此未补造时间。

    对应FPGA接口RTL仅为解释接口而摘录，完整FPGA工程、约束、IP和bit生成必须使用队友完整工程；不能把三个文件作为可独立编译的工程。协议说明和bit哈希共同关联已有配置，不构成设备SRAM的密码学读回证明。
    '''
    put(dest, '05_通信协议与真实日志/通信协议_当前0x29基线.md', protocol)


def add_evidence(dest, original, copy):
    entries = [
        ('HDMI恢复/01_当前实时边界检测_含过期提示.jpg', 'runs/restore_hdmi_live_20261101/live_screen_light.jpg',
         '2026-11-01', '0x29 实时HDMI', '香橙派真实桌面截图，保留displayed image expired提示，不能当显示时效已通过'),
        ('HDMI恢复/02_当前实时检测原屏.png', 'runs/restore_hdmi_live_20261101/live_screen.png',
         '2026-11-01', '0x29 实时HDMI', '显示道路目标和边界推理；有过期提示，和短测流水线数据分开解释'),
        ('绿行/01_已确认绿行历史演示.jpg', 'runs/basic_finish70_20261028/frozen_signal_20261028/green_confirmed_preview.jpg',
         '2026-10-28', '0x29 HDMI 远程驾驶测试', '原图明示REMOTE TEST DRIVER；用于说明图像检测与FPGA回包，不能当实体KEY6截图'),
        ('绿行/02_实测输入帧.png', 'runs/basic_finish70_20261028/recovered_bundle_20261028/evidence/controlled_signal/source_audit/green_go/000015_processed.png',
         '2026-10-28', '0x29 控制场景', '实际输入画面，未叠加决策；实体按键65/90另见独立JSONL'),
        ('红停/01_红灯实际输入帧.png', 'runs/basic_finish70_20261028/recovered_bundle_20261028/evidence/controlled_signal/source_audit/red_stop/000001_processed.png',
         '2026-10-28', '0x29 控制场景', '红灯源画面，不能单靠输入图证明停车；89/90及人工确认另附'),
        ('强行红灯/01_强行场景实际输入帧.png', 'runs/basic_finish70_20261028/recovered_bundle_20261028/evidence/controlled_signal/source_audit/forced_red/000001_processed.png',
         '2026-10-28', '0x29 控制场景', '对应强行红灯场景输入；未找到该实体KEY7批次原屏截图，88/90、回包和人答复为主要证据'),
        ('停车/01_停车历史演示.jpg', 'runs/basic_finish70_20261028/recovered_signal_20261028/live_preview.jpg',
         '2026-10-28', '0x29 HDMI 远程驾驶测试', '显示STOP和UART确认；原图标REMOTE TEST DRIVER，实体停车90/90另附'),
        ('旧版变道示例/left_dashed.jpg', 'runs/basic_finish70_20261027/lane_keys/accepted_previews/left_dashed.jpg',
         '2026-10-27', '0x27 远程驾驶测试', '旧版本左虚线示例；不得转记为当前0x29已完成变道回归'),
        ('旧版变道示例/forced_left_solid.jpg', 'runs/basic_finish70_20261027/lane_keys/accepted_previews/forced_left_solid.jpg',
         '2026-10-27', '0x27 远程驾驶测试', '旧版本强行实线报警示例，非当前版本通过证明'),
        ('旧版变道示例/right_dashed.jpg', 'runs/basic_finish70_20261027/lane_keys/accepted_previews/right_dashed.jpg',
         '2026-10-27', '0x27 远程驾驶测试', '旧版本右虚线示例'),
        ('本地回放历史/01_文件回放_非HDMI.png', 'runs/resume_stable_demo_20261101/screen_after_cache.png',
         '2026-11-01', 'ARM本地文件推理', '明确标注LOCAL FILE INFERENCE；该图不证明HDMI输入，KEY6联动有单独人答复'),
        ('PC道路验证/01_BDD样例.jpg', 'runs/adas/bdd100k_4class_stage42/report/examples/b1cd1e94-549d0bfe.jpg',
         '2026-10-07', 'PC离线识别', 'PC验证可视化，不是香橙派性能或两板联动截图'),
        ('PC导航验证/01_导航开发复核页.jpg', 'runs/navigation_primary_locator_20261018/v3_label_corrected/val_review_20261018_213049_344043/page_01.jpg',
         '2026-10-18', 'PC开发复核', '已知来源导航复核页；不代表导航已在ARM统一程序运行'),
    ]
    catalog = []
    for target, source, date, scope, note in entries:
        p = copy(original / source, Path('07_场景证据') / target)
        catalog.append(dict(file=target, original_path=source, date=date, scope=scope, note=note,
                            sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    folder = dest / '07_场景证据'
    (folder / '图片索引.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding='utf-8')
    put(dest, '07_场景证据/README_逐图说明.md', '# 场景证据与截图使用说明\n\n' +
        table(['文件','日期','层级/版本','可证明和不能证明'], [[x['file'],x['date'],x['scope'],x['note']] for x in catalog]) + '''

所有图片均复制已有原始文件，未改写界面标签、报警值或时间。没有用AI生成图或事后绘制的状态页冒充原始屏幕。红停及强行红灯的图片是实际输入帧；实体功能的完成以连续90帧JSONL、FPGA回包和人工答复一起判断。

尚缺：当前0x29每个实体用例的完整原屏录屏、实物板卡与按键同框录像、导航ARM运行画面、当前统一版本全部转道/风险场景录像。本包不新造不存在的录像，正式比赛演示由队友补录。
''')


def write_provenance(dest, original):
    put(dest, '08_来源与许可证/数据权重第三方与师兄代码来源.md', '''
    # 数据 权重 第三方与师兄代码来源

    整理日期2026年11月2日。文件哈希与来源路径可核对，不把下载到本机的文件自动当作团队原创。运行权重以15项模型清单为准；历史训练或未采用候选单独归档。

    ## 一 数据与标注

    | 来源 | 本项目用途 | 记录与范围 |
    | --- | --- | --- |
    | BDD100K | 道路四类检测、类型边界训练及验证 | 原图/车道与检测标注的预处理；道路历史验证1000图；代码许可与数据许可分别核对 |
    | SJTU S²TLD | 机动车灯色预训练/微调来源 | red/yellow/green/off；与官方视频裁剪混合，XML转YOLO；原数据不作为本包完整再分发附件 |
    | 紫光同创赛题三官方参考视频 | 开发标注、目标域微调、已知场景回放、HDMI演示 | 文件名video_20260929_*；含AI辅助复核标注，已被用于开发，不能称未知道路盲测 |
    | 人工/AI辅助派生标签与退化视图 | 裁剪、灯头/类型/边界修正和压力诊断 | manifest记录annotator与来源；增强、重复采样不增加独立路口数量 |

    官方最初灯色划分记录：train含112507、112739、113343、113452；val为113539；test为112454、113421；111440作为模拟屏幕内容另处理。后续开发持续复核了这些来源，因此不能再将同一路程test片段声明为未知盲测。11月1日实时演示使用113343，属于已知开发来源。

    原灯色训练混合曾使用S²TLD train3647张及官方100张重复4次，重复采样不是增加新图片；后续矩形裁剪与采集退化训练的精确清单以随包manifest为准。当前灯模型先从tiles_stage2初始化，再以12轮矩形采集对齐训练所得last.pt导出；不把旧训练规模直接套为最后模型的独立样本数。

    示例输入中保留官方113343原视频，便于队内复现，来源原路径和SHA在复制来源清单。原素材权利属于相应提供者，不声明本团队享有对素材的公开转授权；发布公共仓库时可只保留文件名、校验值和官方获取方式。

    官方项目入口：[BDD100K](https://github.com/bdd100k/bdd100k)、[BDD100K数据与代码许可说明](https://github.com/bdd100k/bdd100k/blob/master/doc/source/license.rst)、[SJTU S²TLD](https://github.com/Thinklab-SJTU/S2TLD)。BDD100K代码的BSD许可不自动覆盖其数据许可；S²TLD再分发条件以官方说明为准，本轮未据README推断全部数据可任意公开。

    ## 二 实际运行权重

    | 部署模型 | 可追溯上游 |
    | --- | --- |
    | road_int8_heads.rknn | bdd100k_4class_stage42/weights/best.pt，SHA a9af921e...；384×640裸头导出后用128个样例做INT8校准 |
    | lamp_aligned_fp_heads.rknn | lamp_capture_aligned/trial/weights/last.pt，SHA88384105...；初始化为s2tld_official_tiles_stage2/deployment_best.pt，12轮、912次更新 |
    | boundary_fp.rknn | lane_boundary_seg_f721cc098f9b/weights/best.pt，SHA3e868134...；冻结YOLOv8n预训练编码器加语义解码头 |
    | color_guard_batch2_fp.rknn及兼容分支 | signal_emission_guard_20261017_172936_118725/weights/best.pt，SHA4c51451b...；先导出color_guard.onnx再转RKNN |
    | lamp_types_batch2_fp.rknn及兼容分支 | lamp_attributes.onnx的整灯类型输出子图提取，使用相同编码器和类型头，不是重新训练 |
    | lamp_attributes.onnx | signal_attributes_consensus、signal_spatial_tail类型/方向残差、direction repair及控制类型guard等五个已冻结源checkpoint组合导出，详见export_verification.json |

    完整64位SHA、原板路径、模型用途和实际大小见全部运行模型清单；导出记录和已找到的源.pt也随包。不是所有分支都在当前默认路径执行，但保留它们以保证完整源码可选择的兼容参数有对应模型。

    初始YOLO检测/分类/分割架构和预训练初始化来自Ultralytics，不宣称网络骨干从零设计或从零训练。类型边界解码、采样/后处理、协议和场景逻辑是本项目迭代实现，仍需保留上游许可和具体修改说明。

    10月29日灯色与导航新候选没有替换当前模型。源记录中某些早期conversion.json字段adopted=false表示生成时尚待采用，不是最终运行证据；是否实际演示以冻结manifest、主程序参数和11月1日哈希校验为准。

    ## 三 第三方软件与许可证

    | 软件 | 用途 | 来源与许可处理 |
    | --- | --- | --- |
    | Ultralytics 8.3.163项目源码 | YOLO网络、预训练加载、训练和导出 | [官方源码](https://github.com/ultralytics/ultralytics)，本地AGPL-3.0原文随包；训练转换参考保留本项目当前ultralytics代码 |
    | PyTorch | PC训练与ONNX导出 | [官方](https://github.com/pytorch/pytorch)，BSD风格许可原文随包；ARM实际入口不依赖torch |
    | OpenCV | UVC采集、图像处理、显示、几何 | [官方](https://github.com/opencv/opencv)，许可原文从本机依赖保留；板端实测4.11.0 |
    | NumPy | 张量与后处理 | [官方](https://github.com/numpy/numpy)，BSD许可原文保留；板端实测1.26.4 |
    | ONNX Runtime | 灯型/方向CPU推理 | [官方](https://github.com/microsoft/onnxruntime)，MIT原文保留；板端实测1.23.2 |
    | RKNN Toolkit2 / Lite2 / NPU运行库 | 转换和RK3566 NPU推理 | [Rockchip官方](https://github.com/airockchip/rknn-toolkit2)、[官方SDK许可](https://github.com/airockchip/rknn-toolkit2/blob/master/LICENSE)；属于其SDK协议，不概括为MIT/Apache；本包不打包驱动和SDK wheel |
    | Linux V4L2、termios、OpenSSH、X11桌面 | 设备、串口、远程启动与显示 | 系统依赖，使用原系统组件；本包不是重打包的Linux镜像 |

    本机当前Python依赖版本可能与板端冻结版本不同。许可证原文标记为本机依赖，不拿本机pip环境冒充板端锁文件。新项目代码与各第三方文件保留各自来源，不用一个新许可证覆盖所有素材和SDK。

    本资料只列实际交付链中的来源。其他下载过的OpenLane、外部车道模型、VZC-TLD、DINO或实验候选，不因出现在电脑上就宣称已训练进最终权重。若队友追加这些成果，需另给版本、训练记录和来源。

    ## 四 师兄资料和队友FPGA代码

    用户提供过往年完整资料 `D:/fpga紫光同创/2025/`，作为2025成品架构、Orange Pi小型处理器协作、视频/串口与接线参考。该原路径当前不可读，本次没有取得可逐文件比对的原始快照，不能可靠量化复用代码比例，也不能声明没有复用。

    本包ARM运行文件可以通过原文件哈希追溯到本次项目的冻结目录；这只证明版本一致，不证明全部原创。请由掌握去年材料的队友补齐下面表格后再作为正式原创性/复用说明提交：

    | 待核实项 | 队友填写 |
    | --- | --- |
    | 2025原工程名称、作者与版本 | 待补 |
    | 直接复用的具体文件、函数及原路径 | 待补，附原文件和SHA或diff |
    | 只参考思路但重新实现的部分 | 待补具体说明 |
    | 本队修改人、修改日期、修改内容 | 待补，不能由AI猜测成员分工 |
    | 原作者许可及比赛要求 | 待补并保留说明 |

    本包包含的0x29参考bit和三个FPGA接口RTL摘录来自当前合作工程 `D:/PDS_WORK/adas_fullres_20261028`；它们不能替代队友全部FPGA工程或证明团队独立原创。完整FPGA引用、IP与板卡例程归属由FPGA负责人合并说明。

    ## 五 AI辅助与文件边界

    数据标注、错误复核、脚本编写和文档整理使用了AI辅助；标注示例manifest中可见assistant_visual_review_v1等记录。请队友按比赛实际要求填写AI使用声明，不能把AI建议写成人工现场试验，现场确认仍以用户答复与实测日志为证。

    本包未放入SSH私钥、密码、HF token、账号浏览器状态或整个训练数据集。所有复制文件的原路径和SHA见 `复制文件来源与SHA256.json`，可据此区分新整理文档与已存在程序/证据。权限说明中的待补项是资料缺口，不是已经取得的授权结论。
    ''')


class Diagram:
    def __init__(self, width, height, title):
        from PIL import Image, ImageDraw, ImageFont
        self.width, self.height = width, height
        self.image = Image.new('RGB', (width, height), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.ImageFont = ImageFont
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                    '<rect width="100%" height="100%" fill="white"/>']
        self.text(50, 35, title, 38, '#132b40')

    def font(self, size):
        return self.ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', size)

    def text(self, x, y, text, size=24, color='#20384c'):
        self.draw.text((x, y), text, font=self.font(size), fill=color)
        self.svg.append(f'<text x="{x}" y="{y+size}" font-family="Microsoft YaHei,sans-serif" font-size="{size}" fill="{color}">{html.escape(text)}</text>')

    def box(self, x, y, w, h, lines, fill='#edf4fa'):
        self.draw.rounded_rectangle((x, y, x+w, y+h), radius=16, fill=fill, outline='#54768e', width=2)
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{fill}" stroke="#54768e" stroke-width="2"/>')
        for i, line in enumerate(lines):
            self.text(x+20, y+18+i*37, line, 26 if i==0 else 22)

    def arrow(self, points, label=None, label_at=None, color='#557991'):
        self.draw.line(points, fill=color, width=4)
        a, b = points[-2], points[-1]
        angle = math.atan2(b[1]-a[1], b[0]-a[0])
        wing = [(b[0]-16*math.cos(angle+.45), b[1]-16*math.sin(angle+.45)),
                b, (b[0]-16*math.cos(angle-.45), b[1]-16*math.sin(angle-.45))]
        self.draw.polygon(wing, fill=color)
        coords=' '.join(f'{x},{y}' for x,y in points)
        self.svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="4"/>')
        self.svg.append('<polygon points="'+' '.join(f'{x},{y}' for x,y in wing)+f'" fill="{color}"/>')
        if label:
            self.text(*label_at, label, 21, color)

    def save(self, folder, name):
        folder.mkdir(parents=True, exist_ok=True)
        self.image.save(folder / (name+'.png'))
        (folder / (name+'.svg')).write_text('\n'.join(self.svg+['</svg>']), encoding='utf-8')


def draw_diagrams(dest):
    folder = dest / '04_模块图与数据流'
    d=Diagram(1640, 1120, 'ARM 软件模块与当前实际执行路径')
    d.box(390,130,860,140,['真实UVC输入与帧身份','LatestCaptureReader / 有界最新帧 / 图像质量与哈希','/dev/video0 · 1280×720 YUYV · 协商10 FPS'])
    boxes=[(40,['道路目标 NPU','384×640 INT8四类检测','CPU解码与NMS / 每帧运行']),
           (575,['信号场景 NPU 与 CPU','矩形灯、颜色门控、灯型与方向','灯组关联和连续绿灯确认']),
           (1110,['边界场景 NPU 与 CPU','512×288 类型边界网络','虚实线实例与图像中心左右边界'])]
    for x, lines in boxes:
        d.box(x,365,490,150,lines)
        d.arrow([(820,270),(820,315),(x+245,315),(x+245,365)])
    d.box(390,620,860,155,['统一场景与风险仲裁','实体按键state/epoch · 当前图像 · 时效 · STOP/GO/报警','虚拟车道；未知/过期保守停车；导航尚未接入'])
    for x,_ in boxes:
        d.arrow([(x+245,515),(x+245,565),(820,565),(820,620)])
    d.box(90,900,650,130,['USB-UART结果通道','V3上下文 + 最多1个检测框/帧','查询FPGA相机/帧号/epoch/动作确认'])
    d.box(900,900,650,130,['香橙派本机显示','绘制道路、边界、决策和UART状态','画面过期提示保留；不是FPGA全部OSD验收'])
    d.arrow([(650,775),(650,835),(415,835),(415,900)])
    d.arrow([(990,775),(990,835),(1225,835),(1225,900)])
    d.text(50,1060,'模型与配置按冻结SHA锁定；PC导航和未采用候选不计入当前ARM入口。',23)
    d.save(folder,'ARM模块图')
    d=Diagram(1640,1120,'视频与检测结果到 FPGA 的两条独立数据流')
    d.box(40,160,380,150,['电脑视频源','官方参考视频循环播放','HDMI输出；不承担ARM推理'])
    d.box(570,160,470,150,['盘古 FPGA 视频侧','HDMI IN → 视频链 → HDMI OUT','当前0x29固件；不传UART图像'])
    d.box(1190,160,410,150,['MS2130采集卡','HDMI输入 → UVC USB3','接香橙派USB3口'])
    d.arrow([(420,235),(570,235)],'HDMI',(450,200))
    d.arrow([(1040,235),(1190,235)],'HDMI',(1070,200))
    d.box(1190,490,410,165,['香橙派 ARM','USB采集 → NPU/CPU识别','本地决策与显示 / 独立串口','原始图像不经UART回传'])
    d.arrow([(1395,310),(1395,490)],'USB3 UVC',(1420,385))
    d.box(570,490,440,165,['盘古 FPGA 控制侧','实体KEY → 驾驶state/epoch','接收视觉结果、时效/重放保护','回传实际动作、报警及身份'])
    d.arrow([(1190,540),(1010,540)],'UART结果',(1040,500))
    d.arrow([(1010,620),(1190,620)],'按键/状态',(1040,635))
    d.box(40,490,380,165,['板上指示','LED4 = 停车或报警综合灯','不等同独立蜂鸣器证明','无实际电机/舵机接入'])
    d.arrow([(570,565),(420,565)],'逻辑指示',(440,530))
    d.box(1110,860,490,140,['独立显示器','接香橙派HDMI输出','显示ARM推理窗口与UART状态'])
    d.arrow([(1355,655),(1355,860)],'香橙派HDMI',(1380,750))
    d.box(40,860,970,140,['辅助链路与边界','电脑 → USB Cable II → FPGA JTAG，只用于配置；CP2102另接香橙派','默认实机入口不读取PC文件作为推理输入；按键提供意图，不提供灯色答案'])
    d.text(50,1050,'检测结果ACK与画面OSD可见性分开验收；传感器到显示器端到端延迟尚未实测。',23)
    d.save(folder,'视频与检测结果数据流')
    put(dest,'04_模块图与数据流/ARM模块图.mmd','''
    flowchart TD
      A[MS2130 UVC真实采集] --> B[最新帧与图像身份及质量]
      B --> C[道路四类检测 NPU INT8]
      B --> D[信号场景 灯色 灯型 方向与关联]
      B --> E[边界场景 类型边界及几何后处理]
      K[实体键 state和epoch] --> F[场景与风险仲裁 时效检查]
      C --> F
      D --> F
      E --> F
      F --> G[UART上下文与框 读回状态确认]
      F --> H[香橙派本机显示]
      G --> J[FPGA动作保护与LED]
      J --> K
    ''')
    put(dest,'04_模块图与数据流/数据流.mmd','''
    flowchart LR
      PC[电脑官方视频] -->|HDMI| V[FPGA视频输入输出]
      V -->|HDMI| CAP[MS2130]
      CAP -->|USB3 UVC| ARM[Orange Pi 3B NPU和CPU]
      ARM -->|本机HDMI| SCREEN[独立显示器]
      ARM -->|USB UART 上下文与检测框| FPGA[FPGA场景控制与保护]
      FPGA -->|按键和带身份的实际状态| ARM
      FPGA --> LED[停车或报警综合指示]
    ''')


def inline(text):
    text=html.escape(text)
    text=re.sub(r'!\[([^\]]*)\]\(([^)]+)\)',r'<img alt="\1" src="\2">',text)
    text=re.sub(r'(?<!!)\[([^\]]*)\]\(([^)]+)\)',r'<a href="\2">\1</a>',text)
    text=re.sub(r'`([^`]+)`',r'<code>\1</code>',text)
    text=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',text)
    return text


def markdown_html(text):
    result=[];lines=text.splitlines();index=0;code=False
    while index<len(lines):
        line=lines[index]
        if line.startswith('```'):
            code=not code;result.append('<pre><code>' if code else '</code></pre>');index+=1;continue
        if code:
            result.append(html.escape(line)+'\n');index+=1;continue
        if line.startswith('|'):
            result.append('<table>');first=True
            while index<len(lines) and lines[index].startswith('|'):
                row=lines[index];index+=1
                if re.fullmatch(r'[| :\-]+',row):continue
                tag='th' if first else 'td';first=False
                result.append('<tr>'+''.join('<'+tag+'>'+inline(c.strip())+'</'+tag+'>' for c in row.strip('|').split('|'))+'</tr>')
            result.append('</table>');continue
        match=re.match(r'^(#{1,6}) (.*)',line)
        if match:
            level=len(match[1]);result.append(f'<h{level}>'+inline(match[2])+f'</h{level}>')
        elif line.strip():result.append('<p>'+inline(line)+'</p>')
        index+=1
    return '\n'.join(result)


def write_html_views(dest):
    css='''body{font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif;line-height:1.8;max-width:1100px;margin:36px auto;padding:0 28px;color:#16232d;background:#fff}h1{font-size:30px}h2{margin-top:38px}table{border-collapse:collapse;width:100%;margin:20px 0;font-size:14px}th,td{border:1px solid #cad3db;padding:10px 12px;text-align:left;vertical-align:top}th{background:#eaf0f5}tr:nth-child(even){background:#f8fafc}code{font-family:Consolas,monospace;word-break:break-word;background:#f0f3f6;padding:1px 4px}pre{white-space:pre-wrap;background:#f0f3f6;padding:20px;line-height:1.55}pre code{padding:0}img{max-width:100%;height:auto}a{color:#145b8e}@media print{body{max-width:none;margin:0;padding:0;font-size:11pt}h1,h2,h3{break-after:avoid}tr,img,pre{break-inside:avoid}a{color:inherit;text-decoration:none}}'''
    names=['README_先看.md','README_从开机到运行.md','03_技术文档/ARM技术说明_供合并比赛报告.md',
           '03_技术文档/功能验证矩阵.md','03_技术文档/队友合并清单.md',
           '05_通信协议与真实日志/通信协议_当前0x29基线.md',
           '06_测试结果与统计/测试指标与统计方法.md',
           '07_场景证据/README_逐图说明.md','08_来源与许可证/数据权重第三方与师兄代码来源.md']
    for name in names:
        p=dest/name;text=p.read_text(encoding='utf-8');title=text.splitlines()[0].lstrip('# ')
        p.with_suffix('.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>'+html.escape(title)+'</title><style>'+css+'</style><body>'+markdown_html(text)+'</body></html>',encoding='utf-8')
    links=''.join('<li><a href="'+str(Path(n).with_suffix('.html')).replace('\\','/')+'">'+html.escape(Path(n).stem)+'</a></li>' for n in names)
    gallery=read(dest/'07_场景证据/图片索引.json')
    photos=''.join('<h3>'+html.escape(x['file'])+'</h3><p>'+html.escape(x['scope']+'；'+x['note'])+'</p><img loading="lazy" src="07_场景证据/'+x['file']+'">' for x in gallery)
    (dest/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>ARM最终交接</title><style>'+css+'</style><body><h1>ARM 最终交接</h1><p>2026年11月2日整理。资料用于队友合并报告，非已通过全工程验收的声明。</p><ul>'+links+'</ul><h2>模块图</h2><img src="04_模块图与数据流/ARM模块图.png"><img src="04_模块图与数据流/视频与检测结果数据流.png"><h2>场景证据</h2>'+photos+'</body></html>',encoding='utf-8')
