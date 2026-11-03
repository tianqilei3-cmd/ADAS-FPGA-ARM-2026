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
