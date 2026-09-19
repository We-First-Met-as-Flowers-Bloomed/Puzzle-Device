# Puzzle Device｜AI 视觉自动拼图装置

面向 2026 年电子设计竞赛 E 题要求设计的自动拼图实验装置。系统以 MaixCAM2 完成 A4 工作区识别、碎片检测、拼图求解和目标位姿计算，以 STM32F407 完成四轴运动、吸取、旋转和放置。人工完成上电和启动操作后，装置自动执行后续流程。

> 本仓库记录参赛原型的软硬件实现，适合电子设计竞赛复盘、机器视觉与运动控制教学，以及自动控制综合实验参考。实际复现时必须按自己的相机安装位置、机械原点和脉冲当量重新标定。

## 演示视频

项目运行效果已发布到 Bilibili：

- [观看 E 题装置演示与项目动态](https://space.bilibili.com/234483320/dynamic?spm_id_from=333.1387.list.card_title.click)
- 文字说明见 [Demo/README.md](Demo/README.md)

## 系统组成

```mermaid
flowchart LR
    A[上电并等待初始化] --> B[按下 MaixCAM2 启动键]
    B --> C[视觉检测与拼图求解]
    C --> D[生成拾取/放置坐标与旋转角]
    D --> E[UART JSONL 发送至 STM32F407]
    E --> F[四轴运动与电磁铁吸放]
    F --> G{全部碎片完成?}
    G -- 否 --> F
    G -- 是 --> H[声光提示/实验结束]
```

核心模块包括：

- **视觉感知**：检测 A4 纸、分界线和 1—4 块拼图碎片。
- **拼图求解**：计算合法矩形组合、各碎片目标位置和旋转角度。
- **坐标转换**：将相机坐标转换为机构坐标，再转换为绝对脉冲位置。
- **运动执行**：STM32F407 控制 X、Y、Z、R 四轴和电磁铁完成抓取与放置。
- **一键运行**：装置初始化后，由相机端启动键触发一次完整任务。

## 仓库结构

```text
Puzzle-Device/
├── README.md
├── LICENSE
├── Firmware/
│   ├── STM32F407/          # STM32CubeMX/HAL 运动控制工程
│   └── MaixCAM2-Vision/    # AI视觉、拼图求解和坐标输出
├── Hardware/
│   ├── Schematic/          # 嘉立创EDA工程文件
│   ├── PCB/                # PCB资料状态说明
│   └── BOM/                # 元器件清单状态说明
├── Mechanical/
│   ├── STEP/               # STEP资料状态说明
│   └── STL/                # 3D打印模型
├── Docs/
│   ├── System-Design/      # 系统与固件设计资料
│   └── Images/             # 项目图片预留目录
└── Demo/                   # 演示视频入口与操作流程
```

## 快速开始

### 1. 准备工具

- STM32CubeMX 和 Keil MDK-ARM（STM32F407 工程）。
- MaixVision、与设备系统匹配的 MaixCAM2 Runtime。
- 嘉立创EDA专业版（打开 `.eprj2` 工程）。
- 3D 打印机或可读取 STL 的切片软件。
- 3.3 V TTL 串口连接，MaixCAM2 TX/RX 与 STM32 RX/TX 交叉并共地。

### 2. 编译并烧录 STM32F407

1. 进入 [`Firmware/STM32F407`](Firmware/STM32F407)。
2. 可用 STM32CubeMX 打开 `Automatic inspection vehicle.ioc` 查看和重新生成外设配置。
3. 用 Keil 打开 `MDK-ARM/Automatic inspection vehicle.uvprojx`。
4. 选择 STM32F407 对应目标，编译并通过调试器烧录。
5. 首次运动前断开电机负载，检查限位、轴方向、回零逻辑和急停条件。

### 3. 部署 MaixCAM2 视觉程序

1. 进入 [`Firmware/MaixCAM2-Vision`](Firmware/MaixCAM2-Vision)。
2. 将 `config.example.json` 复制为 `config.json`。
3. 使用标定工具生成相机内参和 `frame_to_robot` 变换；确认无误后再设置 `robot_calibrated=true`。
4. 使用 MaixVision 打开整个项目目录并选择“运行项目”，不要只上传 `main.py`。
5. 验证识别和串口输出后，可通过“安装应用”部署，并在设备设置中配置开机启动。

视觉端详细参数、标定命令、UART协议和测试方法见 [MaixCAM2 说明](Firmware/MaixCAM2-Vision/README.md)。

### 4. 接线和联调

1. 保持电机驱动关闭，先验证 MaixCAM2 与 STM32 的串口接线。
2. 使用视觉工程中的 `tools/uart_smoke_test.py` 发送固定测试消息。
3. 检查 STM32 能完整接收一行 JSONL，并确认不会在半包数据到达时启动机构。
4. 分别验证 X、Y、Z、R 轴方向、软硬限位、回零和电磁铁吸放。
5. 低速空载运行完整流程，再安装拼图碎片进行测试。

### 5. 正常操作

1. 放置符合题目范围的 A4 工作区和拼图碎片，确保完整纸边与分界线均在相机画面内。
2. 给装置上电，等待 STM32、驱动器和视觉模块初始化完成。
3. 按下 MaixCAM2 界面中的 `START`。
4. 系统自动识别碎片、规划目标位姿，并依次完成抓取、移动、旋转和放置。
5. 运行过程中不要移动相机、A4 纸或机械结构，也不要进入机构运动范围。

## 关键配置

视觉端默认以 `1280×720`、30 fps 取帧，通过 `/dev/ttyS2`、115200 baud 向主控发送 JSONL。示例动作如下：

```json
{
  "piece_id": 0,
  "pickup_x_pulse": 9600,
  "pickup_y_pulse": 7200,
  "place_x_pulse": 18000,
  "place_y_pulse": 12800,
  "rotation_deg": 30.0
}
```

这些脉冲值仅用于说明协议。更换相机、安装高度、传动机构、驱动细分或机械原点后，必须重新测量并修改配置。

## 资料完整性

当前仓库包含 STM32 与 MaixCAM2 源码、嘉立创EDA工程和三件 STL 模型。PCB Gerber、独立 BOM 和 STEP 模型尚未整理，因此对应目录暂以说明文件占位。欢迎根据原理图和实际装配补充并提交 Pull Request。

## 测试

视觉模块的基础标定测试不需要真实相机：

```powershell
cd Firmware/MaixCAM2-Vision
python -m unittest tests.test_calibration_points -v
```

`tests.test_vision` 中的大部分测试可以直接运行，但部分实拍回归测试依赖未随当前开源包发布的现场图片和本机标定配置，详见 [`tests/README.md`](Firmware/MaixCAM2-Vision/tests/README.md)。STM32 工程的 `tests/` 包含主机端 C 测试和策略检查脚本，可在 Windows/PowerShell 环境按需运行。真实硬件的方向、限位、回零、标定精度和急停仍需逐项实测。

## 开源许可

项目采用 [MIT License](LICENSE)。第三方 STM32 HAL、CMSIS 及其他依赖仍遵循其各自许可证。

## 致谢与交流

如果这个项目对你的电子设计竞赛、自动控制实验或机器视觉学习有帮助，欢迎 Star、Fork、提交 Issue 或 Pull Request。也可以通过上方 Bilibili 页面查看演示并交流复现经验。
