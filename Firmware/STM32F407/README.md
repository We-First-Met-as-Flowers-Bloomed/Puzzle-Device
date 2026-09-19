# STM32F407 Motion Controller

主控工程基于 STM32F407VGT6、STM32CubeMX 和 HAL 库，负责四轴运动、电磁铁吸放、限位与回零、OLED状态显示，以及接收 MaixCAM2 发来的 JSONL 运动计划。

## 打开与编译

1. 使用 STM32CubeMX 打开 `Automatic inspection vehicle.ioc` 查看外设配置。
2. 使用 Keil MDK-ARM 打开 `MDK-ARM/Automatic inspection vehicle.uvprojx`。
3. 编译并通过调试器烧录到 STM32F407VGT6。

## 主要目录

- `app/`：任务状态机和视觉串口协议解析。
- `hardware/`：四轴、电磁铁、限位、位置、OLED和底层驱动封装。
- `Core/`：CubeMX生成的初始化、中断和主程序代码。
- `Drivers/`：CMSIS与STM32F4 HAL库。
- `tests/`：主机端C测试和PowerShell策略检查。

烧录后先在断开负载或低速状态下逐轴验证方向、限位、回零和电磁铁，再连接视觉模块进行整机测试。
