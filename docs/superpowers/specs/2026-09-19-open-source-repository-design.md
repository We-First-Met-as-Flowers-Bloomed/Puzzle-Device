# Puzzle Device 开源仓库设计

## 目标

将现有 STM32 控制工程、MaixCAM2 视觉程序、嘉立创 EDA 工程和 3D 打印文件整理成可阅读、可下载、可复现的公开仓库，并提供清晰的使用路径和演示入口。

## 目录设计

- `Firmware/STM32F407/`：STM32CubeMX/HAL 四轴运动控制、通信和执行机构程序。
- `Firmware/MaixCAM2-Vision/`：视觉检测、几何解算、拼图求解、配置示例和测试。
- `Hardware/Schematic/`、`Hardware/PCB/`、`Hardware/BOM/`：硬件工程文件及缺失资料说明。
- `Mechanical/STL/`、`Mechanical/STEP/`：3D 打印模型及缺失资料说明。
- `Docs/System-Design/`：系统设计和开发资料。
- `Docs/Images/`：README 可引用的项目图片。
- `Demo/`：演示说明与 Bilibili 视频入口。

## 发布策略

源码直接纳入 Git；排除嵌套 `.git`、编译输出、缓存、临时文件、历史发布包和本机配置。原始压缩包保留在仓库目录之外，不上传 GitHub。仓库采用已有 MIT License。

## 验收条件

目录完整；README 包含项目原理、结构、准备、烧录、部署、运行和演示链接；不存在超过 GitHub 单文件限制的文件；不存在明显的本机密钥或敏感配置；Git 工作树干净并成功推送到 `main`。
