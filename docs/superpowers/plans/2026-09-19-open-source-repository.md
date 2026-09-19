# Puzzle Device Open-Source Repository Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将竞赛工程整理为可阅读、可复现并能直接发布到 GitHub 的开源仓库。

**Architecture:** 保留固件、硬件、机械、文档和演示五个清晰边界；源码直接纳入 Git，生成物和本机配置通过清理与 `.gitignore` 排除。根 README 负责项目介绍和完整使用路径，各子目录 README 说明该目录内容及工具要求。

**Tech Stack:** STM32CubeMX/HAL、MaixCAM2/Python、嘉立创EDA、STL、Git/GitHub。

---

### Task 1: 整理源码与设计资料

**Files:**
- Create: `Firmware/STM32F407/`
- Create: `Firmware/MaixCAM2-Vision/`
- Create: `Hardware/Schematic/`
- Create: `Hardware/PCB/README.md`
- Create: `Hardware/BOM/README.md`
- Create: `Mechanical/STL/`
- Create: `Mechanical/STEP/README.md`
- Create: `Docs/System-Design/`

- [ ] 解压 STM32 工程，排除嵌套 `.git` 和编译输出，并复制到 `Firmware/STM32F407/`。
- [ ] 解压 MaixCAM2 工程，删除缓存、临时目录、历史发布包和 `config.json`，保留 `config.example.json`。
- [ ] 复制 EDA 工程、STL 文件和系统设计资料到对应目录。
- [ ] 检查目录中不存在 `.git`、`__pycache__`、`.pytest_cache`、构建缓存和超过 100 MB 的文件。

### Task 2: 编写开源说明

**Files:**
- Modify: `README.md`
- Create: `.gitignore`
- Create: `Firmware/README.md`
- Create: `Hardware/README.md`
- Create: `Mechanical/README.md`
- Create: `Docs/README.md`
- Create: `Demo/README.md`

- [ ] 编写项目概述、功能、系统流程、目录树和技术组成。
- [ ] 编写 STM32 烧录、MaixCAM2 配置部署、接线检查、上电初始化和一键启动步骤。
- [ ] 在 `Demo/README.md` 和根 README 中加入 Bilibili 演示入口。
- [ ] 标注当前未提供的 PCB Gerber、BOM 和 STEP 文件，避免读者误判。

### Task 3: 验证与发布

**Files:**
- Modify: `docs/dev-log.md`
- Modify: `docs/dev-report.md`

- [ ] 扫描敏感信息、绝对路径、缓存、嵌套仓库和超大文件。
- [ ] 运行视觉模块测试；若依赖环境不满足，记录可复现的失败原因。
- [ ] 检查 README 中的相对链接和目录引用均存在。
- [ ] 更新开发日志，提交全部文件并推送 `main`。
- [ ] 通过 GitHub API核对远程提交、README、许可证及主要目录。
