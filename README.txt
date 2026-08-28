# Risk-Agent V2 — 信贷风控模型智能监测与自主迭代系统

> 第五届中国研究生金融科技创新竞赛 · 江苏银行信贷风控赛题 · 揭榜挂帅-14

## 一、项目简介与功能概述

Risk-Agent V2 是一套面向贷前 A 卡（申请评分卡）全生命周期的**智能监测与自主迭代闭环系统**。系统通过五个 Agent 模块（M1~M5）实现"监测告警 → 根因诊断 → 策略推荐 → 灰度验证 → 冠军替换"的自动化闭环，替代传统人工盯盘 + 经验决策模式，解决模型性能衰减难察觉、标签延迟决策盲区、迭代策略一刀切三大技术卡点。

核心功能：
- **双阶段监测**：标签未成熟期（无标签 4 项规则）+ 成熟期（有标签 9 项规则）双轨制，解决 30 天标签延迟
- **双层根因 + LLM 四维归因**：规则层定位漂移源与贡献度，DeepSeek-V4-Flash 大模型做数据/特征/模型/业务四维独立归因
- **三策略自适应迭代**：light（增量微调）/ standard（标签重加权）/ major（特征重构+全量重训），按漂移严重度自动匹配
- **四层灰度部署**：L1 离线回放 → L2-20% → L2-50% → 全量，DeLong + KS Bootstrap 显著性检验
- **Champion/Challenger 动态替换**：显著优或非劣即替换，全程 JSON+DOCX 留痕可审计
- **防泄漏机制**：major 策略延迟重训参与行在评测集中自动剔除，杜绝自评作弊

## 二、模块框架说明

### 2.1 模块职责

系统分为三层：**Agent 执行层**（M1~M5）、**核心基础设施层**（config/metrics/model_utils/model_registry）、**编排层**（orchestrator）。

| 模块 | 代码文件 | 职责 |
|------|----------|------|
| M1 数据层 | agents/m1_data_drift.py | 数据预处理、基线模型训练、时序场景漂移注入 |
| M2 监测层 | agents/m2_monitor.py | 双阶段漂移监测，9 项规则三级告警 |
| M3 诊断层 | agents/m3_rootcause.py + m3_llm_agent.py | 双层根因分析 + LLM 四维归因 |
| M4 迭代层 | agents/m4_iteration.py | 三策略自适应模型迭代 |
| M5 部署层 | agents/m5_deploy.py | 四层灰度部署 + 显著性检验 + 防泄漏 |

### 2.2 核心基础设施

| 模块 | 代码文件 | 职责 |
|------|----------|------|
| 配置中心 | core/config.py | 路径、特征列表、阈值、场景规格、模型参数统一配置 |
| 指标库 | core/metrics.py | KS、AUC、PSI、IV、坏账率、通过率、捕获率、DeLong 检验 |
| 模型工具 | core/model_utils.py | 模型训练、保存、加载、打分、选优（门控拆分） |
| Champion 注册表 | core/model_registry.py | 动态冠军指针，替换历史可审计可回放 |

### 2.3 模块交互关系

数据输入 → M1 预处理 + 基线训练 → M2 双阶段监测告警 → M3 双层根因 + LLM 归因 → M4 三策略迭代 → M5 四层灰度部署 → Champion/Challenger 动态替换 → 回到 M2 持续监测。

各模块通过 JSON 报告传递数据：M2 输出 monitor_report.json → M3 读取并输出 rootcause_report.json → M4 读取并输出 train_report.json → M5 读取并输出 deploy_report.json。编排器按状态机流转串联各模块。

## 三、Agent 结构设计

### 3.1 M1 数据层 Agent

- 角色：数据守门员 + 基线建模师
- 流程：读取原始数据 → 四步预处理（缺失校验→异常值截断→共线性剔除→强特征剔除，34→31 特征）→ 三家选优（XGBoost/LightGBM/LR）→ 基线模型落盘 → 时序场景生成（按 SCENARIO_SPEC 注入漂移）
- 输出：preprocess_report.json、基线模型 .pkl、场景数据 CSV

### 3.2 M2 监测层 Agent

- 角色：模型健康监测员
- 流程：读取当前 champion + 当月数据 → 判断标签是否成熟 → 无标签（4 项规则）或有标签（9 项规则）监测 → 计算 KS/AUC/PSI/坏账率 → 三级告警等级判定
- 输出：monitor_report_*.json

### 3.3 M3 诊断层 Agent

- 角色：根因诊断师 + AI 归因师
- 流程：读取监测报告 → L1 指标级诊断（5 项）→ L2 特征级归因（5 项，噪声底校准）→ 四维漂移类型判定 → 策略推荐 → 调用 LLM 四维归因（证据隔离）→ 综合判定
- 输出：rootcause_report_*.json、m3_llm_trace_*.json

### 3.4 M4 迭代层 Agent

- 角色：模型迭代工程师
- 流程：读取根因报告 + 策略推荐 → 按策略准备训练数据 → 训练候选模型 → 门控选优 → 输出候选模型
- 输出：train_report_*.json、候选模型 .pkl

### 3.5 M5 部署层 Agent

- 角色：安全上线守卫
- 流程：读取候选模型 + champion → L1 离线回放 → L2-20% 护栏 → L2-50% 业务指标 → 全量 DeLong+KS Bootstrap → 三态判定（显著优/非劣/显著劣）→ 替换/归档/回滚
- 输出：deploy_report_*.json、champion_registry.json 更新

### 3.6 通信机制

- **JSON 报告传递**：各 Agent 间通过 output/ 目录下的 JSON 文件传递数据，无内存共享，支持断点续跑。
- **事件日志**：events.jsonl 记录状态机每次流转，含时间戳、场景、状态、上下文数据，支持流式读取与审计回放。
- **Champion 注册表**：champion_registry.json 作为全局状态，记录所有替换/归档/复位事件，各 Agent 通过 model_registry.py 读写。
- **编排器协调**：orchestrator.py 按状态机 DATA_READY→MONITORED→DIAGNOSED→ITERATED→DEPLOYED→DONE 串联各 Agent，每轮调用一个 Agent 并检查输出。

## 四、目录结构与文件说明

```
risk_agent_v2/
├── run_demo.py                     # 一键演示入口（--all/--scenario/--oot/--prepare）
├── lean_canvas.py                  # 精益画布 PPTX 生成脚本
├── requirements.txt                # Python 依赖声明
├── .env.example                    # 环境变量模板（DEEPSEEK_APIKEY）
├── __init__.py
│
├── core/                           # 核心基础设施
│   ├── config.py                   # 全局配置：路径/特征/阈值/场景规格/模型参数
│   ├── metrics.py                  # 指标库：KS/AUC/PSI/IV/坏账率/通过率/捕获率/DeLong
│   ├── model_utils.py              # 模型训练/保存/加载/打分/选优（门控拆分）
│   └── model_registry.py           # Champion 注册表（动态冠军指针）
│
├── agents/                         # 五大 Agent 模块
│   ├── m1_data_drift.py            # M1 数据预处理 + 基线训练 + 时序场景漂移注入
│   ├── m2_monitor.py               # M2 双阶段监测（无标签/有标签）
│   ├── m3_rootcause.py             # M3 双层根因分析（L1+L2）+ 综合判定
│   ├── m3_llm_agent.py             # M3 LLM 四维归因（DeepSeek-V4-Flash，证据隔离）
│   ├── m4_iteration.py             # M4 三策略迭代（light/standard/major）
│   └── m5_deploy.py                # M5 四层灰度部署 + 显著性检验 + 防泄漏
│
├── orchestrator/
│   └── orchestrator.py             # V3 编排器（1次OOT + 4场景×3轮月度循环）
│
├── reports/
│   └── docx_report.py              # Word 报告生成（根因/部署/合并报告）
│
├── tools/                          # 辅助工具脚本
│   ├── build_dashboard_data.py     # 聚合 output/ JSON → dashboard_data.json
│   ├── gen_competition_report.py   # 生成赛题总报告（docx）
│   ├── gen_dev_report.py           # 生成模型开发报告（docx）
│   ├── gen_dashboard.py            # 生成 dashboard HTML
│   └── sync_dashboard.py           # Dashboard 数据同步
│
├── models/                         # 模型文件（pickle 制品）
│   ├── model_v2_baseline.pkl       # 基线模型（LightGBM，31特征）
│   ├── model_v2_baseline.meta.json # 基线元数据
│   └── model_v{n}_{scenario}_{strategy}_r{round}.pkl  # 迭代候选模型
│
├── data/
│   ├── train_data.csv              # 赛题官方训练数据（15万×37列）
│   ├── test_data.csv               # 赛题官方测试数据（2万×37列，含真实标签）
│   └── scenarios/                  # 时序场景数据（M1 生成）
│       ├── parallel_A/             # 无漂移场景
│       ├── parallel_B/             # 轻量特征漂移场景
│       ├── parallel_C/             # 中度标签漂移场景
│       └── parallel_D/             # 重度联合漂移场景
│
├── output/                         # 运行产出（JSON 报告 + DOCX 文档）
│   ├── monitor_report_*.json       # 监测报告
│   ├── rootcause_report_*.json     # 根因分析报告（含 LLM trace）
│   ├── m3_llm_trace_*.json         # M3 LLM 四维归因推理记录
│   ├── train_report_*.json         # 迭代训练报告
│   ├── deploy_report_*.json        # 部署评审报告
│   ├── champion_registry.json      # Champion 替换注册表
│   ├── final_comparison_v3.json    # 跨场景最终对比
│   ├── events.jsonl                # 事件日志（状态机流转）
│   └── 交付文档/                   # 正式交付文档（V2 版本）
│       ├── 技术文档V2.docx
│       ├── 根因分析报告_合并V2.docx
│       ├── 模型开发报告V2.docx
│       ├── 赛题交付报告V2.docx
│       ├── 部署上线报告_合并V2.docx
│       └── 精益画布V2.pptx
│
└── dashboard/
    ├── dashboard.html              # 可视化看板（纯静态，双击即开）
    └── dashboard_data.js           # 看板嵌入数据
```

## 五、部署与运行步骤

### 5.1 环境配置

**硬件要求：**
- CPU：x86-64，4 核以上（无需 GPU）
- 内存：8 GB 以上
- 磁盘：2 GB 以上

**操作系统：** Windows 10/11（已验证）、Linux Ubuntu 20.04+（兼容）、macOS（兼容）

**Python 版本：** 3.10 ~ 3.13（已在 3.13.12 验证）

### 5.2 依赖安装

```bash
# 进入项目上级目录
cd "D:/Work Buddy/risk_agent"

# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows (Git Bash):
source .venv/Scripts/activate
# Linux/macOS:
source .venv/bin/activate

# 安装依赖
pip install -r risk_agent_v2/requirements.txt
```

依赖清单见 `requirements.txt`：

| 包名 | 版本 | 用途 |
|------|------|------|
| lightgbm | 4.7.0 | LightGBM 模型训练 |
| xgboost | 3.4.1 | XGBoost 模型训练 |
| scikit-learn | 1.9.0 | LR 模型 / AUC / StandardScaler |
| numpy | 2.5.2 | 数值计算 |
| pandas | 3.0.5 | 数据处理 |
| scipy | 1.18.0 | DeLong 检验 / 统计函数 |
| python-docx | 1.2.0 | Word 报告生成 |
| openai | ≥1.0.0 | DeepSeek API 调用（LLM 归因） |
| python-dotenv | ≥1.0.0 | .env 环境变量加载 |

### 5.3 数据准备

项目已自带赛题官方数据，位于 `data/` 目录下：
- `data/train_data.csv`：训练集（15 万行 × 37 列，坏账率 4.00%）
- `data/test_data.csv`：测试集（2 万行 × 37 列，坏账率 5.00%，含真实标签）

系统按以下优先级自动查找数据目录（可通过环境变量 `RISK_AGENT_DATA_DIR` 覆盖）：
1. 项目内 `data/` 目录（默认，开箱即用）
2. 工作区上级目录
3. 赛题官方数据目录

### 5.4 LLM API 配置

M3 根因分析模块使用 DeepSeek-V4-Flash 大模型进行四维归因推理。需配置 API Key：

```bash
# 步骤 1：从模板创建环境变量文件
cp risk_agent_v2/.env.example .env

# 步骤 2：编辑 .env，填入你的 DeepSeek API Key
# 文件内容格式：
#   DEEPSEEK_APIKEY=你的API密钥
```

获取 API Key：
1. 访问 [DeepSeek 开放平台](https://platform.deepseek.com/) 注册账号
2. 在「API Keys」页面创建新密钥
3. 将密钥填入 `.env` 文件中的 `DEEPSEEK_APIKEY=` 后面

> **注意**：`.env` 文件已在 `.gitignore` 中排除，不会提交到版本库。请勿将 API Key 硬编码到源代码中。

**无 API Key 时的降级机制**：若未配置 `.env` 或 API Key 为空，M3 LLM 归因模块自动降级为纯规则模式（`_failed()` 兜底），保留双层根因分析结果和规则策略推荐，不影响 M1~M5 核心闭环执行。已输出的 `m3_llm_trace_*.json` 中包含此前运行的 LLM 推理记录。

### 5.5 启动命令

```bash
# 必须在 risk_agent_v2 的上级目录（risk_agent/）下运行

# 一键全链路：M1预处理 → OOT → A/B/C/D 三轮月度循环 → 跨场景对比 → 报告生成
python -m risk_agent_v2.run_demo --all

# 仅 M1 预处理 + 基线训练 + 场景生成
python -m risk_agent_v2.run_demo --prepare

# 仅跑 OOT 场景（官方 test 单月）
python -m risk_agent_v2.run_demo --oot

# 单场景三轮流（parallel_A/B/C/D）
python -m risk_agent_v2.run_demo --scenario parallel_B

# 强制重训基线
python -m risk_agent_v2.run_demo --all --force-retrain
```

### 5.6 运行产出

运行后产出全部位于 `output/` 目录：
- JSON 报告：监测/根因/训练/部署报告，每场景每轮一份
- DOCX 文档：根因分析报告、部署上线报告、模型开发报告、赛题总报告
- 模型文件：迭代候选模型 models/model_v{n}_*.pkl
- 注册表：champion_registry.json 记录 Champion 替换历史
- 事件日志：events.jsonl 记录状态机流转
- 可视化：dashboard/dashboard.html（双击即开）

### 5.7 快速验证

```bash
# 最小验证：仅跑 OOT 场景（约 30 秒）
python -m risk_agent_v2.run_demo --oot
# 预期：scenario=oot_2026_01, strategy=standard, champion 替换为候选模型
```

## 六、核心配置说明

### 6.1 特征工程

- 原始特征：34 个（排除 id_card/apply_time/is_bad）
- 强特征剔除：login_fail_count（单特征 AUC>0.80，反欺诈镜像特征）、max_overdue_days
- 共线性剔除：consumption_level（与 income_level |corr|=0.803）
- 最终特征：31 个

### 6.2 监测阈值

| 规则 | 指标 | LOW | MEDIUM | HIGH |
|------|------|-----|--------|------|
| R1 | KS 下降率 | <5% | 5~15% | >15% |
| R2 | AUC 下降率 | <3% | 3~10% | >10% |
| R3 | 最大特征 PSI | <0.10 | 0.10~0.25 | >0.25 |
| R4 | 群体漂移占比 | <10% | 10~30% | >30% |
| R5 | 模型分 PSI | <0.10 | 0.10~0.25 | >0.25 |
| R6 | 坏账率变化 | <1pp | 1~2pp | >2pp |

### 6.3 场景规格

| 场景 | 描述 | 特征漂移 | 标签翻转 | 预期策略 |
|------|------|----------|----------|----------|
| oot_2026_01 | 真实 OOT | — | — | standard |
| parallel_A | 无漂移 | 无 | 无 | none |
| parallel_B | 轻量特征漂移 | age/city_tier/income_level | 无 | light |
| parallel_C | 中度标签漂移 | 无 | +0.3pp | standard |
| parallel_D | 重度联合漂移 | 5特征 | +0.8pp | major |

## 七、实验结果摘要

基于 `output/final_comparison_v3.json` 的实际运行结果：

| 场景 | 轮次 | 策略 | 候选 KS | 候选 AUC | 决策 |
|------|------|------|---------|----------|------|
| OOT | r1 | standard | 0.6957 | 0.9213 | 非劣上线 |
| B | r2 | light | 0.7709 | 0.9529 | 显著优上线 |
| B | r3 | light | 0.7795 | 0.9552 | 显著优上线 |
| C | r1 | standard | 0.7459 | 0.9367 | 非劣上线 |
| C | r2 | standard | 0.7678 | 0.9420 | 非劣上线 |
| C | r3 | standard | 0.7869 | 0.9444 | 非劣上线 |
| D | r1 | light | 0.7503 | 0.9482 | 显著优上线 |
| D | r2 | light | 0.7726 | 0.9535 | 显著优上线 |
| D | r3 | major | 0.7619 | 0.9407 | 非劣上线 |

- **9 次迭代全部成功上线**（4 次显著优替换 + 5 次非劣替换）
- **0 次回滚**（所有灰度护栏均通过）
- **A 场景（无漂移）三轮均不迭代**，验证不误报

## 八、技术文档索引

| 文档 | 路径 | 内容 |
|------|------|------|
| 技术文档 | output/交付文档/技术文档V2.docx | 技术路线/架构/实验设计/量化指标 |
| 根因分析报告 | output/交付文档/根因分析报告_合并V2.docx | 全场景根因分析（含 LLM 四维归因） |
| 部署上线报告 | output/交付文档/部署上线报告_合并V2.docx | 全场景灰度部署评审 |
| 模型开发报告 | output/交付文档/模型开发报告V2.docx | 按附件1模板的模型开发报告 |
| 赛题总报告 | output/交付文档/赛题交付报告V2.docx | 赛题任务目标/技术指标/成果交付 |

## 九、许可证

本项目为参赛作品，仅用于第五届中国研究生金融科技创新竞赛。
