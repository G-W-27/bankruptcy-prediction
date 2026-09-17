# 上市公司破产风险预测

用 SEC 公开财报数据预测美国上市公司未来一年是否破产。在正样本约 **0.64%** 的极度不平衡设定下，XGBoost 测试集 PR-AUC 为 **0.0940**（约随机基线的 **10.3 倍**），可作为审计与风控的前置筛查工具。

---

## 项目概述

破产事件极稀缺，漏检成本高，人工逐份阅读 10-K / 10-Q 成本高、覆盖慢。本项目把问题定义为：给定财报时点可观测的财务特征，预测该公司在未来 12 个月内是否进入破产。

目标列：`target_bankrupt_1y`（未来一年是否破产）。

面向用户与使用方式：

| 用户 | 痛点 | 模型价值 |
| --- | --- | --- |
| 审计师 | 样本量大，持续经营评估难以全覆盖 | 输出高风险清单，作为 going-concern 前置筛查 |
| 投资者 | 尾部破产风险难以被估值模型覆盖 | 对高杠杆、小盘标的做预警排序 |
| 银行 / 授信 | 违约损失大，需要可解释预警 | 按概率排序客户，辅助限额与贷后重检 |
| 供应商 | 客户突然破产导致应收账款坏账 | 对账期和授信做差异化管理 |

定位：筛查与排序工具，不是自动拒贷或审计意见。落地方式应为「高分进入人工复核」。

---

## 数据集

数据来自 OpenFundex（Hugging Face：`ttchopper/openfundex`）。底层是 SEC EDGAR 定期报告，经 XBRL 抽取资产负债表、利润表、现金流量表，并衍生比率与质量评分。

四份 parquet 分片（列式存储，适合宽表，未纳入 Git）：

- `train_clean.parquet`
- `validation_clean.parquet`
- `test_clean.parquet`
- `recent_clean.parquet`

Notebook 实测规模：

| 分片 | 原始行数 | 建模用样本（去掉标签 NaN） | 正样本比例 |
| --- | --- | --- | --- |
| train | 216,023 | 216,023 | 0.6379%（1,378 / 216,023） |
| validation | 43,875 | 43,874 | 0.3715%（163 条正样本） |
| test | 46,039 | 46,024 | 0.9082% |
| recent | 42,204 | 未用于训练和评估 | — |

四片合计约 34.8 万条。公司家数与完整日历跨度未在 notebook 中打印；根据已有项目说明推断为约 1.1 万+ 家公司、约 2008–2026 年。时间切分同样未在 notebook 中打印 `fiscal_year`，根据已有 README 推断如下：

| 数据集 | 时间范围（根据已有 README 推断） | 设计意图 |
| --- | --- | --- |
| 训练集 | 2008–2019 | 覆盖金融危机及之后的长周期 |
| 验证集 | 2020–2021 | 疫情冲击期，用于模型对比 |
| 测试集 | 2022–2023 | 严格时间外泛化 |
| recent | 约 2024 及以后（根据分片命名推断） | 一年标签窗口可能尚未闭合 |

必须用时间划分，不能随机划分。破产标签依赖未来窗口；随机打散会造成时间泄漏、同一公司相邻财报泄漏，并高估真实可部署性。测试集事件率（0.91%）高于验证集（0.37%），本身说明存在时间分布漂移。

---

## 特征工程

原始表 120 列。原则：只用财报时点可获得的信息，删除会直接泄露答案的字段。

### 原始列分类

| 类别 | 数量 | 说明 |
| --- | --- | --- |
| 全部列 | 120 | 标识、财报科目、衍生指标、标签、QA、元数据 |
| 目标列 `target_*` | 35 | 1y/2y 增长、生存、破产、分位排名等，全部不作为 X |
| 标识符（第一轮规则） | 6 | `cik`, `ticker`, `company_name`, `sic`, `fiscal_year`, `fiscal_quarter` |
| QA 标志 | 3 | `qa_impossible_value`, `qa_temporal_anomaly`, `qa_pass` |
| 其余候选特征 | 79 | 再剔除泄漏、元数据、QA、可疑标签 |

第一轮标识符未包含 `adsh` 和日期列，它们出现在 79 个候选中，第二轮作为元数据删除。

### 删除字段及原因

从 79 个候选中删除 15 列后剩余 64 列。

| 列名 | 类型 | 原因 |
| --- | --- | --- |
| `is_bankrupt` | 泄漏 | 是否已破产的事实标记 |
| `bankruptcy_date` | 泄漏 | 破产日期，未来信息 |
| `bankruptcy_chapter` | 泄漏 | 破产章节，未来信息 |
| `label_timestamp` | 泄漏 | 与打标流程绑定 |
| `is_distressed` | 可疑代理 | 可能由当期数据计算，但与破产高度相关，先删 |
| `adsh` | 元数据 | 单份报告 ID |
| `filing_date` | 元数据 | 申报日期 |
| `period_end_date` | 元数据 | 期末日期；时间信息已由分片使用 |
| `source_file` | 元数据 | 数据工程字段 |
| `parse_timestamp` | 元数据 | 解析时间戳 |
| `enrichment_timestamp` | 元数据 | 富化时间戳 |
| `split` | 元数据 | 划分标记 |
| `qa_impossible_value` | QA | 不可能值标记，不是财务驱动因素 |
| `qa_temporal_anomaly` | QA | 时间异常标记 |
| `qa_pass` | QA | 是否通过 QA |

本 notebook 没有按 `qa_pass` 过滤行，只是不把 QA 当特征（根据代码推断）。

### 分类变量

| 列 | 训练集分布 | 处理 |
| --- | --- | --- |
| `z_prime_zone` | NaN 135,847；distress 45,067；grey 24,474；safe 10,635 | 删除。缺失约 62.9%，且与 `z_prime_score` 重叠 |
| `sic` | 高基数行业代码 | 删除。稀疏、易记行业壳；第一轮已列入标识符 |
| `is_financial` | False 162,002；True 54,021（约 25%） | 保留。金融公司报表结构和 Z' 适用性不同 |

### 缺失值

在当时 64 个候选特征上，训练集缺失率分布：

| 缺失率区间 | 特征数 |
| --- | --- |
| > 50% | 26 |
| 5%–50% | 28 |
| < 5% | 10 |

缺失率最高的 20 个特征：

| 特征 | 缺失率 |
| --- | --- |
| `f_score` | 92.48% |
| `short_term_investments` | 85.44% |
| `dividends_paid` | 84.47% |
| `free_cash_flow_margin` | 77.59% |
| `f_delta_lever` | 76.70% |
| `composite_quality_score` | 75.78% |
| `free_cash_flow` | 75.58% |
| `research_development` | 75.17% |
| `f_delta_margin` | 74.71% |
| `debt_to_equity` | 70.52% |
| `long_term_debt` / `total_debt` | 69.04% |
| `gross_margin` | 66.55% |
| `capex` | 65.69% |
| `accrual_ratio` | 65.37% |
| `gross_profit` | 64.92% |
| `cash_conversion_ratio` | 64.89% |
| `operating_cash_flow` | 64.66% |
| `graham_number` | 63.46% |
| `z_prime_zone` | 62.89% |

删留规则（对应 notebook 注释：删除缺失率 >80% + 分类变量 + 重复衍生指标）：

| 规则 | 操作 | 依据 |
| --- | --- | --- |
| >80% 删除 | `f_score`, `short_term_investments`, `dividends_paid` | 有效样本过少 |
| 50%–80% 择优保留 | 现金流、债务、毛利、应计等 | 对破产机制关键；XGBoost 可原生处理 NaN |
| 50%–80% 中再删重复衍生 | `f_delta_lever`, `f_delta_margin`, `composite_quality_score`, `research_development`, `graham_number` | 缺失高且与已留指标重叠 |
| <50% 默认保留 | 多数资产负债表与利润表科目 | 覆盖面足够 |

### 最终 55 个特征

| 类别 | 特征 |
| --- | --- |
| 资产 | `total_assets`, `current_assets`, `cash_and_equivalents`, `accounts_receivable`, `inventory`, `property_plant_equipment`, `goodwill`, `intangible_assets` |
| 负债 | `total_liabilities`, `current_liabilities`, `accounts_payable`, `long_term_debt`, `total_debt` |
| 权益与每股价值 | `stockholders_equity`, `retained_earnings`, `shares_outstanding`, `book_value_per_share`, `ncav_per_share`, `tangible_book_value_per_share`, `net_working_capital_per_share` |
| 利润表 | `revenue`, `cost_of_revenue`, `gross_profit`, `operating_income`, `net_income`, `eps_basic`, `eps_diluted`, `interest_expense`, `income_tax_expense`, `depreciation_amortization`, `sga_expense` |
| 现金流 | `operating_cash_flow`, `capex`, `free_cash_flow` |
| 比率 | `gross_margin`, `operating_margin`, `net_margin`, `roe`, `roa`, `debt_to_equity`, `current_ratio`, `cash_conversion_ratio`, `accrual_ratio`, `free_cash_flow_margin` |
| 评分 / 行业 | `f_roa`, `f_cfo`, `f_delta_roa`, `f_accrual`, `f_delta_liquid`, `f_equity`, `f_delta_turn`, `z_prime_score`, `beneish_coverage`, `graham_defensive_score`, `is_financial` |

标签清洗后矩阵形状：`X_train (216023, 55)`，`X_val (43874, 55)`，`X_test (46024, 55)`。

---

## EDA

训练集类别分布：

| 类别 | 样本量 | 比例 |
| --- | --- | --- |
| 未在一年内破产（负类） | 214,645 | 99.36% |
| 一年内破产（正类） | 1,378 | **0.6379% ≈ 0.64%** |

各分片正样本率：训练集 0.64%，验证集 0.37%，测试集 0.91%。事件率随时间变化，不是平稳分类问题。

不能用准确率。若全部预测「不破产」，训练集准确率约 99.36%，验证集约 99.63%，测试集约 99.09%。这等于从不预警，漏检成本最高的一类全错。

主指标用 PR-AUC。ROC-AUC 看 TPR vs FPR，负类极多时假阳性被稀释，曲线容易虚高。PR-AUC 围绕正类的精确率–召回率；随机分类器的 PR-AUC 约等于正样本率。本项目随机基线：验证集 **0.0037**，测试集 **0.0091**。

---

## 建模

二分类，输出破产概率，用排序质量评估。Notebook 未锁定业务阈值。

### 逻辑回归 Baseline

流水线：中位数填充 → 标准化 → 逻辑回归。

| 设置 | 取值 | 作用 |
| --- | --- | --- |
| 填充 | `SimpleImputer(strategy='median')` | 抗极值，适合财务厚尾 |
| 标准化 | `StandardScaler` | 系数可比较，优化更稳 |
| 类别权重 | `class_weight='balanced'` | 按频率加权，避免全预测负类 |
| 其他 | `max_iter=1000`, `random_state=42` | 收敛与复现 |

局限：线性、填充会扭曲「未披露」信息、对共线科目敏感。用于给出可复现下限。

### XGBoost

直接使用未填充特征（原生处理缺失），验证集作为 `eval_set`。

| 超参数 | 取值 |
| --- | --- |
| `n_estimators` | 300 |
| `max_depth` | 6 |
| `learning_rate` | 0.05 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `scale_pos_weight` | 155.77（训练集负/正 = 214,645 / 1,378） |
| `eval_metric` | `aucpr` |
| `random_state` | 42 |

`scale_pos_weight` 把正类梯度放大约 156 倍。`aucpr` 使评估与 PR-AUC 对齐。未做系统网格搜索。

### 为何 ROC-AUC 容易虚高

验证集上逻辑回归 ROC-AUC 为 0.8556、XGBoost 为 0.9223，看起来很强；同一验证集 PR-AUC 分别只有 0.0329 和 0.1537。模型大体能把正样本排在负样本前面，但高风险名单里仍会混入大量未破产公司。风控关心名单质量，故以 PR-AUC 为主、ROC-AUC 为辅。

---

## 结果

训练集 ROC / PR 未在 notebook 中计算，避免训练集聚光。下表为验证集与测试集。相对倍数 = 模型 PR-AUC / 该集正样本率。

| 模型 | 数据集 | ROC-AUC | PR-AUC | 随机基线 PR-AUC | 相对基线倍数 |
| --- | --- | --- | --- | --- | --- |
| 逻辑回归 | 验证集 | 0.8556 | 0.0329 | 0.0037 | 约 8.9 倍 |
| XGBoost | 验证集 | 0.9223 | 0.1537 | 0.0037 | 约 41.4 倍 |
| 逻辑回归 | 测试集 | 未计算 | 未计算 | 0.0091 | — |
| XGBoost | 测试集 | 0.8818 | 0.0940 | 0.0091 | 约 10.3 倍 |

验证集上 XGBoost 的 PR-AUC 约为逻辑回归的 4.7 倍（0.1537 / 0.0329）。测试集 PR-AUC 从 0.1537 降到 0.0940，与 2020–2021 和 2022–2023 的事件率、宏观环境变化一致。即便下降，测试集仍约是随机基线的 10 倍。逻辑回归未出测试集分数，baseline 泛化对比不完整（根据代码如实说明）。

业务结论：模型可作为审计、风控的前置筛查，按概率输出高风险企业清单供人工复核，不能替代审计意见或授信决策。

---

## SHAP

使用 `shap.TreeExplainer`，在验证集随机抽取 2,000 行，计算 mean(|SHAP|)。

Top 15 特征重要性：

| 排名 | 特征 | mean abs SHAP |
| --- | --- | --- |
| 1 | `net_income` | 0.6009 |
| 2 | `retained_earnings` | 0.5753 |
| 3 | `current_ratio` | 0.5005 |
| 4 | `stockholders_equity` | 0.4125 |
| 5 | `goodwill` | 0.4042 |
| 6 | `cash_and_equivalents` | 0.2847 |
| 7 | `ncav_per_share` | 0.2459 |
| 8 | `tangible_book_value_per_share` | 0.2152 |
| 9 | `total_liabilities` | 0.2108 |
| 10 | `accounts_payable` | 0.1998 |
| 11 | `intangible_assets` | 0.1915 |
| 12 | `interest_expense` | 0.1870 |
| 13 | `operating_income` | 0.1776 |
| 14 | `roa` | 0.1673 |
| 15 | `book_value_per_share` | 0.1632 |

按财务维度分组：

| 维度 | Top15 中的特征 | 业务含义 |
| --- | --- | --- |
| 盈利能力 | `net_income`, `operating_income`, `roa` | 持续亏损是最直接的流量信号 |
| 偿债 / 流动性 | `current_ratio`, `cash_and_equivalents`, `interest_expense` | 短债覆盖、现金缓冲、利息负担 |
| 权益价值 | `stockholders_equity`, `book_value_per_share`, `tangible_book_value_per_share`, `ncav_per_share` | 净资产被掏空或每股清算价值过低 |
| 资产质量 | `goodwill`, `intangible_assets` | 商誉与无形资产生注、账面资产偏虚 |
| 累积状况 | `retained_earnings`, `total_liabilities`, `accounts_payable` | 历史累积盈亏、杠杆、对供应商占款 |

Top 2 为净利润与留存收益，与 Altman Z-Score / Z' 的核心变量一致；第 3 为流动比率。说明模型学到的是真实财务规律（当期不赚钱、累积盈余为负、短期偿债弱），而不是冷门衍生指标刷分。SHAP 仍是相关性而非因果性。

---

## 局限

| 局限 | 说明 |
| --- | --- |
| 时间分布漂移 | 验证 PR-AUC 0.1537 降到测试 0.0940，模型需定期重训 |
| 正样本极少 | 训练仅 1,378 个正样本，验证仅 163 个，PR-AUC 方差偏大 |
| 相关非因果 | SHAP 高不代表干预该科目就能降低破产概率 |
| 无行业差异化 | 只用 `is_financial` 粗分；金融公司 Z' 不完全适用 |
| 标签窗口 | 法律破产不等于经济违约；recent 分片未评估 |
| Baseline 不完整 | 逻辑回归未出测试集分数 |

改进方向：分行业建模、时间衰减权重、与线性模型堆叠并做概率校准、以 PR-AUC 为目标的超参搜索、按漏检/误报成本定阈值、部署时做特征校验与 drift 监控。

---

## 如何运行

安装依赖：

```bash
pip install -r requirements.txt
```

`requirements.txt` 版本：`pandas==1.3.5`，`numpy==1.21.6`，`scikit-learn==1.0.2`，`xgboost==1.6.2`，`shap==0.42.1`，`matplotlib==3.5.3`，`pyarrow==12.0.1`。另需 Jupyter。若从 Hugging Face 拉数，还需 `datasets`（未写入 requirements.txt，根据 data 说明推断）。

数据加载：

```python
from datasets import load_dataset
ds = load_dataset("ttchopper/openfundex")
```

将四个 parquet 放到 notebook 内核工作目录（代码为 `pd.read_parquet("train_clean.parquet")` 等），或改路径指向 `data/`。

运行分析：

```bash
jupyter notebook notebooks/01_full_analysis.ipynb
```

Notebook 顺序：读入数据 → 选定目标并查看不平衡 → 列分类与去泄漏 → 缺失处理得到 55 特征 → 标签清洗 → 逻辑回归 → XGBoost 验证/测试 → SHAP。`recent_clean.parquet` 仅查看形状，未建模。

项目结构：

```text
bankruptcy-prediction-GitHub/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README_data.md
└── notebooks/
    └── 01_full_analysis.ipynb
```

没有独立 `src/` 训练脚本或已保存模型文件，实验以 notebook 为准。

---

## 面试 Q&A

**Q1：为什么主指标是 PR-AUC，而不是 ROC-AUC 或准确率？**

正样本 0.64%，全预测负类准确率就超过 99%。ROC 的 FPR 被海量负类稀释，验证集上会出现「ROC 0.92 但 PR 只有 0.15」的反差。PR-AUC 的随机基线等于事件率（验证 0.0037、测试 0.0091），便于说明比瞎猜强多少。业务关心高风险名单能抓住多少真破产，对应的是 Precision–Recall。

**Q2：`scale_pos_weight=155.77` 怎么来的？和 `class_weight='balanced'` 有何不同？**

定义为训练集负样本数 / 正样本数，即 214,645 / 1,378 ≈ 155.77，让正类在梯度中的权重与负类总权重相当。逻辑回归用 sklearn 的 `class_weight='balanced'`（按频率自动反比加权）。两者目标类似：一个是线性模型样本权，一个是提升树正类尺度。不加权时模型会几乎只优化负类损失。

**Q3：为什么按时间划分而不是随机划分？**

标签是「未来一年是否破产」。随机划分会让同一公司相邻财报、相近宏观年份同时出现在训练和测试中，造成泄漏和虚高。真实场景永远是用历史模型打未来分数。测试集事件率（0.91%）与验证集（0.37%）不同，PR-AUC 从 0.1537 降到 0.0940，说明时间外更难，也比随机划分更可信。

**Q4：模型能否直接落地拒贷或出具审计意见？**

不能当自动决策。测试 PR-AUC 0.0940 表示排序有信息量，但精确率仍低，Top 名单会有大量误报。正确落地是：概率排序 → 人工复核 → 结合行业与非财务信息。还需要处理概念漂移、金融与非金融报表差异、以及相关非因果。可作为审计扫描、贷后抽检、供应商账期管理的前置层；配上阈值成本、校准和定期重训后，才适合进入半自动流程。
