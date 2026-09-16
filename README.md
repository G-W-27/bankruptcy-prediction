\# 上市公司破产风险预测模型



用SEC公开财报数据预测公司未来一年是否会破产。模型可辅助审计和风控团队优先分配资源，识别高风险公司。



\## 数据来源



\- \*\*数据集\*\*：OpenFundex（基于SEC EDGAR财报数据）

\- \*\*规模\*\*：35万+条记录，覆盖1.1万+家公司

\- \*\*时间范围\*\*：2008-2026年

\- \*\*目标列\*\*：`target\_bankrupt\_1y`（未来一年是否破产）

\- \*\*正样本比例\*\*：0.64%（极度类别不平衡）



\## 数据划分



严格按时间划分，避免数据泄漏：



| 数据集 | 时间范围 | 记录数 | 正样本比例 |

|---|---|---|---|

| 训练集 | 2008-2019 | 216,023 | 0.64% |

| 验证集 | 2020-2021 | 43,874 | 0.37% |

| 测试集 | 2022-2023 | 46,024 | 0.91% |



\## 方法



\### 特征工程

\- 删除泄露答案的列（破产日期、破产章节等）

\- 删除标识符列（公司ID、股票代码等）

\- 删除缺失率>80%的特征

\- 最终保留55个财务指标



\### 模型对比



| 模型 | 验证集PR-AUC | 测试集PR-AUC |

|---|---|---|

| 逻辑回归（baseline） | 0.0329 | - |

| XGBoost | \*\*0.1537\*\* | \*\*0.0940\*\* |

| 随机基线 | 0.0037 | 0.0091 |



XGBoost的PR-AUC是随机基线的\*\*41倍\*\*（验证集）和\*\*10倍\*\*（测试集）。



\## 模型解释（SHAP）



使用Tree SHAP识别Top 15风险驱动因素：



| 维度 | 特征 |

|---|---|

| 盈利能力 | net\_income, operating\_income, roa |

| 偿债能力 | current\_ratio, cash\_and\_equivalents, interest\_expense |

| 权益价值 | stockholders\_equity, book\_value\_per\_share, tangible\_book\_value\_per\_share, ncav\_per\_share |

| 资产质量 | goodwill, intangible\_assets |

| 累积状况 | retained\_earnings, total\_liabilities, accounts\_payable |



\*\*核心发现\*\*：Top 2特征（净利润、留存收益）与经典Altman Z-Score的核心变量一致，说明模型学到的是真实的财务规律。



\## 结果与局限



\*\*结果\*\*：

\- XGBoost验证集PR-AUC 0.1537，测试集PR-AUC 0.0940

\- 模型在未见过的2022-2023年数据上仍有预测能力（随机基线的10倍）



\*\*局限\*\*：

\- 时间分布漂移导致测试集性能下降，模型需定期重训

\- SHAP解释的是相关性而非因果性

\- 金融公司Z'-Score缺失，需单独处理



\## 如何运行



```bash

\# 1. 安装依赖

pip install -r requirements.txt



\# 2. 数据从HuggingFace加载

from datasets import load\_dataset

ds = load\_dataset("ttchopper/openfundex")



\# 3. 运行notebook

jupyter notebook notebooks/01\_full\_analysis.ipynb

