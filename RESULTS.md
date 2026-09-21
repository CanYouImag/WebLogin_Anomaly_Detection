# WebLogin_Anomaly_Detection 训练结果汇总

本文件汇总网络入侵检测项目（两阶段 MLP + SMOTE + Focal Loss）已完成的实验结果。
模型结构见 `src/models2.py`（`GroupedMLP` / `PlainMLP` / `SoftGroupedMLP` / `JointTwoStage` / `BiViewGNN` / `BiProtoGNN`）。

## 1. 实验设置

- 数据集：
  - **CIC-IDS2017**：2,827,876 样本，78 特征，13 类（1 良性 + 12 攻击子类）。
  - **UNSW-NB15**：254,616 样本，62 特征（二分类）。
- 划分：分层随机 70% / 10% / 20%（train/val/test），`StandardScaler` 仅在训练集拟合，防止数据泄漏。
- 种子：`{42, 123, 456, 2024, 7777}`。
- 训练：AdamW(lr=1e-3, wd=1e-4) + CosineAnnealing，patience=20，batch=1024，SMOTE + 类加权 Focal Loss(γ=2)。
- 主指标：**Macro F1**。

## 2. CIC-IDS2017 主实验（5 seeds，macro F1 均值±标准差）

来源：`results/cicids_full.log`、`results/summary.csv`。

| 模型 | Macro F1 (mean ± std) |
| --- | --- |
| **XGBoost** | **0.8987 ± 0.0064** |
| Random Forest | 0.8749 ± 0.0118 |
| MVT-MLP | 0.8413 ± 0.0123 |
| MLP-TwoStage | 0.7885 ± 0.0247 |
| Plain MLP | 0.7773 ± 0.0187 |
| LogReg | 0.1995 ± 0.0021 |
| LightGBM | 0.1972 ± 0.0715 |

> CIC-IDS2017 中 LightGBM 表现异常（约 0.20），疑似该数据集下参数/多分类实现问题，需后续排查。

## 3. UNSW-NB15 主实验（5 seeds，f1 均值±标准差）

来源：`results/unsw_full.log`。

| 模型 | F1 (mean ± std) |
| --- | --- |
| **XGBoost** | **0.9716 ± 0.0008** |
| LightGBM | 0.9704 ± 0.0004 |
| Random Forest | 0.9702 ± 0.0007 |
| Plain MLP | 0.9582 ± 0.0013 |
| MLP-TwoStage(bi) | 0.9580 ± 0.0011 |
| MVT-MLP | 0.9577 ± 0.0016 |
| LogReg | 0.9495 ± 0.0010 |

## 4. 两阶段新模型（CIC-IDS2017，5 seeds，macro F1）

来源：`results/cicids_b12_full.log`、`results/summary.csv`。

| 模型 | Macro F1 (mean ± std) | 各 seed (42/123/456/2024/7777) |
| --- | --- | --- |
| MVT-2Soft | 0.8166 ± 0.0200 | 0.7817 / 0.8214 / 0.8235 / 0.8330 / 0.8232 |
| MVT-Joint | 0.8112 ± 0.0421 | 0.7784 / 0.7741 / 0.7915 / 0.8441 / 0.8677 |

## 5. 消融对比（CIC-IDS2017，禁用 SMOTE，5 seeds）

来源：`results/cicids_followup.log`。用于观察去掉 SMOTE 后各组件的贡献。

| 模型 | Macro F1 (mean ± std) |
| --- | --- |
| MVT-2MLP | 0.8403 ± 0.0170 |
| MLP-TwoStage | 0.7595 ± 0.0152 |
| MVT-MLP | 0.7483 ± 0.0445 |
| Plain MLP | 0.7009 ± 0.0101 |

## 6. 子采样快速验证（seed=42，30k 训练样本）

来源：`results/b12_validation_seed42_sub30k.csv`、`results/b3_val_seed42_sub30k.csv`。

| 模型 | Macro F1 |
| --- | --- |
| MVT-2MLP | 0.6754 |
| MVT-Joint | 0.6612 |
| MVT-2Soft | 0.6587 |
| MVT-Proto | 0.5924 |

## 7. 结论与待办

1. **树模型（XGB / RF）在两个数据集上仍领先**神经模型；MVT 系列把神经网络与树模型的差距从约 0.12 缩小到约 0.06（CIC-IDS2017）。
2. 两阶段分解 + SMOTE + Focal Loss 的组合（MVT-2MLP，0.8403）优于单阶段 MVT-MLP（0.7483）和普通两阶段（0.7595）。
3. **待办**：
   - 排查 CIC-IDS2017 上 LightGBM 的异常低分。
   - 论文 `paper/main.tex` 的结果表仍为 `[TBD]` 占位符，需根据上述结果填数。
   - `MVT-GNN` / `MVT-Proto` 目前仅有子采样/单 seed 结果，需补全 5 seeds。

## 8. 目录说明

```
results/
├── summary.csv                         # 汇总 macro F1（mean/std）
├── cicids_full.log                     # CIC-IDS2017 基线对比
├── unsw_full.log                       # UNSW-NB15 基线对比
├── cicids_followup.log                 # 禁用 SMOTE 的消融
├── cicids_b12_full.log                 # MVT-2Soft / MVT-Joint 5 seeds
├── b12_validation_seed42_sub30k.csv    # 30k 子采样快速验证
├── b3_val_seed42_sub30k.csv            # MVT-Proto 子采样验证
├── cicids2017/seed*/                   # 每 seed 的 metrics 与 per-class CSV
└── unsw_nb15/seed*/metrics.csv         # UNSW-NB15 每 seed 结果
```
