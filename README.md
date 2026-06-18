# 基于两阶段MLP与SMOTE融合Focal Loss的网络入侵检测

基于 CIC-IDS2017 数据集的网络入侵检测系统，使用两阶段 MLP 框架配合 SMOTE 过采样和 Focal Loss，与随机森林、XGBoost 进行对比实验。

## 项目结构

```
WebLogin_Anomaly_Detection/
├── src/
│   ├── data_loader.py          # CIC-IDS2017 多文件加载、清洗、标签编码
│   ├── feature_engineering.py  # 78维数值特征标准化
│   ├── model.py                # MLP 模型定义（可配置隐藏层）
│   ├── train.py                # 两阶段训练（SMOTE + Focal Loss）
│   ├── baseline.py             # RF / XGBoost 基线对比
│   ├── predict.py              # 单条/批量推理
│   └── evaluate.py             # 独立评估脚本
├── notebooks/
│   └── EDA.ipynb               # 探索性数据分析
├── data/
│   ├── raw/                    # CIC-IDS2017 原始 CSV（需自行下载）
│   └── processed/              # 清洗后的数据
├── models/
│   └── saved/                  # 训练好的模型权重
├── requirements.txt
└── README.md
```

## 功能

- **数据预处理**：合并多文件 CSV、缺失值填充、无穷值替换、特征标准化
- **两阶段分类**：第一阶段二分类（BENIGN vs Attack），第二阶段12分类（攻击子类型）
- **不平衡处理**：SMOTE 过采样 + Focal Loss + 类别权重
- **基线对比**：随机森林、XGBoost，三种子平均结果
- **模型推理**：单条网络流量预测或批量 CSV 预测

## 运行

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载数据

从 [CIC-IDS2017 官方页面](https://www.unb.ca/cic/datasets/ids-2017.html) 下载原始 CSV 文件，放入 `data/raw/MachineLearningCVE/` 目录。

> 需要下载 8 个 CSV 文件，约 2.8 万条记录，合计约 500MB。

### 3. 完整训练

```bash
# 训练 + 评估（包含两阶段MLP和基线对比）
python src/baseline.py

# 或单独训练两阶段MLP
python src/train.py
```

### 4. 推理预测

```bash
# 单条预测
python src/predict.py

```

## 实验结果

| 模型 | Macro F1 | 准确率 |
|------|----------|--------|
| XGBoost | 0.9047 | 0.9990 |
| 随机森林 | 0.8748 | 0.9988 |
| MLP-两阶段（平均） | 0.8364 | 0.9873 |
| MLP-两阶段（最优） | 0.8671 | 0.9969 |

## 依赖

Python 3.10+，主要依赖见 `requirements.txt`。
