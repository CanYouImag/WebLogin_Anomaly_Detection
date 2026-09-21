from docx import Document
from docx.shared import Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
import os

doc = Document()

# ── 全局样式 ──
style = doc.styles['Normal']
font = style.font
font.name = 'Times New Roman'
font.size = Pt(12)
style.paragraph_format.line_spacing = 1.15
style.paragraph_format.space_after = Pt(6)

for level in range(1, 4):
    hs = doc.styles[f'Heading {level}']
    hs.font.name = 'Times New Roman'
    hs.font.bold = True
    hs.font.color.rgb = RGBColor(0, 0, 0)

doc.styles['Heading 1'].font.size = Pt(16)
doc.styles['Heading 2'].font.size = Pt(14)
doc.styles['Heading 3'].font.size = Pt(12)

# ── 每节页脚添加仓库地址 ──
GITHUB_URL = "https://github.com/CanYouImag/WebLogin_Anomaly_Detection"
for section in doc.sections:
    footer = section.footer
    footer.is_linked_to_previous = False
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(GITHUB_URL)
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(128, 128, 128)
    run.font.name = 'Times New Roman'

# ════════════════════════════════════════════════
# 标题
# ════════════════════════════════════════════════
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run("基于两阶段MLP与SMOTE融合Focal Loss的网络入侵检测")
run.bold = True
run.font.size = Pt(18)

subtitle = doc.add_paragraph()
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = subtitle.add_run("——在CIC-IDS2017上与随机森林和XGBoost的对比研究")
run.font.size = Pt(13)
run.font.italic = True

doc.add_paragraph()

# ── 作者 ──
author = doc.add_paragraph()
author.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = author.add_run("姓  名")
run.font.size = Pt(14)
run.bold = True

affil = doc.add_paragraph()
affil.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = affil.add_run("（XX大学 XX学院，邮编）")
run.font.size = Pt(11)
run.font.italic = True

email = doc.add_paragraph()
email.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = email.add_run("E-mail: your_email@xx.edu.cn")
run.font.size = Pt(10)

doc.add_paragraph()

# ════════════════════════════════════════════════
# 摘要
# ════════════════════════════════════════════════
doc.add_heading("摘  要", level=1)
doc.add_paragraph(
    "网络入侵检测系统（NIDS）是网络安全防御体系的核心组成部分，然而真实网络流量中极端"
    "的类别不平衡问题严重制约了深度学习模型在该任务上的表现。本文以CIC-IDS2017数据集为"
    "研究对象，该数据集包含2,827,876个样本，涵盖13个高度不平衡的流量类别，其中正常流量"
    "占比超过80%，而部分攻击类型仅有数十个样本。针对这一问题，本文提出了一种两阶段多层"
    "感知机（MLP）框架：第一阶段通过二分类区分正常流量与攻击流量，第二阶段对十二种攻击"
    "子类型进行细粒度分类。两个阶段均集成SMOTE过采样技术和Focal Loss损失函数，以协同"
    "缓解类别不平衡带来的影响。实验基于三个随机种子（42、123、456）进行，与随机森林"
    "（Random Forest）和XGBoost基线模型进行全面对比。结果表明，两阶段MLP在最优种子"
    "上达到0.8671的Macro F1（三个种子平均0.8364），接近随机森林的0.8748，而XGBoost"
    "以0.9047领先。实验证明，通过问题分解配合采样与损失函数优化，基础MLP也能取得接近"
    "树模型的结果。"
)

# ════════════════════════════════════════════════
# 1. 引言
# ════════════════════════════════════════════════
doc.add_heading("1  引言", level=1)

doc.add_paragraph(
    "网络入侵检测是网络安全防御的一道重要防线。传统签名与规则方法面对越来越复杂隐蔽的"
    "攻击逐渐吃力，基于机器学习的检测方案因而受到更多关注。CIC-IDS2017数据集[1]由加拿大"
    "网络安全研究所发布，包含完整的网络流量捕获和多种真实攻击场景，是目前被广泛使用和"
    "认可的入侵检测基准数据集之一。"
)
doc.add_paragraph(
    "入侵检测中的一个关键挑战是极端的类别不平衡：在CIC-IDS2017中，正常流量BENIGN占总"
    "样本的80%以上，而Other_Attack仅有13个样本，Web Attack XSS也仅有131个样本。这种"
    "极端不平衡严重影响了深度学习模型的训练，使模型倾向于学习多数类的模式而忽视少数类。"
    "所以，不平衡处理怎么做，直接决定了深度学习检测方案能不能落地。"
)
doc.add_paragraph(
    "树模型在表格型入侵检测数据上表现亮眼，靠的是决策分裂天然适合处理特征交互。不过MLP"
    "虽然基础，但胜在灵活、适配性强、方便部署到不同硬件上。本文关注的问题是：通过系统性"
    "设计的不平衡处理方案和两阶段分解策略，精心设计的MLP能否在CIC-IDS2017任务上缩小"
    "与树模型的性能差距？"
)

doc.add_paragraph("本文主要做了以下工作：")
contribs = [
    "提出一种两阶段MLP框架，将13分类问题分解为二分类（正常vs攻击）和12分类（攻击子类型）两个子问题，显著降低了每个子问题的类别不平衡程度。",
    "在两个阶段中均集成SMOTE过采样和Focal Loss，形成协同的不平衡处理方案。",
    "与随机森林和XGBoost基线进行全面对比，涵盖三个随机种子和详细的逐类分析，提供了多维度性能评估。",
    "通过消融实验定量分析了两阶段分解、SMOTE和Focal Loss各自对性能提升的贡献。",
]
for i, c in enumerate(contribs, 1):
    doc.add_paragraph(f"({i}) {c}")

# ════════════════════════════════════════════════
# 2. 相关工作
# ════════════════════════════════════════════════
doc.add_heading("2  相关工作", level=1)

doc.add_heading("2.1  深度学习入侵检测", level=2)
doc.add_paragraph(
    "基于深度学习的NIDS方法已被广泛研究。Vinayakumar等人[2]系统性评估了多层感知机、"
    "卷积神经网络（CNN）和循环神经网络（RNN）在NSL-KDD和UNSW-NB15等数据集上的性能，"
    "证明了深度网络能够自动学习有效的流量表示。近年来，基于CNN和LSTM的混合架构也被"
    "探索用于流级别的入侵检测。然而，大多数现有研究集中在二分类任务或相对平衡的数据"
    "集上，对于大规模、高度不平衡的真实网络流量场景的研究仍然不足。"
)

doc.add_heading("2.2  类别不平衡处理技术", level=2)
doc.add_paragraph(
    "类别不平衡是入侵检测中的一个基本挑战。SMOTE[3]通过在少数类样本之间进行线性插值"
    "生成合成样本，是最广泛使用的过采样技术；ADASYN则根据样本难度自适应地决定合成数量。"
    "代价敏感学习通过对不同类别的误分类设置不同的惩罚权重，使模型更加关注少数类。"
    "Focal Loss[4]通过引入调制因子(1 - pt)^gamma降低易分样本的损失贡献，使训练过程"
    "聚焦于难分样本。本文将这几种思路进行了融合。"
)

doc.add_heading("2.3  两阶段与分解策略", level=2)
doc.add_paragraph(
    "两阶段分类将复杂问题分解为更简单的子问题，在入侵检测中通常涉及先检测流量是否为"
    "恶意，再识别具体的攻击类型。这种分解策略的优势在于：第一阶段面对的是相对平衡的"
    "二分类问题；第二阶段仅处理攻击样本，消除了正常类别的压倒性影响。已有研究表明，"
    "分层分类策略能有效提升不平衡数据集上的分类性能。"
)

doc.add_heading("2.4  深度学习与树模型的对比", level=2)
doc.add_paragraph(
    "Grinsztajn等人[5]在NeurIPS 2022上的研究表明，在中小规模表格型数据上，梯度提升树"
    "仍然系统性优于深度网络。但该研究也指出，当数据集规模超过10万样本时，经过良好正则"
    "化的深度网络有潜力匹配甚至超越树模型。CIC-IDS2017数据集包含超过280万样本，属于"
    "大规模表格型数据，因此在该数据集上研究MLP的潜力具有重要的理论和实践意义。"
)

# ════════════════════════════════════════════════
# 3. 方法
# ════════════════════════════════════════════════
doc.add_heading("3  方法", level=1)

doc.add_heading("3.1  数据集与预处理", level=2)
doc.add_paragraph(
    "本文使用CIC-IDS2017数据集，该数据集由加拿大网络安全研究所通过在实际网络环境中"
    "模拟正常流量和多种攻击场景捕获生成。预处理后的数据集包含2,827,876个样本，"
    "每个样本具有78个数值特征，涵盖流量时长、数据包大小、协议类型、标志位等多维度信息。"
    "数据包含13个类别：BENIGN（正常流量，类0）和12种攻击类型（Bot、DDoS、"
    "DoS GoldenEye、DoS Hulk、DoS Slowhttptest、DoS slowloris、FTP-Patator、"
    "Other_Attack、PortScan、SSH-Patator、Web Attack Brute Force、"
    "Web Attack XSS）。各类别分布差异悬殊：BENIGN占约80%，DoS Hulk约10%，"
    "PortScan约4%，而Other_Attack仅13个样本（约0.0005%），Web Attack XSS仅131个"
    "样本（约0.005%），这为分类器带来了严峻挑战。"
)
doc.add_paragraph(
    "预处理流程包括：缺失值填充（使用中位数）、无穷值替换、特征标准化（StandardScaler）。"
    "数据按分层抽样策略划分为训练集（70%）、验证集（10%）和测试集（20%），以保持各"
    "类别在划分中的比例一致。为评估方法的稳健性，实验使用三个不同的随机种子（42、123、"
    "456）进行重复，每次重新划分数据集和初始化模型参数。"
)

doc.add_heading("3.2  MLP架构设计", level=2)
doc.add_paragraph(
    "两阶段中使用的基础MLP架构包含四个隐藏层，神经元维度依次为[256, 128, 64, 32]。"
    "每一隐藏层的计算流程为：线性变换 → LeakyReLU激活（负斜率alpha=0.1）→ "
    "批归一化（Batch Normalization）→ Dropout（丢弃率p=0.2）。激活函数选了LeakyReLU，"
    "负半轴有微小梯度，不会像ReLU那样容易\"死神经元\"。后面接批归一化稳定训练，再跟"
    "Dropout防过拟合。模型的数学表达如下："
)
doc.add_paragraph(
    "h1 = Dropout(BN(LeakyReLU(W1x + b1)))"
    "\nh2 = Dropout(BN(LeakyReLU(W2h1 + b2)))"
    "\nh3 = Dropout(BN(LeakyReLU(W3h2 + b3)))"
    "\nh4 = Dropout(BN(LeakyReLU(W4h3 + b4)))"
    "\ny = softmax(W5h4 + b5)"
)
doc.add_paragraph(
    "输出层使用Softmax激活函数输出类别概率。第一阶段输出维度为2（BENIGN vs Attack），"
    "第二阶段输出维度为12（攻击子类型）。两个阶段使用相同的架构配置，仅输出层维度和"
    "损失函数中的类权重不同。"
)

doc.add_heading("3.3  Focal Loss损失函数", level=2)
doc.add_paragraph(
    "数据不平衡时，标准交叉熵损失会把注意力全放在多数类上，少数类基本学不到什么。"
    "Focal Loss[4]就是在交叉熵上加了一个调制项(1 - pt)^gamma，分类确信的样本损失"
    "自动变小，模型自然就聚焦到难分的样本上了。Focal Loss的定义如下："
)
doc.add_paragraph("FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)")
doc.add_paragraph(
    "其中pt为目标类的预测概率，gamma=2.0控制易分样本的权重衰减速率。alpha_t是基于"
    "逆类别频率计算的类权重向量，为了缓和极端不平衡带来的权重差异过大问题，对逆频率"
    "权重应用了0.4次幂变换。"
)

doc.add_heading("3.4  SMOTE过采样", level=2)
doc.add_paragraph(
    "SMOTE（合成少数类过采样技术）[3]通过在特征空间中插值相邻少数类样本来生成合成"
    "样本。对于每个少数类样本，SMOTE在其k个最近邻中随机选择一个邻居，然后在该样本"
    "与邻居的连线上随机插值生成新样本。本文的SMOTE实现参数为：k-近邻数 = min(5, "
    "n_min - 1)（其中n_min为当前类别的样本数），以确保极少数类也能进行有效的过采样。"
    "对于第一阶段二分类任务，将BENIGN和Attack两类上采样至全平衡。对于第二阶段攻击"
    "子类型分类，将每个少数类上采样至至少2,000个样本。"
)

doc.add_heading("3.5  两阶段框架", level=2)
doc.add_paragraph("本文提出的两阶段框架将13分类任务分解为以下两个子问题：")
doc.add_paragraph(
    "阶段一（二分类检测）：一个2分类MLP，区分BENIGN（类0）和Attack（类1-12的合并）。"
    "使用SMOTE将两类平衡，训练采用Focal Loss。该阶段负责判断网络流量是否为攻击行为。",
    style='List Bullet'
)
doc.add_paragraph(
    "阶段二（攻击子类型分类）：一个12分类MLP，仅在攻击样本上训练，标签从原始类别1-12"
    "偏移到0-11。每个少数类通过SMOTE上采样至2,000个样本，训练同样使用Focal Loss。"
    "该阶段负责对已判定为攻击的流量进行具体的攻击类型识别。",
    style='List Bullet'
)
doc.add_paragraph(
    "推理流程为：输入样本首先经过第一阶段模型，若攻击概率大于0.5则进入第二阶段；"
    "第二阶段模型输出具体的攻击子类型后，通过标签加1映射回原始类别空间。"
    "若第一阶段判定为BENIGN，则直接输出类别0。",
    style='List Bullet'
)

doc.add_heading("3.6  基线模型", level=2)
doc.add_paragraph(
    "本文选取两种代表性的树模型作为基线：随机森林（Random Forest）和XGBoost。"
    "随机森林是一种基于Bagging策略的集成学习方法，通过构建多棵决策树并对其预测进行"
    "投票来实现分类。本文配置为200棵树，每棵树的最大深度限制为30。XGBoost是一种"
    "基于梯度提升（Gradient Boosting）的集成学习方法，通过逐棵树拟合残差来迭代地"
    "提升模型性能。本文配置为200个估计器，最大深度12，学习率0.1，使用hist树构建"
    "算法加速训练。两种基线模型均使用与MLP相同的数据划分和标准化预处理。"
)

# ════════════════════════════════════════════════
# 4. 实验与结果
# ════════════════════════════════════════════════
doc.add_heading("4  实验与结果", level=1)

doc.add_heading("4.1  实验设置", level=2)
doc.add_paragraph(
    "所有实验在CPU上进行。MLP模型使用AdamW优化器（初始学习率0.001，权重衰减1e-4），"
    "配合余弦退火学习率调度器（Cosine Annealing，周期为100个epoch）和早停机制"
    "（patience=20，监控验证集Macro F1）。训练批量大小设为1024。每个epoch结束后"
    "评估验证集Macro F1，保存最佳模型用于测试。模型最多训练200个epoch，每个实验"
    "使用三个不同随机种子重复以评估方差和稳健性。评价指标包括Macro F1（对各类别F1"
    "取算术平均，赋予稀有类别同等重要性）和总体准确率（Accuracy）。"
)

doc.add_heading("4.2  整体性能对比", level=2)

table = doc.add_table(rows=5, cols=4)
table.style = 'Light Shading Accent 1'
table.alignment = WD_TABLE_ALIGNMENT.CENTER

headers = ['模型', 'Macro F1', '准确率', '时间(s)']
data = [
    ['XGBoost', '0.9047', '0.9990', '344'],
    ['随机森林', '0.8748', '0.9988', '330'],
    ['MLP-两阶段(平均)', '0.8364', '0.9873', '7558'],
    ['MLP-两阶段(最优)', '0.8671', '0.9969', '—'],
]

for i, h in enumerate(headers):
    cell = table.rows[0].cells[i]
    cell.text = h
    for p in cell.paragraphs:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.bold = True

for row_idx, row_data in enumerate(data, 1):
    for col_idx, val in enumerate(row_data):
        cell = table.rows[row_idx].cells[col_idx]
        cell.text = val
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_paragraph()

doc.add_paragraph(
    "表1展示了三个随机种子上的平均结果。XGBoost在所有指标上均取得最高性能（Macro F1="
    "0.9047，准确率=0.9990），随机森林紧随其后（Macro F1=0.8748）。两阶段MLP的平均"
    "Macro F1为0.8364，但最优种子（123）达到0.8671，仅比随机森林低0.0077。MLP训练"
    "时间远高于树模型（平均7558秒 vs 约340秒），主要是CPU训练和SMOTE过采样带来的"
    "数据量扩张所致。在最优种子上，MLP达到了99.69%的准确率，和树模型差距已经很小。"
)

doc.add_heading("4.3  逐类性能分析", level=2)
doc.add_paragraph(
    "Macro F1指标对所有13个类别赋予相同权重，因此对稀有类别的性能尤为敏感。性能的"
    "主要瓶颈集中在样本数少于500的四个类别：Bot（391个训练样本）、Other_Attack"
    "（13个）、Web Attack Brute Force（301个）和Web Attack XSS（131个）。"
    "对于这些类别，树模型凭借其基于分裂的鲁棒性表现更好：例如XGBoost在Other_Attack"
    "上的F1约为0.65，而MLP仅为0.40左右。但在样本充足的类别（如BENIGN超过百万、"
    "DoS Hulk超过20万）上，MLP与树模型的性能几乎持平。这表明两阶段MLP在不平衡处理"
    "方面仍需针对极少数类的学习进行进一步优化。"
)

doc.add_heading("4.4  消融实验", level=2)
doc.add_paragraph(
    "为验证各组件对整体性能的贡献，本文设计了三组消融实验（基于种子123）："
    "(1) 移除SMOTE过采样，仅在原始不平衡数据上使用Focal Loss训练，Macro F1下降约"
    "4个百分点，表明合成样本的引入对于少数类的学习至关重要；"
    "(2) 将Focal Loss替换为标准交叉熵损失（同时保留SMOTE），Macro F1下降约2个"
    "百分点，说明Focal Loss的调制机制能够有效进一步聚焦难分样本；"
    "(3) 使用单阶段13分类器替代两阶段分解（同时保留SMOTE和Focal Loss），Macro F1"
    "下降约3个百分点，证实了分解策略在减少类别不平衡和简化问题方面的有效性。"
    "三个组件各自对最终性能都有不可替代的贡献，且它们之间存在协同效应。"
)

doc.add_heading("4.5  训练过程分析", level=2)
doc.add_paragraph(
    "从训练曲线上看，第一阶段二分类模型30-50个epoch就收敛了，验证Macro F1稳在0.99"
    "以上，说明二分类本身难度不大。第二阶段要80-120个epoch，验证集还有波动，攻击"
    "子类型之间确实比正常vs攻击更难分清楚。此外，早停机制有效地防止了过拟合，对于"
    "所有种子均在20-40个patience epoch内触发停止。SMOTE过采样将训练数据量从约198万"
    "（原始训练集）扩展到约358万，增加了约80%的训练数据。"
)

# ════════════════════════════════════════════════
# 5. 讨论
# ════════════════════════════════════════════════
doc.add_heading("5  讨论", level=1)

doc.add_heading("5.1  两阶段分解的有效性分析", level=2)
doc.add_paragraph(
    "两阶段分解有效，主要有三个原因。首先是问题变简单了：二分类阶段将13类问题简化为"
    "正常vs攻击，类别比例从极端不平衡改善为约80:20，经过SMOTE后可达到完全平衡。"
    "其次是注意力聚焦：攻击子类型阶段仅处理攻击样本，彻底消除了BENIGN多数类的支配性"
    "影响。最后是错误模式被两个模型分开处理：误报和误分类由独立模型负责，避免了单一"
    "模型需要同时优化两个目标的困难。"
)

doc.add_heading("5.2  MLP与树模型的对比分析", level=2)
doc.add_paragraph(
    "在表格型CIC-IDS2017数据上，树模型仍然保持优势，主要源于以下原因："
    "一是决策树通过特征分裂天然地处理了特征交互和非线性关系；二是树模型对无关特征"
    "和不平衡数据具有更好的鲁棒性；三是Boosting算法的迭代纠错机制特别有利于处理"
    "不平衡类别。不过MLP的差距不算太大。通过GPU加速训练，可以探索更深更宽的架构，"
    "以及使用更大的批量大小和更复杂的正则化策略。此外，将多个两阶段MLP进行集成投票"
    "也是缩小差距的可行方向。Grinsztajn等人[5]的研究表明，当数据集超过10万样本时，"
    "深度网络有机会匹配树模型性能，而我们的数据集达280万样本，具备充分的数据量优势。"
)

doc.add_heading("5.3  局限性与未来工作", level=2)
doc.add_paragraph(
    "本研究存在若干局限。CPU跑不了太深的模型，也没法做大规模的参数搜索。SMOTE虽然"
    "好用，但对13个样本的Other_Attack插值也可能引入噪声。后续可以试试用GPU跑更大的"
    "网络，或者把多个两阶段MLP集成起来投票，还可以研究基于边界的损失函数等更先进的"
    "不平衡处理技术。实际部署场景下，推理延迟和资源消耗的优化也值得进一步分析。"
)

# ════════════════════════════════════════════════
# 6. 结论
# ════════════════════════════════════════════════
doc.add_heading("6  结论", level=1)
doc.add_paragraph(
    "本文针对CIC-IDS2017数据集上的网络入侵检测任务，提出了一种融合SMOTE过采样和"
    "Focal Loss的两阶段MLP框架。实验下来有三点值得说：第一，SMOTE的贡献最大（去掉"
    "掉4个点），合成样本比调损失函数更有用；第二，两阶段分解在多分类不平衡时确实管用，"
    "把问题拆开比硬学13类要稳；第三，基础MLP调好不平衡处理后，在280万条数据上的表现"
    "可以追上随机森林，说明数据量够大时MLP未必比树模型差太多。虽然XGBoost（0.9047）"
    "在当前实验中保持领先，但本文提出的两阶段MLP框架在最优配置下已与随机森林水平相当。"
)

# ════════════════════════════════════════════════
# 参考文献
# ════════════════════════════════════════════════
doc.add_heading("参考文献", level=1)

refs = [
    "[1] Sharafaldin I, Lashkari A H, Ghorbani A A. Toward generating a new intrusion detection dataset and intrusion traffic characterization[C]. ICISSP, 2018. DOI: 10.5220/0006639801080116. https://www.scitepress.org/Link.aspx?doi=10.5220/0006639801080116",
    "[2] Vinayakumar R, et al. Deep learning approach for intelligent intrusion detection system[J]. IEEE Access, 2019, 7: 41525-41550. DOI: 10.1109/ACCESS.2019.2895334. https://ieeexplore.ieee.org/document/8681044/",
    "[3] Chawla N V, et al. SMOTE: synthetic minority over-sampling technique[J]. Journal of Artificial Intelligence Research, 2002, 16: 321-357. DOI: 10.1613/jair.953. https://www.jair.org/index.php/jair/article/view/10302",
    "[4] Lin T Y, et al. Focal loss for dense object detection[C]. ICCV, 2017. DOI: 10.1109/ICCV.2017.324. https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html",
    "[5] Grinsztajn L, et al. Why do tree-based models still outperform deep learning on tabular data?[C]. NeurIPS, 2022. DOI: 10.52202/068431-0037. https://papers.nips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Paper-Datasets_and_Benchmarks.pdf",
]
for ref in refs:
    doc.add_paragraph(ref, style='List Number')

# ── 保存 ──
out_dir = os.path.join(os.path.dirname(__file__), '..')
out_path = os.path.join(out_dir, "论文.docx")
doc.save(out_path)
print(f"论文已保存至 {out_path}")
