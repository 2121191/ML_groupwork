# 💳 金融欺诈检测任务（机器学习大作业）

本次机器学习大作业的选题为复现并改进一篇发表于 CCF-A 类顶级期刊 *IEEE Transactions on Knowledge and Data Engineering (TKDE)* 的最新论文，论文题目为：

**Enhancing Attribute-driven Fraud Detection with Risk-aware Graph Representation (2025)**

> ✅ 我们参考了作者公开的论文原文与代码，并基于其开源项目进行了分析与一点点改进。

- 📄 [论文地址（IEEE Xplore）](https://ieeexplore.ieee.org/document/10470584)
- 📦 [项目地址（GitHub）](https://github.com/AI4Risk/antifraud)

---

## 说明
> 运行 python feature_engineering/data_process.py 以预处理论文模型所需数据集。

## 📁 项目结构

存储库的组织如下:
- `data/`: 数据集文件;
- `config/`: 不同模型的配置文件;
- `feature_engineering/`: 数据处理;
- `methods/`: 模型的实现;
- `main.py`: 组织所有的模型;
- `requirements.txt`: 依赖项;
- `different_models_test`:论文模型和经典机器学习模型训练测试结果notebook展示;
- `exportToHTML`:html文件

### Data Description

有三个数据集，YelpChi、Amazon 和 S-FFSD，用于模型实验。

<!-- YelpChi and Amazon can be downloaded from [here](https://github.com/YingtongDou/CARE-GNN/tree/master/data) or [dgl.data.FraudDataset](https://docs.dgl.ai/api/python/dgl.data.html#fraud-dataset).

Put them in `/data` directory and run `unzip /data/Amazon.zip` and `unzip /data/YelpChi.zip` to unzip the datasets. -->

YelpChi 和 Amazon 数据集来自 [CARE-GNN](https://dl.acm.org/doi/abs/10.1145/3340531.3411903),其原始源数据可以在这里找到 [this repository](https://github.com/YingtongDou/CARE-GNN/tree/master/data).

S-FFSD 是金融欺诈半监督数据集的模拟和小型版本。S-FFSD 的描述如下：
|Name|Type|Range|Note|
|--|--|--|--|
|Time|np.int32|from $\mathbf{0}$ to $\mathbf{N}$|$\mathbf{N}$ denotes the number of trasactions.  |
|Source|string|from $\mathbf{S_0}$ to $\mathbf{S}_{ns}$|$ns$ denotes the number of transaction senders.|
|Target|string|from $\mathbf{T_0}$  to $\mathbf{T}_{nt}$ | $nt$ denotes the number of transaction reveicers.|
|Amount|np.float32|from **0.00** to **np.inf**|The amount of each transaction. |
|Location|string|from $\mathbf{L_0}$  to $\mathbf{L}_{nl}$ |$nl$ denotes the number of transacation locations.|
|Type|string|from $\mathbf{TP_0}$ to $\mathbf{TP}_{np}$|$np$ denotes the number of different transaction types. |
|Labels|np.int32|from **0** to **2**|**2** denotes **unlabeled**||

## 🚀 训练与评估

如果你想测试原论文中的模型以及变体，请在终端运行以下命令：

```bash
python main.py --method gtan
python main.py --method rgtan
python main.py --method rgtan-n
python main.py --method rgtan-r
python main.py --method rgtan-a
python main.py --method rgtan-aff
```

## Requirements
```
python           3.7
scikit-learn     1.0.2
pandas           1.3.5
numpy            1.21.6
networkx         2.6.3
scipy            1.7.3
torch            1.12.1+cu113
dgl-cu113        0.9.1
tqdm             4.67.1
matplotlib       3.5.3
seaborn          0.12.2
```
## 🙏 致谢
我们衷心感谢原论文作者团队的工作与开源精神，使我们有机会深入学习并复现先进的图神经网络方法在金融欺诈检测中的应用。

此外，我们也感谢本课程的任课教师为我们提供了这个深入探索机器学习与图建模交叉技术的平台。通过这次大作业，锻炼了我们论文阅读与复现能力，也体会到了科研从构想到落地实现的全过程。

希望未来课程中能有更多机会探索如图神经网络、深度表示学习等前沿方向，也期待将这些技术进一步应用到现实的安全与金融风控场景中。
