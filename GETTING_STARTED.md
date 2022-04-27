# Getting Started with Fastreid

## 准备预训练模型

Feature Extractor会自动下载预训练的模型，但如果你的网络没有连接，你可以手动下载训练前模型，并把它放在`~/.cache/torch/checkpoints`。 .

## 用cython编译以加速计算

```bash
conda install -c anaconda cython
cd fastreid/evaluation/rank_cylib; make all
```

## 图像数据的格式要求
完整的Fast-ExtreID项目需要为每个数据集自定义读取数据集的方法。

目前Feature Extractor使用的模型是通过[Market1501](http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip) 数据集训练的，
你可以选择按照Market1501数据集的格式放置文件，其对应数据集配置文件为`fastreid/data/datasets/market1501.py`

同时你也可以选择更改其配置文件，只要配置文件中的类初始化方法能够得到正确格式的变量`train`, `query`, `gallery`即可


## 命令行中提取特征并评估模型性能

"tools/train_net.py"文件用于训练fast-extreid中提供的所有配置。特征存放于`logs/market1501/bagtricks_R50/latent_feat.csv`

```bash
python tools/train_net.py
```
