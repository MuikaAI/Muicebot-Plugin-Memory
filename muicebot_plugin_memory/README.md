<div align=center>
  <img width=200 src="https://bot.snowy.moe/logo.png"  alt="image"/>
  <h1 align="center">MuiceBot-Plugin-Memory</h1>
  <p align="center">Muicebot RAG 记忆插件✨</p>
</div>
<div align=center>
  <a href="https://nonebot.dev/"><img src="https://img.shields.io/badge/nonebot-2-red" alt="nonebot2"></a>
  <img src="https://img.shields.io/badge/Code%20Style-Black-121110.svg" alt="codestyle">
  <a href='https://qm.qq.com/q/v8BrBiEcuc'><img src="https://img.shields.io/badge/QQ群-MuiceHouse-blue" alt="QQ群组"></a>
</div>

## 介绍✨

`MuiceBot-Plugin-Memory` 是一个基于 `LUFY` RAG 方法实现的记忆插件，支持句子级的记忆检索、对话结束后自动生成对话总结和关键总结（用户印象）

原始论文: [Should RAG Chatbots Forget Unimportant Conversations? Exploring Importance and Forgetting with Psychological Insights](https://arxiv.org/abs/2409.12524v1)

## 检索方程

$$
S = w_A·A + w_P·P + w_L·L + w_{R1}·R1 − w_{R2}·R2
Importance = e^{-\frac{∆t}{S}}
Score = Cos. Sim. + α · Importance
$$

其中：

| 变量               | 符号     | 权重   |
| ------------------ | -------- | ------ |
| 唤醒               | $w_A$    | 2.76   |
| 困惑度             | $w_P$    | -0.28  |
| LLM 估计的重要性   | $w_L$    | 0.44   |
| 最相关记忆计数     | $w_{R1}$ | 1.02   |
| 第二最相关记忆计数 | $w_{R2}$ | -0.012 |

- **情感唤起**（A）：使用 RoBERTa（Liu et al., 2019）结合 EMOBANK（Buechel and Hahn, 2017）微调，捕捉用户发言中的情绪强度。

- **惊喜元素**（P）：通过使用 GPT2-Large 模型评估困惑度，表示发言的不可预测性。在本项目中由于此变量对重要性分数的贡献不显著故设置为 0

- **LLM 估计的重要性**（L）：由语言模型估算的用户发言的重要性。

- **检索引起的遗忘**（R1, R2）：一个记忆出现在前 2 个相关记忆中的频率。最相关的记忆得到强化，而第二个最相关的记忆则有选择地不提及以促进遗忘（Hirst and Echterhoff, 2012）。

## 先决条件

- `muicebot` 版本号大于等于 `1.0.2` （还没发布）

- 要启用 LUFY 重要性评估且计算情感唤起分数，你需要下载 [roberta-large](https://huggingface.co/FacebookAI/roberta-large) 基准模型和专为 LUFY 研究微调的情感唤醒度计算 [best_roberta_large.pth](https://github.com/ryuichi-sumida/LUFY/raw/refs/heads/main/Code/roberta/best_roberta_large.pth?download=) 模型。要使用这些模型，建议使用 8GB 显存以上的显卡

在机器人目录下执行(需要国际互联网访问能力):

```shell
mkdir models
cd models
git clone https://huggingface.co/FacebookAI/roberta-large --depth 1
```

在新建的 models 文件夹下载并放置微调模型: [best_roberta_large.pth](https://github.com/ryuichi-sumida/LUFY/raw/refs/heads/main/Code/roberta/best_roberta_large.pth?download=)

## 安装

*WIP*

## 配置

*WIP*

## 下一步工作

[] 添加嵌入向量缓存

[] 新增多种记忆种类，提高记忆粒度

[] 新增群聊环境中的记忆

[] 微调适用于中国互联网环境的情感唤醒度计算模型

## 致谢和引用

本插件基于论文 [Should RAG Chatbots Forget Unimportant Conversations? Exploring Importance and Forgetting with Psychological Insights](https://arxiv.org/abs/2409.12524v1) 编写，感谢来自京都大学信息学研究生院的研究者们