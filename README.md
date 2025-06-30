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

`MuiceBot-Plugin-Memory` 是一个基于 `LUFY` RAG 方法实现的记忆插件，支持句子级的记忆检索、对话结束后自动生成对话总结和关键总结（用户印象）、甚至是支持 AI 调用 Function Call 函数达到手动记忆的目的

原始论文: [Should RAG Chatbots Forget Unimportant Conversations? Exploring Importance and Forgetting with Psychological Insights](https://arxiv.org/abs/2409.12524v1)

## 检索方程

$$
S = w_A·A + w_P·P + w_L·L + w_{R1}·R1 − w_{R2}·R2
$$

$$
Importance = e^{-\frac{∆t}{S}}
$$

$$
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

- 如果要计算 LUFY 重要性评估中的情感唤起分数，你需要下载 [roberta-large](https://huggingface.co/FacebookAI/roberta-large) 基准模型和专为 LUFY 研究微调的情感唤醒度计算 [best_roberta_large.pth](https://github.com/ryuichi-sumida/LUFY/raw/refs/heads/main/Code/roberta/best_roberta_large.pth?download=) 模型。要使用这些模型，建议使用 4GB 显存以上的显卡

## 安装

向机器人对话：

```
.store install memory
```

（可能会）报错，关闭机器人

在命令行窗口中执行数据库迁移：

```shell
nb orm upgrade
```

**如欲启用 LUFY 中的情感唤醒分数计算:**

在机器人目录下执行(需要国际互联网访问能力):

```shell
cd ./plugins/store/meme
pip install .[roberta]
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
cd ../../..
mkdir models
cd models
git clone https://huggingface.co/FacebookAI/roberta-large --depth 1
```

在新建的 models 文件夹下载并放置微调模型: [best_roberta_large.pth](https://github.com/ryuichi-sumida/LUFY/raw/refs/heads/main/Code/roberta/best_roberta_large.pth?download=)

## 配置

### memory_rag_api_key

- 说明: 访问嵌入模型所需的 API Key

- 类型: str

### memory_rag_embedding_model

- 说明: 嵌入模型名称

- 类型: str

- 默认值: `text-embedding-v4`

### memory_rag_base_url

- 说明: OpenAI 兼容端口的嵌入模型 base_url

- 类型: str

- 默认值: `https://dashscope.aliyuncs.com/compatible-mode/v1`

### memory_session_expire_time

- 说明: 会话过期时间（分钟），在此期间用户若不发送消息则视为对话结束并开始总结对话

- 类型: int

- 默认值: 60

### memory_session_min_epoch

- 说明: 最小会话轮数，会话轮数只有大于或等于时才会触发记忆保存机制

- 类型: int

- 默认值: 5

### memory_summary_model

- 说明: 总结模型配置(记忆总结、重要性估计)，默认使用全局模型

- 类型: Optional[str]

- 默认值: None

### memory_cosine_similarity

- 说明: 余弦相似度阈值，低于此值的将不被回忆（除非进行大量实验，否则不要轻易更改此项）

- 类型: float

- 默认值: 0.8

### memory_retain_proportion

- 说明: 记忆保留比例（在对话中有多少消息会被保存。除非进行大量实验，否则不要轻易更改此项）

- 类型: float

- 默认值: 0.1

### memory_lufy_importance_coefficient

- 说明: "LUFY 模型 importance 权重(a), 当将其值设为 0 时表示不启用 LUFY 模型，改为仅余弦相似度模式（注意：在保存记忆时，目前仍然计算 importance 值）

- 类型: float

- 默认值: 0.1

### memory_lufy_enable_arousal

- 说明: 启用 LUFY 唤醒度计算（先决条件：已下载 roberta-large 模型）

- 类型: bool

- 默认值: False

## 下一步工作

- [X] 支持设定总结模型

- [X] 支持 AI 手动发起检索

- [X] 添加嵌入向量缓存

- [ ] 新增多种记忆种类，提高记忆粒度

- [ ] 新增群聊环境中的记忆

- [ ] 微调适用于中国互联网环境的情感唤醒度计算模型(长期计划)

## Reference

本插件的设计/实现参考了以下论文中的思想：

**Ryuichi Sumida, Koji Inoue, Tatsuya Kawahara**. *Should RAG Chatbots Forget Unimportant Conversations? Exploring Importance and Forgetting with Psychological Insights*, arXiv 2024

<https://arxiv.org/abs/2409.12524v1>