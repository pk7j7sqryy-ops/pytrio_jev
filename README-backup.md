# Qwen3.5-4B-jev— 决策模型

**代码与复现资源**
>
> - 完整代码：[`pytrio_jev(https://github.com/pk7j7sqryy-ops/pytrio_jev)**`](https://github.com/pk7j7sqryy-ops/pytrio_jev.git)) / 快速启动：[`start.md`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/09-pytrio-jev/start.md)
> - 方法参考：[`09-pytrio-jev`](https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/09-pytrio-jev) ,[Intern-Decision](https://github.com/InternLM/Intern-Decision)、[Jev 架构逆向](https://archerhume.com/posts/jevs-architecture-unmasked)、[Datawhale AgentJev 教程](https://mp.weixin.qq.com/s/F7i83rn8oDzQ-sg13MKwWA)
> - 训练数据：[`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)、[`Team-ACE/ToolACE`](https://huggingface.co/datasets/Team-ACE/ToolACE)
> - SwanLab：[查看本次实验公开的训练记录](https://swanlab.cn/@js9527/agentic-rl-lab-pytrio-jev/v1/8w9aka/runs?utm_source=website_qr&utm_medium=qr_scan)
> - PyTRIO：[官网](https://pytrio.com) / [官方文档](https://docs.pytrio.com/docs)

训练使用了 2160 条训练数据、2 个 epoch、312 步 LoRA，在与 Intern-Decision 完全一致的评测集上，五项平均分从 base 的 76.30 提到 83.16：Typed Decision 和 ToolACE 两项直接超过 Jev，其余大部分逼近 Jev。
---

## 目录

- [0. 一句话理解](#0-一句话理解)
- [1. 项目结构](#1-项目结构)
- [2. 数据怎么来（准备数据）](#2-数据怎么来准备数据)
- [3. 训练流程](#3-训练流程)
- [4. 数据处理流程（核心）](#4-数据处理流程核心)
  - [4.1 数据长什么样（canonical record）](#41-数据长什么样canonical-record)
  - [4.2 四种题型](#42-四种题型)
  - [4.3 `encode()`：文本 → 模板 → tokens → markers](#43-encode文本--模板--tokens--markers)
  - [4.4 `decision` 占位符：为什么要它](#44-decision-占位符为什么要它)
  - [4.5 `build_sft_datum()`：拼出训练样本](#45-build_sft_datum拼出训练样本)
  - [4.6 为什么中间引号闭合不算 loss](#46-为什么中间引号闭合不算-loss)
- [5. 推理流程（核心）](#5-推理流程核心)
- [6. 快速上手命令](#6-快速上手命令)
- [7. 关键设计决策汇总](#7-关键设计决策汇总)
- [9. 温度校准与评测指标](#9-温度校准与评测指标)
- [10. 温度校准：黄金分割与牛顿法](#10-温度校准黄金分割与牛顿法)

---

## 1.什么是Jev
以下是 Jev 的官方介绍： 
	Jev 并非生成式大语言模型（LLM），它是面向 Agent、软件执行循环的「System One 快速决策模型」。它不输出自由文本、思维链，而是直接输出经过校准概率的结构化决策，专门承担 Agent 流程里的判断、路由、校验闸门工作。

OpenRouter 通俗解读：它不生成文本 Token，是直接输出概率的分类决策模型，适合意图识别、路由判定等二选一 / 多选型决策场景。

使用方式：向模型传入一段背景状态（state）以及一组附带固定候选选项的问题（questions），模型仅返回「选择哪个选项、对应置信概率」。
示例输入： 
状态：客户正在进行商品售后咨询 问题：判断客户当前情绪状态，A 一般，B 紧急，C 非常紧急 
模型输出：选择 A，概率 90%；选项 B，概率 9%；选择 C，概率 1%


这不是我想象出来的需求。OpenRouter 的 "Top models by task" 统计里，**Classification 一个任务就占了全平台 10.4% 的花销**，是 General 类别（31.2%）下最大的单一任务：

![OpenRouter Top models by task：Classification 占全平台 10.4% 花销](./images/openrouter.png)

很有趣的是榜单上的模型：做分类任务花得最多的模型，是 Claude Opus 5、GPT-6 Astra、Gemini 3.1 Pro 这种最贵的一档。也就是说，大量企业和用户正在真金白银地用昂贵的大模型做分类决策。这是真实场景下的消耗，而这部分需求本来用一个便宜几个数量级的决策模型就能做。

这些场景不需要大模型思前想后写几千字分析，**只需要一个快、稳、概率可信的决策**。让一个通用大模型每次都想半天再做这种三选一，是拿慢思考做快思考的活。我的看法是：快思考和慢思考需要协同配合，慢思考负责开放世界的推理与规划，快思考负责高频、固定、可枚举的路由与决策。Jev 这类模型填的就是快思考这一格。


## 2. show结果

评测集与 Intern-Decision 官方榜单完全一致（Jevbench 三档 + Typed Decision test + ToolACE 310 条），均为acc准确率（%）：

| 模型                    |   Easy | Original |  Hard | Typed | ToolACE |        平均 |
| --------------------- | -----: | -------: | ----: | ----: | ------: | --------: |
| Qwen3.5-4B（base）      | 100.00 |    87.50 | 55.86 | 50.40 |   87.74 |     76.30 |
| **PyTrio-Jev（本文）**    | 100.00 |    94.44 | 52.25 | 77.55 |   92.58 | **83.37** |
| **PyTrio-Jev（+温度校准）** | 100.00 |    94.44 | 51.33 | 77.40 |   92.58 | **83.16** |
| Intern-Decision-0.8B  |  97.92 |    80.56 | 52.25 | 77.35 |   94.52 |     80.52 |
| Intern-Decision-2B    | 100.00 |    84.72 | 63.96 | 79.35 |   96.45 |     84.90 |
| Intern-Decision-4B    | 100.00 |    98.61 | 73.87 | 80.55 |   96.45 |     89.90 |
| Jev                   | 100.00 |    98.61 | 72.07 | 73.35 |   91.29 |     87.06 |
| Laya                  |  95.83 |    72.22 | 28.83 | 35.95 |   63.87 |     59.34 |

官方数值来自 [Intern-Decision 公开榜单](https://github.com/InternLM/Intern-Decision)。注意两个口径差异：官方模型在 Typed Decision 上是**零样本**评测，而我们是 specialist 模式（train/test 同 workflow）；官方是全参数微调，我们是 LoRA，且仅仅用了 2400 条通用数据。

Local T=1 为PyTrio-Jev（本文），Local T=0.979 为PyTrio-Jev（+温度校准）
![[0-expand-prj/Decision-Model-Jev/agentic-rl-lab/09-pytrio-jev/images/benchmark-local.svg]]




---

##  项目结构

```
jev/
├── 00-prepare-data.py      下载 + 切分数据集（HF）
├── data.py                 数据处理核心：compile_record / Encoder / build_sft_datum
├── train.py                异步 LoRA masked SFT 训练主循环
├── inference.py            受限 softmax 推理
├── cyber-security-data.py  额外数据源（安全告警，流式采样 200 条）
├── templates/qwen3_5_jev.jinja   钉死的 chat template
├── prompt/                 提示词文件（data / sft 两类）
└── datasets/
    ├── train.jsonl         训练集（typed-decisions 960 + ToolACE 1200）
    ├── dev.jsonl           早停评估集（120 条）
    ├── calibration.jsonl   温度拟合集（120 条）
    └── eval/*.jsonl        五个评测基准
```
## 3. 做了什么

1、数据准备
`00-prepare-data.py` 从 HuggingFace 拉两个数据集，统一转成 **canonical record** 格式：

训练和验证数据：

| 来源                           | 用途     | 切分                                  |
| ---------------------------- | ------ | ----------------------------------- |
| `LocalLLaMA/typed-decisions` | 训练数据   | typed-decisions train 切出 960 案例     |
| `Team-ACE/ToolACE`           | 训练数据   | 采样 1200 条，避开 eval 集                 |
| LocalLLaMA/typed-decisions   | 训练验证数据 | typed-decisions 案例级切分 120 案例（选优用）   |
| LocalLLaMA/typed-decisions   | 温度校准数据 | typed-decisions 案例级切分 120 案例（温度拟合用） |

评估数据：

| 基准                  |              规模 | Intern-Decision-4B 参考 |
| ------------------- | --------------: | --------------------: |
| Jevbench Easy       |              48 |               100.00% |
| Jevbench Original   |              72 |                98.61% |
| Jevbench Hard       |             111 |                73.87% |
| Typed Decision test | 400 案例 / 2000 题 |                80.55% |
| ToolACE             |             310 |                96.45% |

2、训练

`train.py` 是一个 **异步 LoRA masked SFT** 训练循环，整体结构：

```
加载 train/dev JSONL
   │
   ├─ 从头训练：create_lora_training_client(base_model, rank, seed)
   │  或 续训：create_training_client_from_state_with_optimizer(resume_from)
   │
   ├─ 每个 epoch：shuffle → 按 batch 切片
   │     ├─ 每条 record → encoder.build_sft_datum(record)   ← 关键：数据在这里被编码
   │     ├─ forward_backward_async(datums, "cross_entropy")  ← 前向+反向
   │     └─ optim_step_async(adam_params)                    ← 更新
   │
   ├─ 定期：save_weights（推理权重 + 训练状态双份）
   └─ 定期：dev 集前向算 loss → 早停判断
```

几个值得注意的工程点：

- **双份存档**：每个存档点同时保存「推理权重（LoRA，给采样用）」和「训练状态（含优化器，给断点续训）」。
- **异步流水线**：`forward_backward_async` 提交后不等待，loss 计算和 SwanLab 上报交给后台 task，训练不断流。
- **早停**：按 `dev_eval_interval`（占总 step 比例）周期性跑 dev loss，连续 `early_stop_patience` 个评估点没创新低就停。
- **masked loss 对齐**：dev 评估和训练 loss 同构，都是 `loss_sum / supervised_tokens`，只监督决策位置。
数据从「一条结构化记录」到「模型可训练的一个样本」，走三步：
> `compile_record` → `encode` → `build_sft_datum`。


原始jsonl数据长什么样（canonical record）

一条记录包含四块：**状态**（state）、**问题**（questions）、**答案**（targets），外加一个 id。

```jsonc
{
  "id": "tr_security_incidents_000178",
  "state": {                      // 案情：给模型看的输入事实
    "alert":   { "description": "...", "evidence": "...", "rule": "mass_download" },
    "context": { "asset_criticality": "medium", "change_window_active": true, "source_on_allowlist": false },
    "history": { "credential_rotation_days_ago": 3, "distinct_countries_30d": 4, "logins_30d": 688, "prior_alerts_90d": 0 },
    "principal": { "mfa_enrolled": true, "name": "svc-analytics-473", "privileges": ["repo:read"], "type": "ci_runner" }
  },
  "questions": {                  // 要判断哪些字段、各有哪些选项
    "credential_compromise": { "type": "noul",   "instructions": "..." },
    "disposition":          { "type": "choice", "criteria": { "close_benign": "...", "contain": "...", "investigate": "...", "monitor": "..." } },
    "severity":             { "type": "score",  "criteria": ["Negligible...", "Low...", "Moderate...", "High...", "Critical..."] },
    "true_positive":        { "type": "noul",   "criteria": { "false": "...", "true": "..." } },
    "urgency":              { "type": "score",  "criteria": ["...", "...", "...", "..."] }
  },
  "targets": {                    // 正确答案
    "credential_compromise": { "label": "true" },
    "disposition":          { "label": "investigate" },
    "severity":             { "label": "2" },
    "true_positive":        { "label": "true" },
    "urgency":              { "label": "2" }
  }
}

```


数据处理：
a、要那些数据？
- `model_input` = `ids[:-1]`（去掉最后一个 token）
- `target_tokens` = `ids[1:]`（右移一位），其中答案位置被覆盖成正确符号token id
- `weights` = 只在答案位置为 1，其余全 0

b、怎么处理
`build_sft_datum()`：拼出训练样本

`build_sft_datum(record)` 把上面编码好的东西拼成 PyTRIO 的 `Datum`：

```python
encoded = self.encode(record)
ids = encoded["ids"]
target_tokens = list(ids[1:])          # 预测目标 = 输入右移一位
weights = [0.0] * len(target_tokens)    # 默认全 0
for field, symbol in encoded["targets"].items():
    position = encoded["markers"][field] - 1   # ← 关键：marker 前一位置
    target_tokens[position] = self.symbol_token_id(symbol)  # 覆盖成正确符号 id
    weights[position] = 1.0                         # 只有这里算 loss
```

**为什么是 `marker - 1`？** 因为 `markers[field]` 指向的是 `decision` 这个占位 token 的位置，而「预测答案符号」这件事发生在它**前一个位置**——模型在这个位置读的是前面的引号 `"`，要预测的下一个 token 才是答案符号（详见 图示）。


通过代码处理后的实际训练输入数据`model_input`（转tokenid之前）：
```
<|im_start|>system
You are a careful decision assistant. Use the state and decision schema in the user message to make the requested decisions. For every field, choose exactly one answer symbol (e.g. A, B, C, ...) from its listed options and return one valid JSON object mapping each field name to its chosen symbol. Use the field names and symbols exactly as given. Do not include explanations, Markdown, or extra text.<|im_end|>
<|im_start|>user
Return one answer for every field using the supplied answer symbols.

## State
{
  "customer_message": "I bought headphones yesterday and noticed today that I was charged twice. Please refund the extra charge as soon as possible."
}
## Decision schema
team: Which team should handle this ticket?
    A = billing: Payments and refunds
    B = delivery: Shipping and delivery
    C = technical: Product technical support
urgency: Choose the priority for handling this ticket.
    A = 0: Low: general inquiry
    B = 1: Medium: an issue requiring follow-up
    C = 2: High: a financial issue such as a duplicate charge
refund_requested: Does the customer explicitly request a refund?
    A = no: The answer is no (negative, or disagree with the claim).
    B = yes: The answer is yes (affirmative, or align with the claim).<|im_end|>
<|im_start|>assistant
<think>

</think>

{
    "team": "decision",
    "urgency": "decision",
    "refund_requested": "decision"
}<|im_end|>
```
 [248045, 8678, 198,  .......248046, 198]，经过`ids[:-1]`[248045, 8678, 198,  .......248046]
- `248045` → `<|im_start|>`
- `248046` → `<|im_end|>`
- `198` → `\n`（换行符）

target_tokens（转tokenid之前）数据：
```
system
You are a careful decision assistant. Use the state and decision schema in the user message to make the requested decisions. For every field, choose exactly one answer symbol (e.g. A, B, C, ...) from its listed options and return one valid JSON object mapping each field name to its chosen symbol. Use the field names and symbols exactly as given. Do not include explanations, Markdown, or extra text.<|im_end|>
<|im_start|>user
Return one answer for every field using the supplied answer symbols.

## State
{
  "customer_message": "I bought headphones yesterday and noticed today that I was charged twice. Please refund the extra charge as soon as possible."
}
## Decision schema
team: Which team should handle this ticket?
    A = billing: Payments and refunds
    B = delivery: Shipping and delivery
    C = technical: Product technical support
urgency: Choose the priority for handling this ticket.
    A = 0: Low: general inquiry
    B = 1: Medium: an issue requiring follow-up
    C = 2: High: a financial issue such as a duplicate charge
refund_requested: Does the customer explicitly request a refund?
    A = no: The answer is no (negative, or disagree with the claim).
    B = yes: The answer is yes (affirmative, or align with the claim).<|im_end|>
<|im_start|>assistant
<think>

</think>

{
    "team": "A",
    "urgency": "C",
    "refund_requested": "B"
}<|im_end|>

```
 [248045, 8678, 198,  .......248046, 198]，经过`ids[1:]`[8678, 198,  .......248046, 198]
- `248045` → `<|im_start|>`
- `248046` → `<|im_end|>`
- `198` → `\n`（换行符）
- 
loss数据（Brief Loss，稀疏loss）：
{
  "team": "A",            ---实际只算这里数据A的loss，weight为1，其他为0
  "urgency": "B",  ---实际只算这里数据A的loss，weight为1，其他为0
  "refund_requested": "C"
}

相关问题：
`a、decision` 占位符：为什么要它？
	**`decision` 占位符是用来「标记位置」的，不是用来「当答案」的。**

	训练时模型最终需要三样东西：`ids[:-1]`（输入）、`ids[1:]`（预测目标）、`weights`（哪些位置要监督）。前两个都好确定，唯独 **weights 不好定**——因为我们要「只监督答案符号」，得知道答案符号到底在 token 流的哪个位置。

	`decision` 占位符就是干这个的，它有两个作用：

1. **定位（markers）**：`decision` 在 Qwen3.5 词表里是单 token（id 61781），且正文里几乎不会作为独立 token 出现。在骨架里扫它的命中位置，就能精确知道「每个字段的答案该落在哪」→ 得到 markers。
2. **替换成符号**：训练后模型实际输出的是 A/B/C 这些符号，不是 `true`/选项文本/分数。markers 让「把答案位置从 `decision` 覆盖成正确符号 id」这件事变成一行代码。

b、**为什么不用「全错位监督」替代占位符？** 
	全错位监督（每个 token 都算 loss）虽然省事，但会把训练信号浪费在「学格式」上，决策能力被稀释。占位符让「只监督 5 个符号」从「很难」变成「一行代码」。**占位符不是训练的必要条件，而是「稀疏监督」的工程实现手段。**

c、为什么中间引号闭合不算 loss



```
位置:     ...  835         836           837         838   ...
模型读:   ...  引号"      decision      引号"        逗号,   ...
模型预测:      B          → 引号"      → 逗号,       → ...
weight:        1.0 ✅      0.0 ❌        0.0 ❌       0.0 ❌
```

模型在位置 835（读到引号 `"`）预测出答案符号 `B`——**这一个位置算 loss**。而位置 836 预测的是「闭合引号」、837 预测「逗号」……这些是 JSON 格式的固有内容，**不算 loss**。

**根本原因：这里只有一次前向传播，不需要「闭合」——每个字段只用预测一个值（一个符号）。** 引号、逗号这些格式 token 的预测不参与监督，权重置 0。

> 更精确地讲：模型在答案位置看到的是前面的引号 `"`，不是 `decision`。`decision` 从头到尾只扮演「占位 + 定位」两个角色，最后被覆盖掉，从不作为监督目标出现。





 为什么中间引号闭合不算 loss

这是你笔记里反复琢磨的一个点，用图示看清楚：

```
位置:     ...  835         836           837         838   ...
模型读:   ...  引号"      decision      引号"        逗号,   ...
模型预测:      B          → 引号"      → 逗号,       → ...
weight:        1.0 ✅      0.0 ❌        0.0 ❌       0.0 ❌
```

模型在位置 835（读到引号 `"`）预测出答案符号 `B`——**这一个位置算 loss**。而位置 836 预测的是「闭合引号」、837 预测「逗号」……这些是 JSON 格式的固有内容，**不算 loss**。

**根本原因：这里只有一次前向传播，不需要「闭合」——每个字段只用预测一个值（一个符号）。** 引号、逗号这些格式 token 的预测不参与监督，权重置 0。

> 更精确地讲：模型在答案位置看到的是前面的引号 `"`，不是 `decision`。`decision` 从头到尾只扮演「占位 + 定位」两个角色，最后被覆盖掉，从不作为监督目标出现。

3、 推理


推理流程（核心）

推理不生成文本，而是**一次前向，读出所有字段的候选概率**。

```
encoder.encode(record, include_targets=False)   ← 同样的 prompt（骨架仍带 decision 占位）
        │
        ▼
sampling_client.sample(...)                      ← max_tokens=1，要 topk_prompt_logprobs
        │                                          （在 marker 位置读「下一个 token」的 top-k 分布）
        ▼
_field_logprobs：每个字段，从 top-k 里取候选符号的 logprob
        │                                          （候选没进 top-k 的，用 floor = min(topk)-10 近似为概率 0）
        ▼
softmax（可加温度）→ 每个字段的概率分布
        │
        ▼
_build_decisions：按题型出预测结果
        ├─ choice → 最高概率选项
        ├─ noul   → p_yes（yes 概率）
        └─ score  → 期望分（各等级概率加权求和）
```

关键点：

- **`topk_prompt_logprobs[m]`** 是「给定前 m 个 token，第 m 个 token 的分布」；marker 恰好在第 m 位，所以这个位置就是训练时监督的那个预测点——**训练和推理完全对齐**。
- **受限 softmax**：不是在整个词表上 softmax，而是只取该字段的候选符号（如 `A/B/C/D`）过滤后归一化，保证输出一定是合法选项。
- **置信度 `believe`**：`inference.py` 按题型从候选概率分布计算 `[0, 1]` 内的分数；它是分布确定性的指标，不是经过校准的正确率概率，也不执行阈值拒答。
  - `noul`：`|2·p_yes - 1|`，yes/no 越接近五五开，分数越低。
  - `choice`：将最大候选概率相对均匀分布的超出幅度归一化：`max(0, (p_max - 1/n) / (1 - 1/n))`。
  - `score`：先找概率最高的等级，再计算概率加权的等级距离 `spread`，并用均匀分布下的平均距离 `uniform_spread` 归一化：`max(0, 1 - spread / uniform_spread)`。概率集中在某个等级时分数高；分布越分散，分数越低。
  - 空分布返回 `0`；只有一个候选时返回 `1`（`noul` 仍按 yes 概率公式计算）。


在训练中，输入占位符，输出具体选项
但是推理中，是输入带有占位符的input，输出各个占位符号的候选topk的log概率，然后进行softmax，在argmax选择

推理输入：
同上前面的训练输入ids，没有去除最后一位

推理输出：

推理用受限 softmax 而非生成
![受限 softmax 推理](./images/inference-principle.png)

每个位置的topk概率
[None, [(846, -0.20170411467552185), (8678, -1.7017041444778442), (1785, -9.076704025268555)], [(198, -0.01624806970357895), (248046, -4.2662482261657715), (271, -6.6412482261657715)..........],


推理输出处理这个概率：

1、处理对应位置
```
def _field_logprobs(encoder: Encoder, encoded: dict, response) -> dict:  
    """从一次前向的 topk_prompt_logprobs 提取各字段候选符号的 logprob。  
  
    topk_prompt_logprobs[m] 是"给定前 m 个 token，第 m 个 token 的分布"，    marker 在第 m 位，所以该位置就是训练时监督的预测点。    候选未进 top-k 时按 floor = min(topk) - 10 近似（约等于概率 0）。    """    assert response.input_tokens == len(encoded["ids"]), "本地/远端 prompt 长度不一致"  
    fields: dict[str, dict[str, float]] = {}  
    for field, position in encoded["markers"].items():  
        # 读取该字段答案占位符位置的候选 token logprob。  
        topk = dict(response.topk_prompt_logprobs[position])  
        # 候选 token 未出现在 top-k 时，用低分近似其概率为 0。  
        floor = min(topk.values()) - 10.0  
        fields[field] = {  
            symbol: topk.get(encoder.symbol_token_id(symbol), floor)  
            for symbol, _, _ in encoded["options"][field]  
        }  
    return {"encoded": encoded, "logprobs": fields}
```

2、中间处理输出结果
```
{
	'encoded': {
		'messages': [{
			'role': 'system',
			'content': 'You are a careful decision assistant. Use the state and decision schema in the user message to make the requested decisions. For every field, choose exactly one answer symbol (e.g. A, B, C, ...) from its listed options and return one valid JSON object mapping each field name to its chosen symbol. Use the field names and symbols exactly as given. Do not include explanations, Markdown, or extra text.'
		}, {
			'role': 'user',
			'content': 'Return one answer for every field using the supplied answer symbols.\n\n## State\n{\n  "customer_message": "真是太感谢你们了，订单晚到了三天，还顺便多扣了一笔钱，服务真周到。我倒也不是非要你们马上处理，毕竟等账单越滚越大也挺有意思的。"\n}\n## Decision schema\nteam: 应该由哪个团队处理这条投诉？\n    A = billing: 账单、重复扣款和退款\n    B = delivery: 订单延迟、配送和物流问题\n    C = technical: 产品或技术故障\nurgency: 这条投诉的处理优先级是什么？\n    A = 0: 低：一般咨询或不紧急的问题\n    B = 1: 中：需要跟进，但暂时没有明显紧急风险\n    C = 2: 高：涉及资金损失或需要尽快处理的问题\nrefund_requested: 客户是否明确要求退还多扣的钱？\n    A = no: The answer is no (negative, or disagree with the claim).\n    B = yes: The answer is yes (affirmative, or align with the claim).'
		}, {
			'role': 'assistant',
			'content': '{\n    "team": "decision",\n    "urgency": "decision",\n    "refund_requested": "decision"\n}'
		}],
		'fields': ['team', 'urgency', 'refund_requested'],
		'options': {
			'team': [('A', 'billing', '账单、重复扣款和退款'), ('B', 'delivery', '订单延迟、配送和物流问题'), ('C', 'technical', '产品或技术故障')],
			'urgency': [('A', '0', '低：一般咨询或不紧急的问题'), ('B', '1', '中：需要跟进，但暂时没有明显紧急风险'), ('C', '2', '高：涉及资金损失或需要尽快处理的问题')],
			'refund_requested': [('A', 'no', 'The answer is no (negative, or disagree with the claim).'), ('B', 'yes', 'The answer is yes (affirmative, or align with the claim).')]
		},
		'targets': None,
		'ids': [248045, 248046, 198],
		'markers': {
			'team': 343,
			'urgency': 352,
			'refund_requested': 361
		}
	},
	'logprobs': {
		'team': {
			'A': -0.09375613182783127,
			'B': -2.7187561988830566,
			'C': -4.218756198883057
		},
		'urgency': {
			'A': -2.5136938095092773,
			'B': -1.1386936902999878,
			'C': -0.5136936902999878
		},
		'refund_requested': {
			'A': -0.8993051052093506,
			'B': -0.5243051052093506
		}
	}
}
```
'‘’‘

 
 3、继续处理--模型推理输出概率后的四种题型标签归一化

每个字段都有一个 `type`，决定选项怎么展开成符号：

| 题型 | 含义 | 选项展开方式 | 符号来源 |
|------|------|-------------|---------|
| `choice` | 从若干选项中选一个 | `criteria` 字典的 key 按**插入序** | A, B, C, … |
| `score`  | 打分（0~N 等级） | `criteria` 列表按**等级序** | A, B, C, … |
| `noul`   | 二值判断（yes/no） | 固定 `[no, yes]` | A=no, B=yes |

> 符号表是固定的 62 个单 token 符号：`A-Z` + `a-z` + `0-9`（`ANSWER_SYMBOLS`）。
> 因为服务端词表不可扩展，选符号必须用词表里已有的**单 token**，才能精确映射到 token id。

标签归一化（`normalize_label`）：`noul` 型的 `true/false/1/0` 统一归一成 `yes/no`；其他题型直接转字符串。这样「标签 → 选项值 → 下标 → 符号」是一条干净链路。

4、置信度添加

为什么加？因为模型可能会输出多个概率一样的选项，比如33%，33%，33%，这种情况置信度为0，因为模型不知道选哪个，概率都一样。需要通过算法输出置信度，后端做意图识别，才好判断是否准确

```
def _computer_believe(kind: str, probabilities: dict[str, float]) -> float:  
    """根据题型和候选概率计算置信度。"""  
    if not probabilities:  
        return 0.0  
  
    values = list(probabilities.values())  
    if kind == "noul":  
        return float(abs(2 * probabilities["yes"] - 1))  
  
    if kind == "score":  
        levels = [probabilities[level] for level in sorted(probabilities, key=int)]  
        count = len(levels)  
        if count == 1:  
            return 1.0  
        most_likely = max(range(count), key=levels.__getitem__)  
        spread = sum(  
            probability * abs(index - most_likely)  
            for index, probability in enumerate(levels)  
        )        uniform_spread = (  
            sum(abs(index - (count - 1) / 2) for index in range(count)) / count  
        )  
        return float(max(0.0, 1.0 - spread / uniform_spread))  
  
    count = len(values)  
    if count == 1:  
        return 1.0  
    return float(max(0.0, (max(values) - 1 / count) / (1 - 1 / count)))
```

##推理最终输出结果
```
{
  "team": {
    "type": "choice",
    "probabilities": {
      "billing": 0.9186083212294024,
      "delivery": 0.0665437591373926,
      "technical": 0.014847919633204955
    },
    "prediction": "billing",
    "believe": 0.8779124818441034
  },
  "urgency": {
    "type": "score",
    "probabilities": {
      "0": 0.08101013695167852,
      "1": 0.32040134517300634,
      "2": 0.5985885178753151
    },
    "expected_score": 1.5175783809236365,
    "prediction": "2",
    "believe": 0.27636757138545487
  },
  "refund_requested": {
    "type": "noul",
    "probabilities": {
      "no": 0.4073334000459302,
      "yes": 0.5926665999540697
    },
    "p_yes": 0.5926665999540697,
    "prediction": "yes",
    "believe": 0.18533319990813935
  }
}
```


4、温度校验
为什么要温度校验？因为模型输出90%，可能不是90%，有可能那是80%。存在夸大或者缩小情况，这时候就需要去校准这个值，不然后续判断决策的阈值就可能不准（假如模型不准，意图识别阈值是90%还是80%来判断选谁呢）
以下两个指标用于判断是否准确：
	a、ECE：Expected Calibration Error，期望校准误差  
	为什么评： 模型不只要给答案，也会给各候选答案的概率。ECE 检查这些概率能不能代表真实正确率：模型说“我有 80% 把握”的那些决策，是否夸大了或者缩小，真实没有80%  
	代码用最大候选概率作为该次决策的置信度：  
	接着把置信度分进 10 个区间：[0,0.1)、[0.1,0.2)，一直到 [0.9,1.0]。每个区间计算：   
	     该区间内的平均置信度  
	     该区间内的实际正确率  
	     再按该区间包含的决策占比加权，计算二者差距：  
	     例如某个区间里有 20 条决策，平均置信度 0.8，但实际只对了 14 条，正确率 0.7。该区间差距是 0.1，再乘以它在全部决策中的占比。  
	ECE 越低越好；0 表示这些分桶上的置信度和实际正确率一致，就是概率是准的  
	个人理解：ECE就是描述概率的准确性 ，越低 ，说明概率越准 ，模型输出的概率就没有夸大或者缩小  
  
	 b、Brier Score：布里尔分数，也称概率预测误差  
	为什么评：ECE 只看分桶后的“平均置信度与平均正确率”，可能掩盖单条预测的概率偏差。Brier Score 逐条、逐候选项检查概率与真实结果的距离。  
	对每次决策的每个候选标签，正确标签的真实值记为 1，其他标签记为 0：  
	 最后对所有决策取平均。  
	 比如标准答案有 A、模型给 P(A)=0.8, P(B)=0.2：onehot为（1，0）  
	 计算：（0.8-1）^2+（0.2-0）^2=0.08  
	 如果模型反而给 P(A)=0.4, P(B)=0.6：  
	 计算：（0.4-1）^2+（0.6-0）^2=0.52  
	 第二种误差更大，所以 Brier 越低越好。它同时惩罚答错和概率给得不合理；多分类时，代码把所有候选标签的平方误差相加。  
	   
	 个人理解：就是类似MSE均方误差，越小越好，说明和目标拟合
怎么校准？
	做法是温度缩放（temperature scaling）：在 120 案例 / 600 题的 calibration 划分上，对逆温度二分搜索使 NLL 最小。拟合得到 **T = 0.979**，NLL 从 0.48419 降到 0.48410，校准结果：
	![[0-expand-prj/Decision-Model-Jev/agentic-rl-lab/09-pytrio-jev/images/calibration-local.svg]]






5、评估

9.2

| Suite               | Decisions | T=1 Accuracy | T=0.979 Accuracy | T=1 ECE | T=0.979 ECE | T=1 Brier | T=0.979 Brier |
| ------------------- | --------: | -----------: | ---------------: | ------: | ----------: | --------: | ------------: |
| Jevbench Easy       |        48 |      100.00% |          100.00% |  0.0021 |      0.0020 |    0.0002 |        0.0001 |
| Jevbench Original   |        72 |       94.44% |           94.44% |  0.0334 |      0.0285 |    0.0739 |        0.0735 |
| Jevbench Hard       |       111 |       52.25% |           51.35% |  0.1741 |      0.1796 |    0.5856 |        0.5901 |
| Typed Decisions     |     2,000 |       77.55% |           77.40% |  0.0348 |      0.0357 |    0.3174 |        0.3180 |
| ToolACE             |       310 |       92.58% |           92.58% |  0.0386 |      0.0360 |    0.1233 |        0.1229 |
| **五项简单平均 Accuracy** |         — |   **83.37%** |       **83.16%** |       — |           — |         — |             — |

> 平均 Accuracy 是五个 suite 准确率的简单算术平均，不是按 `decisions` 加权后的总体准确率。温度校准是在独立 calibration split 上以最小化 NLL 为目标拟合，不能保证每个测试 suite 的 ECE、Brier 或 Accuracy 都改善。本次 T=0.979 在 Original、Easy、ToolACE 的 ECE/Brier 略有改善，但 Hard、Typed Decisions 的这两项指标略差；整体平均 Accuracy 比 T=1 低 0.21 个百分点。由于温度缩放理论上不改 argmax，这些 Accuracy 差异应视为运行间输出差异/数值扰动，不能归因于温度本身。

```text
本地对照（五项 suite 简单平均 Accuracy）
T=1.000   83.37%  ██████████████████████████████████████████████
T=0.979   83.16%  █████████████████████████████████████████████
```

总体指标定义

评测同时关注**决策是否选对**和**概率是否可信**。`eval.py` 对每条记录中的每个决策字段分别统计，因此 `decisions` 是字段级决策数，不一定等于 JSONL 记录数。

| 指标                                     | 衡量什么                      | 计算方法                                                                | 趋势   |
| -------------------------------------- | ------------------------- | ------------------------------------------------------------------- | ---- |
| Accuracy（准确率）                          | 预测标签是否与标准答案一致             | 正确决策数 ÷ 决策总数                                                        | 越高越好 |
| ECE（Expected Calibration Error，期望校准误差） | 置信度与实际正确率是否匹配             | 以 `max(P)` 为置信度，分为 10 个等宽区间；对每个区间计算「平均置信度与实际正确率之差的绝对值」，再按区间样本占比加权求和 | 越低越好 |
| Brier Score（布里尔分数）                     | 候选概率分布与真实 one-hot 标签的平方误差 | 每个决策对所有候选计算 `(p - y)^2` 并求和，再对决策取平均；正确候选 `y=1`，其他候选 `y=0`           | 越低越好 |


---


---

## 6. 快速上手命令

```bash
# 准备数据（仓库根目录）
uv run python 09-pytrio-jev/00-prepare-data.py

# 小规模测试
uv run python train.py --max-samples 128 --batch-size 16 --epochs 1 --save-every 0 --swanlab-mode disabled

# 正式训练
uv run python train.py --epochs 2 --batch-size 16 --save-every 100 --swanlab-mode online

# 换基座模型（experiment/weights 名字自动跟随）
uv run python train.py --base-model Qwen/Qwen3.5-9B --epochs 2

# 推理（不传 --weights 则评 base 模型）
uv run python 09-pytrio-jev/inference.py --weights trio://...
```

---

## 7. 关键设计决策汇总

| 决策                      | 原因                                    |
| ----------------------- | ------------------------------------- |
| 用符号 A/B/C 而非原始标签文本      | 结构化输出，候选集小且可控                         |
| `decision` 占位符          | 唯一单 token，精确标记答案位置，实现稀疏监督             |
| 只监督 marker 前一位置         | 训练信号集中在「决策」上，不被格式 token 稀释            |
| 推理用受限 softmax 而非生成      | 一次前向出全部字段概率，快且输出必合法                   |
| 训练无温度、无采样               | 交叉熵直接对 logits 算确定性 loss，目标确定，不需要从分布采样 |
| 温度仅在推理后处理出现             | 只做概率标定（锐度调整），不改 argmax、不做随机采样         |
| `markers[field] - 1` 定位 | 预测发生在占位符的前一位（读引号、预测符号）                |
| 双份存档（推理权重 + 训练状态）       | 兼顾采样与断点续训                             |
| 异步前向/反向 + 后台 loss 上报    | 训练流水线不断流，提高吞吐                         |
| dev 集与训练 loss 同构        | 早停判断准确，评估与训练目标一致                      |

---



---

## 10. 温度校准：黄金分割与牛顿法

> 先厘清温度的作用域：**训练不涉及温度**——masked SFT 用的是交叉熵，直接对 logits 算确定性 loss，没有采样动作，温度无处介入。温度只出现在**推理后处理**阶段，作用是概率标定，不是随机采样。

训练完成后，模型对各候选答案会给出 logprob。对候选分数除以温度 `T` 后做 softmax，可以调整概率分布的锐度：`T > 1` 会让概率更平，`T < 1` 会让概率更尖。温度只改变概率，不改变最高分候选（argmax）。

校准的目标是在独立 calibration 集上，找到一个共享温度 `T`，使所有字段实例的平均负对数似然（NLL）最低。一个实例就是一条记录中的一个字段，保存为 `(候选 logprob 向量, gold 下标)`。例如一条记录含 5 个待预测字段，就贡献 5 个实例。

### 10.1 数据只收集一次

`calibrate.py` 和 `calibrate_newton.py` 都先调用 `collect_instances()`：对 calibration 集做模型前向，收集每个字段的候选 logprob 和正确答案下标。之后优化温度时不会再次请求模型，只会在 CPU 上重复使用这些已收集的数值。

### 10.2 黄金分割法：试温度、比较平均 NLL

`calibrate.py` 在 `log(T) ∈ [log(0.01), log(100)]` 上用黄金分割搜索。每次给定一个 `T`，`mean_nll()` 都遍历全部实例，计算每个实例的 `-log(p_gold)`，再取平均，得到这个温度的一个总分。搜索过程不断比较候选温度的总分，并缩小区间，最后取区间中点对应的温度。

简单示意：

```text
T=0.5  → 全部字段的平均 NLL=0.94
T=1.0  → 全部字段的平均 NLL=0.72
T=2.0  → 全部字段的平均 NLL=0.68  ← 目前最好
T=4.0  → 全部字段的平均 NLL=0.75
```

以上数字仅用于说明：实际程序会在区间内逐步细化搜索，不是只比较这四个固定值。代码搜索的是 `x=log(T)`，因此在 T 空间按倍率来切分；结束后用 `exp(x)` 还原温度。黄金比例约为 `0.618`，使保留区间每轮缩到原来的约 61.8%。

当前实现固定迭代 80 轮，每轮重新计算两个候选点的平均 NLL，未缓存复用函数值。若有 `N` 个字段实例，拟合阶段约有 `80 × 2 × N` 次实例级 NLL 计算，另有拟合前后各一次整体 NLL 评估。这里的“计算”是对已缓存分数做数学运算，不是模型前向。

### 10.3 牛顿法：用梯度和曲率更新温度

`calibrate_newton.py` 优化 `w=1/T`，从 `w=1`（即 `T=1`）开始。每轮 `_grad_and_hessian()` 遍历所有实例，为每个实例计算当前概率分布下的梯度贡献和曲率贡献，再分别求平均得到整体梯度 `g` 和二阶导 `h`。随后用牛顿更新：

```text
w_new = w - g / h
```

所有字段共同更新的是同一个 `w`，不是每个字段各算一个温度再取平均。达到收敛条件后，返回 `T=1/w`；最多迭代 50 轮。若有 `N` 个字段实例、实际迭代 `K` 轮，迭代部分约遍历 `K × N` 次实例，此外还会在拟合前后各计算一次整体 NLL。

### 10.4 两种方法对比

| 方法 | 每轮做什么 | 计算量（N 个实例） | 特点 |
|------|------------|-------------------|------|
| 黄金分割 | 对候选温度计算平均 NLL，再缩小区间 | 当前实现约 `160 × N` 次实例级 NLL 计算 | 不需要导数，直观稳定；固定 80 轮 |
| 牛顿法 | 对全部实例计算梯度、曲率并更新 `w` | `K × N` 次实例级梯度/曲率计算 | 通常几轮收敛，遍历次数少；每实例计算更复杂，需关注数值收敛 |

例如 `N=600`：黄金分割拟合阶段约 `96,000` 次实例级 NLL 计算；牛顿法若 `K=5`，迭代阶段约 `3,000` 次实例级梯度/曲率计算。两者单次计算内容不同，因此这只是遍历规模对比，不代表实际运行时间必定相差 32 倍；应以实际耗时和校准集 NLL 验证。
