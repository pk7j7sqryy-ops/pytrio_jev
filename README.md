# jev — 网络安全决策模型

> 微调一个大语言模型，让它面对一条安全告警（或一个多字段判断任务）时，**一次性给出所有字段的决策符号**（A/B/C/…）。
>
> 核心思想：把「多字段结构化决策」压缩成「只监督答案符号」的**稀疏监督 masked SFT**，推理时用**受限 softmax** 一次性读出每个字段的概率分布，而不是逐字生成 JSON。

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

---

## 0. 一句话理解

| 阶段 | 做了什么 | 关键手段 |
|------|---------|---------|
| 数据 | 把「告警状态 + 多个判断问题 + 正确答案」编成一段对话 | canonical record → chat messages |
| 训练 | 只在**答案符号的位置**算 loss，其余位置权重为 0 | `decision` 占位符 + markers 定位 |
| 推理 | 一次前向，在每个 marker 位置读 top-k logprob，过滤候选后 softmax 取值 | 受限 softmax，不生成文本 |

---

## 1. 项目结构

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

---

## 2. 数据怎么来（准备数据）

`00-prepare-data.py` 从 HuggingFace 拉两个数据集，统一转成 **canonical record** 格式：

| 来源 | 用途 | 切分 |
|------|------|------|
| `LocalLLaMA/typed-decisions` | 主训练数据（多字段判别决策） | 每个 workflow 240 train / 30 dev / 30 calibration |
| `Team-ACE/ToolACE` | 工具调用决策（单工具选择） | 采样 1200 条，避开 eval 集 |

切分是**案例级确定性切分**（用 `hashlib` 对 id 做稳定哈希），保证复现。ToolACE 只保留「标准格式 + 单工具调用」的样本，并把「该调用哪个工具」也转成一个 `choice` 字段。

**安全告警数据**（`cyber-security-data.py`）：从 `Vyber07/cyber-security` 流式取前 200 条到本地，是 README 示例里那种 `alert/evidence/rule` 形态的来源。

> 运行方式：仓库根目录 `uv run python 09-pytrio-jev/00-prepare-data.py`

---

## 3. 训练流程

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

---

## 4. 数据处理流程（核心）

> 这是整个项目最精妙的部分，也是你笔记的重点。数据从「一条结构化记录」到「模型可训练的一个样本」，走三步：
> `compile_record` → `encode` → `build_sft_datum`。

### 4.1 数据长什么样（canonical record）

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

### 4.2 四种题型

每个字段都有一个 `type`，决定选项怎么展开成符号：

| 题型 | 含义 | 选项展开方式 | 符号来源 |
|------|------|-------------|---------|
| `choice` | 从若干选项中选一个 | `criteria` 字典的 key 按**插入序** | A, B, C, … |
| `score`  | 打分（0~N 等级） | `criteria` 列表按**等级序** | A, B, C, … |
| `noul`   | 二值判断（yes/no） | 固定 `[no, yes]` | A=no, B=yes |

> 符号表是固定的 62 个单 token 符号：`A-Z` + `a-z` + `0-9`（`ANSWER_SYMBOLS`）。
> 因为服务端词表不可扩展，选符号必须用词表里已有的**单 token**，才能精确映射到 token id。

标签归一化（`normalize_label`）：`noul` 型的 `true/false/1/0` 统一归一成 `yes/no`；其他题型直接转字符串。这样「标签 → 选项值 → 下标 → 符号」是一条干净链路。

### 4.3 `encode()`：文本 → 模板 → tokens → markers

`Encoder.encode(record)` 返回一个 dict，包含六样东西：

```
encoded: dict
├── messages : list[dict]    对话文本（system + user + assistant 骨架）
├── fields   : list[str]     字段名列表
├── options  : dict          每个字段的选项（symbol, value, description）
├── targets  : dict          正确答案符号（如 {"disposition": "C"}）
├── ids      : list[int]     整段对话的 token 序列
└── markers  : dict[str,int] 每个字段答案占位符在 ids 中的位置
```

**messages 的结构**（`compile_record` 产出）：

- **system**：固定系统提示词（要求只输出一个 JSON、每个字段选一个符号、不输出多余文字）。
- **user**：`## State`（state 的 JSON 原文）+ `## Decision schema`（每个字段的题目说明 + `A = xxx: 解释` 选项列表）。
- **assistant**：一个 JSON **骨架**，每个字段的值都放占位符 `"decision"`：

```json
{
    "credential_compromise": "decision",
    "disposition": "decision",
    "severity": "decision",
    "true_positive": "decision",
    "urgency": "decision"
}
```

**ids 的生成**：把 messages 套进钉死的 `qwen3_5_jev.jinja` 模板（`apply_chat_template`），再 tokenize。模板会展开成 `<|im_start|>system ... <|im_end|>` 这种 Qwen 控制符格式。

**markers 的定位**（这是关键技巧）：不是全文扫描 `decision` 这个词，而是——

1. 先算出「prefix」（system+user 部分）的 token 序列长度；
2. 只取**骨架部分**（assistant 的 JSON）的 token 序列；
3. 在骨架里找 `decision`（marker id）的命中位置；
4. 按字段顺序，跳过字段名 key 里可能出现的 `decision`，逐个取 value 位置。

这样即使 `state` 或 `instructions` 正文里恰好出现 "decision" 这个词，也不会误判——因为只在骨架里扫。

### 4.4 `decision` 占位符：为什么要它

这是你笔记里最核心的一点，值得单独讲透。

**`decision` 占位符是用来「标记位置」的，不是用来「当答案」的。**

训练时模型最终需要三样东西：`ids[:-1]`（输入）、`ids[1:]`（预测目标）、`weights`（哪些位置要监督）。前两个都好确定，唯独 **weights 不好定**——因为我们要「只监督答案符号」，得知道答案符号到底在 token 流的哪个位置。

`decision` 占位符就是干这个的，它有两个作用：

1. **定位（markers）**：`decision` 在 Qwen3.5 词表里是单 token（id 61781），且正文里几乎不会作为独立 token 出现。在骨架里扫它的命中位置，就能精确知道「每个字段的答案该落在哪」→ 得到 markers。
2. **替换成符号**：训练后模型实际输出的是 A/B/C 这些符号，不是 `true`/选项文本/分数。markers 让「把答案位置从 `decision` 覆盖成正确符号 id」这件事变成一行代码。

**为什么不能一开始就在 assistant 骨架里写正确答案符号？** 两个原因：

- **泄题**：如果直接写 `{"disposition": "C", ...}`，模型在输入里就看到了答案，等于作弊。
- **无法区分位置**：state 描述里到处是单词和字符，`C`、`B` 这种单字符 token 在正文里满天飞，你分不清「这里是监督位置」还是「恰好出现了一个字母 C」。

**为什么不用「全错位监督」替代占位符？** 全错位监督（每个 token 都算 loss）虽然省事，但会把训练信号浪费在「学格式」上，决策能力被稀释。占位符让「只监督 5 个符号」从「很难」变成「一行代码」。**占位符不是训练的必要条件，而是「稀疏监督」的工程实现手段。**

### 4.5 `build_sft_datum()`：拼出训练样本

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

**为什么是 `marker - 1`？** 因为 `markers[field]` 指向的是 `decision` 这个占位 token 的位置，而「预测答案符号」这件事发生在它**前一个位置**——模型在这个位置读的是前面的引号 `"`，要预测的下一个 token 才是答案符号（详见 4.6 的图示）。

最终得到 `Datum`：
- `model_input` = `ids[:-1]`（去掉最后一个 token）
- `target_tokens` = `ids[1:]`（右移一位），其中答案位置被覆盖成正确符号 id
- `weights` = 只在答案位置为 1，其余全 0

### 4.6 为什么中间引号闭合不算 loss

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

---

## 5. 推理流程（核心）

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
- **置信度 `believe`**：按题型用不同公式从概率分布算出一个 0~1 的置信度（noul 用 `|2·p_yes - 1|`，score 用与均匀分布的偏离度，choice 用最大概率偏离均匀的程度）。

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

| 决策 | 原因 |
|------|------|
| 用符号 A/B/C 而非原始标签文本 | 结构化输出，候选集小且可控 |
| `decision` 占位符 | 唯一单 token，精确标记答案位置，实现稀疏监督 |
| 只监督 marker 前一位置 | 训练信号集中在「决策」上，不被格式 token 稀释 |
| 推理用受限 softmax 而非生成 | 一次前向出全部字段概率，快且输出必合法 |
| `markers[field] - 1` 定位 | 预测发生在占位符的前一位（读引号、预测符号） |
| 双份存档（推理权重 + 训练状态） | 兼顾采样与断点续训 |
| 异步前向/反向 + 后台 loss 上报 | 训练流水线不断流，提高吞吐 |
| dev 集与训练 loss 同构 | 早停判断准确，评估与训练目标一致 |
