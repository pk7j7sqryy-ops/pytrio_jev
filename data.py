"""判别决策格式编译：canonical record → chat messages → tokens/weights。

- 候选符号表为 62 个单 token 符号（A-Z a-z 0-9），choice 按插入序、score 按等级序、
  noul 固定 [no, yes]；
- assistant 内容是 JSON 骨架，值的位置放占位符 marker。不加 special token（TRIO 服务端
  词表不可扩展），用已有单 token "decision"（Qwen3.5 tokenizer id 61781）；
- labels 只落在 marker 位置（右移后 = marker 前一位置预测答案符号），其余全 0。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytrio as trio

SYSTEM_PROMPT = (
    "You are a careful decision assistant. Use the state and decision schema in "
    "the user message to make the requested decisions. For every field, choose "
    "exactly one answer symbol (e.g. A, B, C, ...) from its listed options and "
    "return one valid JSON object mapping each field name to its chosen symbol. "
    "Use the field names and symbols exactly as given. Do not include "
    "explanations, Markdown, or extra text."
)
ANSWER_SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
MARKER = "decision"
NOUL_YES = "The answer is yes (affirmative, or align with the claim)."
NOUL_NO = "The answer is no (negative, or disagree with the claim)."
TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "qwen3_5_jev.jinja"


def question_options(question: dict) -> list[tuple[str, str]]:
    """按题型展开为有序的 (value, description) 列表。"""
    kind = question["type"]
    criteria = question.get("criteria")
    if kind == "choice":
        return [(str(k), str(v)) for k, v in criteria.items()]
    if kind == "score":
        if isinstance(criteria, list):
            return [(str(i), str(v)) for i, v in enumerate(criteria)]
        return [(str(k), str(v)) for k, v in criteria.items()]
    if kind == "noul":
        descriptions = criteria or {}
        yes = next(
            (str(v) for k, v in descriptions.items() if str(k).lower() in {"yes", "true", "1"}),
            NOUL_YES,
        )
        no = next(
            (str(v) for k, v in descriptions.items() if str(k).lower() in {"no", "false", "0"}),
            NOUL_NO,
        )
        return [("no", no), ("yes", yes)]
    raise ValueError(f"unsupported question type: {kind!r}")


def normalize_label(question: dict, label: object) -> str:
    """noul 的 true/false/1/0 归一到 yes/no，其余题型转字符串。"""
    if question["type"] == "noul":
        return "yes" if str(label).lower() in {"yes", "true", "1"} else "no"
    return str(label)


def compile_record(record: dict, include_targets: bool = True) -> dict:
    """把一条 canonical record 编译成 messages 和符号映射。

    返回 fields / options（field → [(symbol, value, description)]）/ targets（field → 符号）。
    """
    questions = record["questions"]
    fields: list[str] = []
    options: dict[str, list[tuple[str, str, str]]] = {}
    targets: dict[str, str] = {}
    schema_lines: list[str] = []
    for field, question in questions.items():
        opts = question_options(question)
        field = str(field)
        fields.append(field)
        symbols = ANSWER_SYMBOLS[: len(opts)]
        options[field] = [
            (symbol, value, description)
            for symbol, (value, description) in zip(symbols, opts)
        ]
        schema_lines.append(f"{field}: {question['instructions']}")
        for symbol, value, description in options[field]:
            schema_lines.append(f"    {symbol} = {value}: {description}")
        if include_targets:
            label = normalize_label(question, record["targets"][field]["label"])
            values = [value for _, value, _ in options[field]]
            targets[field] = symbols[values.index(label)]

    state = json.dumps(record.get("state"), ensure_ascii=False, indent=2)
    user_text = (
        "Return one answer for every field using the supplied answer symbols.\n\n"
        f"## State\n{state}\n"
        "## Decision schema\n" + "\n".join(schema_lines)
    )
    skeleton = json.dumps(dict.fromkeys(fields, MARKER), ensure_ascii=False, indent=4)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": skeleton},
    ]
    return {
        "messages": messages,
        "fields": fields,
        "options": options,
        "targets": targets if include_targets else None,
    }


class Encoder:
    """绑定 tokenizer 与钉死的 chat template，负责 tokenization 和 marker 定位。"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.template = TEMPLATE_PATH.read_text()
        marker_ids = tokenizer.encode(MARKER, add_special_tokens=False)
        assert len(marker_ids) == 1, "marker 必须是单 token"
        self.marker_id = marker_ids[0]

    def encode(self, record: dict, include_targets: bool = True) -> dict:
        """渲染完整训练/推理文本，返回 token ids 与各字段的 marker 索引。

        marker 索引靠"prefix 之外、骨架之内"定位，不扫全文，因此 instructions 或
        state 里出现单词 decision 也不会误判。
        """
        compiled = compile_record(record, include_targets)
        messages = compiled["messages"]
        ids = self.tokenizer.encode(
            self.tokenizer.apply_chat_template(
                messages, chat_template=self.template, tokenize=False
            ),
            add_special_tokens=False,
        )
        prefix_ids = self.tokenizer.encode(
            self.tokenizer.apply_chat_template(
                messages[:-1],
                chat_template=self.template,
                tokenize=False,
                add_generation_prompt=True,
            ),
            add_special_tokens=False,
        )
        assert ids[: len(prefix_ids)] == prefix_ids, "prefix/skeleton 边界 tokenization 不稳定"
        skeleton_ids = ids[len(prefix_ids) :]
        # 骨架里 marker token 的命中包含字段名中的出现（如 ToolACE 的字段就叫 "decision"，
        # 骨架为 {"decision": "decision"}），按字段顺序先跳过 key 里的命中，再取 value。
        hits = [
            i for i, token_id in enumerate(skeleton_ids) if token_id == self.marker_id
        ]
        markers: dict[str, int] = {}
        cursor = 0
        for field in compiled["fields"]:
            cursor += self.tokenizer.encode(field, add_special_tokens=False).count(
                self.marker_id
            )
            markers[field] = len(prefix_ids) + hits[cursor]
            cursor += 1
        assert cursor == len(hits), "骨架中 marker 数与字段数不一致"
        return {
            **compiled,
            "ids": ids,
            "markers": markers,
        }

    def symbol_token_id(self, symbol: str) -> int:
        return self.tokenizer.encode(symbol, add_special_tokens=False)[0]

    def build_sft_datum(self, record: dict) -> trio.Datum:
        """构造 masked SFT 的 Datum：只在 marker 前一位置监督答案符号。"""
        encoded = self.encode(record)
        ids = encoded["ids"]
        target_tokens = list(ids[1:])
        weights = [0.0] * len(target_tokens)
        for field, symbol in encoded["targets"].items():
            position = encoded["markers"][field] - 1
            target_tokens[position] = self.symbol_token_id(symbol)
            weights[position] = 1.0
        return trio.Datum(
            model_input=trio.ModelInput.from_ints(ids[:-1]),
            loss_fn_inputs={
                "target_tokens": np.asarray(target_tokens, dtype=np.int64),
                "weights": np.asarray(weights, dtype=np.float32),
            },
        )


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]



#本文件数据处理流程
# train.py的train里轮次的每一批数据的每行数据--调用-》data.py的build_sft_datum方法处理数据
#数据如下
# {
# 	"id": "tr_security_incidents_000178",
# 	"state": {
# 		"alert": {
# 			"description": "an unusual volume of files read in a short window",
# 			"evidence": "Between 2026-09-15 01:00:12 UTC and 01:04:58 UTC, the unprivileged CI runner service accessed 4,237 files in the /var/build/artifacts directory, reading a total of 12.3 GB of data. The source IP 198.51.100.77, which is not on the approved list, initiated the activity while MFA was enforced and the runner's credentials had been rotated on 2026-09-12. A scheduled change window was open during this period.",
# 			"rule": "mass_download"
# 		},
# 		"context": {
# 			"asset_criticality": "medium",
# 			"change_window_active": true,
# 			"source_on_allowlist": false
# 		},
# 		"history": {
# 			"credential_rotation_days_ago": 3,
# 			"distinct_countries_30d": 4,
# 			"logins_30d": 688,
# 			"prior_alerts_90d": 0
# 		},
# 		"principal": {
# 			"mfa_enrolled": true,
# 			"name": "svc-analytics-473",
# 			"privileges": ["repo:read"],
# 			"type": "ci_runner"
# 		}
# 	},
# 	"questions": {
# 		"credential_compromise": {
# 			"instructions": "The evidence indicates a credential or account has been compromised.",
# 			"type": "noul"
# 		},
# 		"disposition": {
# 			"criteria": {
# 				"close_benign": "Expected, explainable activity; close without analyst time.",
# 				"contain": "Contain the host or account immediately; do not wait for triage.",
# 				"investigate": "Warrants an analyst opening an investigation.",
# 				"monitor": "Not clearly malicious, but worth watching for recurrence."
# 			},
# 			"instructions": "How should this security alert be dispositioned?",
# 			"type": "choice"
# 		},
# 		"severity": {
# 			"criteria": ["Negligible: no access to anything sensitive.", "Low: limited access, easily reversed.", "Moderate: access to internal systems or non-public data.", "High: access to production, secrets or customer data.", "Critical: active compromise of crown-jewel systems."],
# 			"instructions": "How severe is the potential impact if this alert is real?",
# 			"type": "score"
# 		},
# 		"true_positive": {
# 			"criteria": {
# 				"false": "Benign activity, a misconfiguration, or a known false positive.",
# 				"true": "The underlying behaviour is malicious or unauthorised."
# 			},
# 			"instructions": "This alert reflects genuinely malicious or unauthorised activity.",
# 			"type": "noul"
# 		},
# 		"urgency": {
# 			"criteria": ["No time pressure; can wait indefinitely.", "Routine; handle within the normal queue.", "Elevated; should be handled within the same week.", "Critical; requires action within the same day."],
# 			"instructions": "How quickly must a responder act on this alert?",
# 			"type": "score"
# 		}
# 	},
# 	"targets": {
# 		"credential_compromise": {
# 			"label": "true"
# 		},
# 		"disposition": {
# 			"label": "investigate"
# 		},
# 		"severity": {
# 			"label": "2"
# 		},
# 		"true_positive": {
# 			"label": "true"
# 		},
# 		"urgency": {
# 			"label": "2"
# 		}
# 	}
# }

#1、先经过encoded = self.encode(record) 处理数据，得到的结果为
# encoded = {
#     "messages": [system+user（state原文+schema（A/B/C/D：选项内容）+assistant（只有占位符）],   # ① 来自 compile_record

            # schema内容：
                        # credential_compromise: The evidence indicates a credential or account has been compromised.
                        # A = no: ...
                        # B = yes: ...
                        # disposition: How should this security alert be dispositioned?
                        # A = close_benign: ...
                        # ...
                        # urgency: How quickly must a responder act on this alert?
                        # A = 0: ... D = 3: ...
                #   assistant
                # {
                #     "credential_compromise": "decision",
                #     "disposition": "decision",
                #     "severity": "decision",
                #     "true_positive": "decision",
                #     "urgency": "decision"
                # }

#     "fields":   ["credential_compromise", "disposition", "severity", "true_positive", "urgency"],   # ② 来自 compile_record

#     "options":  {credential_compromise:
                    #     A = no:  The answer is no (negative, or disagree with the claim).
                    #     B = yes: The answer is yes (affirmative, or align with the claim).
                    # disposition:。。。。。},   # ③ 来自 compile_record，真实 dump 出来的   A/B/C/D：选项内容

#     "targets":  {
                    #   "credential_compromise": "B",   # "true" → 归一 "yes" → 下标1 → B，假如是false 则是下标0，A
                    #   "disposition": "C",             # "investigate" → 下标2 → C
                    #   "severity": "C",                # "2" → 下标2 → C
                    #   "true_positive": "B",           # "true" → "yes" → B
                    #   "urgency": "C"                  # "2" → 下标2 → C
                    # },   # ④ 来自 compile_record

#     "ids":      [
                #   # 前 30 个（system 指令开头）：
                #   248045, 8678, 198, 2523, 513, 264, 16097, 5307, 17313, 13,
                #   5272, 279, 1528, 321, 5307, 10485, 303, 279, 1156, 1876,
                #   310, 1236, 279, 10897, 10856, 13, 1690, 1396, 2002, 11,
                #   # …… 中间是 state 案情 + schema 选项 ……
                #   # 后 30 个（骨架尾部）：
                #   198, 262, 328, 74385, 763, 328, 61781, 487, 198, 262,
                #   328, 1802, 52339, 763, 328, 61781, 487, 198, 262, 328,
                #   5382, 2179, 763, 328, 61781, 1, 198, 92, 248046, 198,
                # ],   # ⑤ 来自 encode
                #还不是最终输入模型的，248045 / 248046 = <|im_start|> / <|im_end|>（chat template 的控制符）
                                   #这个需要替换 ---》 61781 = decision（占位符 marker）

#     "markers":  {
# #         "credential_compromise": 836,
# #         "disposition": 845,
# #         "severity": 853,
# #         "true_positive": 862,
# #         "urgency": 871,
# #     },   # ⑥ 来自 encode
# }







# encoded = self.encode(record)
# encoded: dict
# ├── messages: list[dict]   对话文本
# ├── fields: list[str]      问题字段名
# ├── options: dict          每个字段的选项及符号
# ├── targets: dict          正确答案符号
# ├── ids: list[int]         对话 token 序列
# └── markers: dict[str,int] 每个答案占位符的位置

# 举例encoded = {
#     "ids": [ ... 877 个 token id ... ],
#     "markers": {
#         "credential_compromise": 836,
#         "disposition": 845,
#         "severity": 853,
#         "true_positive": 862,
#         "urgency": 871,
#     },
#     "targets": {
#         "credential_compromise": "B",
#         "disposition": "C",
#         "severity": "C",
#         "true_positive": "B",
#         "urgency": "C",
#     },
#     "fields": ["credential_compromise", "disposition", "severity", "true_positive", "urgency"],
#     "options": { ... },
# }


# decision 占位符是用来「标记位置」的，不是用来「当答案」的。
#模型最终要ids[:-1]和ids[1:]以及weight，ids[:-1]和ids[1:]都好确定，但是weight不好确定，
# 所以需要答案的具体位置，和代替符号A，B，C（实际训练后也是输出这些为结果），所以markers作用
# 作用1：是替换为abc，不在是true、选项，分数。
# 作用2：是告诉weight哪个位置的权重为1，好只计算这些loss



# 具体拆成两个原因：
#
# 1. 答案符号不能直接写进 assistant 骨架
#
# 如果一开始就在 assistant 里写 {"disposition": "C", ...}，那 ids 里 C 的位置已经是正确答案了。但这样的话：
#
# 训练时模型会在输入里直接看到答案，等于泄题；
# 而且你没法区分「这里是要监督的位置」和「别的地方恰好也出现字母 C」——state 描述里到处是单词和字符，C、B 这种单字符 token 在正文里满天飞。
# 2. decision 是唯一的「定位锚点」
# decision 这个词在 Qwen3.5 词表里是单个 token（id 61781）​，而且正常语料里几乎不会在 state/instructions 里作为独立 token 出现。所以：
# encode 阶段：在骨架 token 里扫 decision 的命中位置，就能精确知道「每个字段的答案应该落在哪」→ 得到 markers。
# build_sft_datum 阶段：拿 markers[field] - 1 直接算出 weight=1 的位置，把那里的 target 从 61781 覆盖成 B/C 的正确符号 id。
# 整个过程 decision 只扮演「占位 + 定位」两个角色，最后被覆盖掉，从不作为监督目标出现（这点上一条已验证：模型在答案位置看到的是前面的 " 引号，不是 decision）。


#实际训练输入-转tokenid之前：
# <|im_start|>system
# You are a careful decision assistant. Use the state and decision schema in the user message to make the requested decisions. For every field, choose exactly one answer symbol (e.g. A, B, C, ...) from its listed options and return one valid JSON object mapping each field name to its chosen symbol. Use the field names and symbols exactly as given. Do not include explanations, Markdown, or extra text.
# <|im_end|>
# <|im_start|>user
# Return one answer for every field using the supplied answer symbols.
#
# ## State
# {
#   "alert": { "description": "...", "evidence": "...", "rule": "mass_download" },
#   "context": { "asset_criticality": "medium", "change_window_active": true, "source_on_allowlist": false },
#   "history": { "credential_rotation_days_ago": 3, "distinct_countries_30d": 4, "logins_30d": 688, "prior_alerts_90d": 0 },
#   "principal": { "mfa_enrolled": true, "name": "svc-analytics-473", "privileges": ["repo:read"], "type": "ci_runner" }
# }
# ## Decision schema
# credential_compromise: The evidence indicates a credential or account has been compromised.
#     A = no: ...
#     B = yes: ...
# disposition: How should this security alert be dispositioned?
#     A = close_benign: ...
#     B = contain: ...
#     C = investigate: ...
#     D = monitor: ...
# severity: How severe is the potential impact if this alert is real?
#     A = 0: ...  B = 1: ...  C = 2: ...  D = 3: ...  E = 4: ...
# true_positive: This alert reflects genuinely malicious or unauthorised activity.
#     A = no: ...
#     B = yes: ...
# urgency: How quickly must a responder act on this alert?
#     A = 0: ...  B = 1: ...  C = 2: ...  D = 3: ...
# <|im_end|>
# <|im_start|>assistant
# <think>
#
# </think>
#
# {
#     "credential_compromise": "decision",
#     "disposition": "decision",
#     "severity": "decision",
#     "true_positive": "decision",
#     "urgency": "decision"
# }


#对比的target
# <|im_start|>system
# You are a careful decision assistant. ...
# <|im_end|>
# <|im_start|>user
# ... state + schema ...
# <|im_end|>
# <|im_start|>assistant
# <think>
#
# </think>
#
# {
#     "credential_compromise": "B",      ← decision → B
#     "disposition": "C",                ← decision → C
#     "severity": "C",                   ← decision → C
#     "true_positive": "B",              ← decision → B
#     "urgency": "C"                     ← decision → C
# }
# <|im_end|>

# 位置:     ...  835         836           837         838   ...
# 模型读:   ...  引号"      decision      引号"        逗号,   ...
# 模型预测:      B          → 引号"      → 逗号,       → ...
# weight:        1.0 ✅      0.0 ❌        0.0 ❌       0.0 ❌

#为什么中间引号闭合不算loss
#见过，但不专门算这部分」 —— 精确地讲：
# 1、因为只有一次前向传播 不需要闭合 只用预测一个值
