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
#
# decision 这个词在 Qwen3.5 词表里是单个 token（id 61781）​，而且正常语料里几乎不会在 state/instructions 里作为独立 token 出现。所以：
#
# encode 阶段：在骨架 token 里扫 decision 的命中位置，就能精确知道「每个字段的答案应该落在哪」→ 得到 markers。
# build_sft_datum 阶段：拿 markers[field] - 1 直接算出 weight=1 的位置，把那里的 target 从 61781 覆盖成 B/C 的正确符号 id。
# 整个过程 decision 只扮演「占位 + 定位」两个角色，最后被覆盖掉，从不作为监督目标出现（这点上一条已验证：模型在答案位置看到的是前面的 " 引号，不是 decision）。