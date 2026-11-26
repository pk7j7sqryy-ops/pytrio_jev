"""受限 softmax 推理：一次前向读出全部字段的候选分布。

不生成文本——把含 marker 骨架的完整 prompt 发给 sample()，靠 topk_prompt_logprobs
在每个 marker 位置读"下一个 token"的 top-k 分布，过滤到该字段候选符号后归一化。

用法：
uv run python 09-pytrio-jev/inference.py --weights trio://...   # 不传则评 base
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pytrio as trio
from dotenv import load_dotenv

from data import Encoder, load_jsonl

SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / ".env")
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"


def _field_logprobs(encoder: Encoder, encoded: dict, response) -> dict:
    """从一次前向的 topk_prompt_logprobs 提取各字段候选符号的 logprob。

    topk_prompt_logprobs[m] 是"给定前 m 个 token，第 m 个 token 的分布"，
    marker 在第 m 位，所以该位置就是训练时监督的预测点。
    候选未进 top-k 时按 floor = min(topk) - 10 近似（约等于概率 0）。
    """
    assert response.input_tokens == len(encoded["ids"]), "本地/远端 prompt 长度不一致"
    fields: dict[str, dict[str, float]] = {}
    for field, position in encoded[
        "markers"
    ].items():  ##markers= {'team': 303, 'urgency': 312, 'refund_requested': 321}
        # 找到对应marker位置token的topk概率凑成字典{位置：概率}{32: -0.0018220028141513467, 33: -6.626821994781494, 34: -7.626821994781494}
        topk = dict(
            response.topk_prompt_logprobs[position]
        )  # [None, [(846, -0.22570660710334778), (8678, -1.6007065773010254), (1785, -9.038207054138184)], [(198, -0.01626366749405861), (2
        # 找最小概率然后-10
        floor = min(topk.values()) - 10.0  # top-k 里最小那个概率再减 10
        # 取对应每个选项的值
        fields[field] = {
            symbol: topk.get(encoder.symbol_token_id(symbol), floor)
            for symbol, _, _ in encoded["options"][field]  # 取对应的选择A，B，C，D
        }
    #         print('encoded["options"][field]=',encoded["options"][field])#[('A', 'billing', 'Payments and refunds'), ('B', 'delivery', 'Shipping and delivery'), ('C', 'technical', 'Product technical support')]
    # fields= {'team': {'A': -0.0018878034316003323, 'B': -6.626887798309326, 'C': -7.501887798309326}}
    #         print('fields=',fields)
    # print('fields=',fields)#{'team': {'A': -0.0018878034316003323, 'B': -6.626887798309326, 'C': -7.501887798309326}, 'urgency': {'A': -6.756525039672852, 'B': -5.256525039672852, 'C': -0.0065251621417701244}, 'refund_requested': {'A': -5.504185199737549, 'B': -0.0041853697039186954}}

    return {"encoded": encoded, "logprobs": fields}


def candidate_logprobs(
    sampling_client: trio.SamplingClient,
    encoder: Encoder,
    record: dict,
) -> dict:
    """一次前向，返回 {field: {symbol: logprob}}（未归一、未加温度）。"""
    encoded = encoder.encode(record, include_targets=False)
    # print(encoded["messages"])
    # print(encoded["ids"])
    # tokenizer = sampling_client.get_tokenizer()
    # ids = tokenizer.encode(
    #     tokenizer.apply_chat_template(
    #         encoded["messages"], chat_template=encoder.template, tokenize=False
    #     ),
    #     add_special_tokens=False,
    # )
    # print(ids)
    # 用加上占位符号的tokenid，作为推理输入
    response = sampling_client.sample(
        prompt=trio.ModelInput.from_ints(encoded["ids"]),
        num_samples=1,
        sampling_params=trio.SamplingParams(max_tokens=1),
        include_prompt_logprobs=True,  # 每个
        topk_prompt_logprobs=max(
            len(opts) for opts in encoded["options"].values()
        ),  # 取最多的概率分布，比如问题A：三个选项，问题B：2个选项。那占位符号输出概率就是三个后选项
    ).result()
    # print(response)
    # 描述eos结束符号信息：response.sequences= [SampleSequence(stop_reason='stop', tokens=[248044], text='', logprobs=[-0.0001456631434848532])]
    #
    # print('response.sequences=',response.sequences)
    # print('markers=',encoded["markers"])#markers= {'team': 303, 'urgency': 312, 'refund_requested': 321}
    # print('prompt_logprobs=', response.prompt_logprobs)  # 327
    # print('prompt_logprobs=',len(response.prompt_logprobs))#327
    # print('topk_prompt_logprobs=',response.topk_prompt_logprobs)#327
    # print('topk_prompt_logprobs=',len(response.topk_prompt_logprobs))#327
    # print('encoded["ids"]=',len(encoded["ids"]))#327
    # print('response.topk_prompt_logprobs[303]=',dict(response.topk_prompt_logprobs[303]))
    # print('output_tokens=', response.output_tokens)  # 1
    for i, seq in enumerate(response.sequences):
        print(f"Sample {i + 1}: {repr(seq)}")
    return _field_logprobs(encoder, encoded, response)


async def candidate_logprobs_async(
    sampling_client: trio.SamplingClient,
    encoder: Encoder,
    record: dict,
) -> dict:
    """candidate_logprobs 的异步版，供并发评测/校准使用。"""
    encoded = encoder.encode(record, include_targets=False)
    response = await sampling_client.sample_async(
        prompt=trio.ModelInput.from_ints(encoded["ids"]),
        num_samples=2,
        sampling_params=trio.SamplingParams(max_tokens=1),
        include_prompt_logprobs=True,
        topk_prompt_logprobs=max(len(opts) for opts in encoded["options"].values()),
    )
    return _field_logprobs(encoder, encoded, response)


def softmax(logprobs: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logprobs / temperature
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def _computer_believe(kind: str, probabilities: dict[str, float]) -> float:
    """
    Jev（TypeSafe 原版）执信度计算
    kind：具体type
    probabilities：{'billing': 0.9186083251288092, 'delivery': 0.06654375594933727, 'technical': 0.014847918921853654}
    """
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
        )
        uniform_spread = (
            sum(abs(index - (count - 1) / 2) for index in range(count)) / count
        )
        return float(max(0.0, 1.0 - spread / uniform_spread))

    count = len(values)
    if count == 1:
        return 1.0
    return float(max(0.0, (max(values) - 1 / count) / (1 - 1 / count)))


def _build_decisions(
    record: dict, encoded: dict, logprobs: dict, temperature: float
) -> dict[str, dict]:
    """候选 logprob → 各字段的分布与预测。
    #选值
    choice → {value: prob} + argmax；noul → p_yes；score → 概率加权的期望分。
    """
    out: dict[str, dict] = {}
    for field, options in encoded["options"].items():
        symbols = [symbol for symbol, _, _ in options]
        probs = softmax(np.array([logprobs[field][s] for s in symbols]), temperature)
        by_value = {value: float(p) for (_, value, _), p in zip(options, probs)}
        # print('by_value=',by_value)#{'billing': 0.9186083251288092, 'delivery': 0.06654375594933727, 'technical': 0.014847918921853654}
        question = record["questions"][field]
        kind = question["type"]
        result: dict = {"type": kind, "probabilities": by_value}
        best = max(by_value, key=by_value.get)
        if kind == "noul":
            result["p_yes"] = by_value["yes"]
            result["prediction"] = "yes" if by_value["yes"] >= by_value["no"] else "no"
        elif kind == "score":
            result["expected_score"] = float(
                sum(int(value) * p for value, p in by_value.items())
            )
            result["prediction"] = best
        else:
            result["prediction"] = best
        result["believe"] = _computer_believe(kind, by_value)
        out[field] = result
    return out


def decide(
    sampling_client: trio.SamplingClient,
    encoder: Encoder,
    record: dict,
    temperature: float = 1.0,
) -> dict[str, dict]:
    """对一条记录输出各字段的分布与预测（同步版）。"""
    collected = candidate_logprobs(sampling_client, encoder, record)
    return _build_decisions(
        record, collected["encoded"], collected["logprobs"], temperature
    )


async def decide_async(
    sampling_client: trio.SamplingClient,
    encoder: Encoder,
    record: dict,
    temperature: float = 1.0,
) -> dict[str, dict]:
    """对一条记录输出各字段的分布与预测（异步版）。"""
    collected = await candidate_logprobs_async(sampling_client, encoder, record)
    return _build_decisions(
        record, collected["encoded"], collected["logprobs"], temperature
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="候选符号受限 softmax 推理")
    parser.add_argument(
        "--base-model", default=os.getenv("JEV_BASE_MODEL", DEFAULT_MODEL)
    )
    parser.add_argument(
        "--weights",
        default=os.getenv("JEV_INFERENCE_WEIGHTS") or None,
        help="trio:// LoRA 权重路径，不传则评 base",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("JEV_INFERENCE_TEMPERATURE", "1.0")),
    )
    input_path = os.getenv("JEV_INFERENCE_INPUT")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(input_path) if input_path else None,
        help="canonical JSONL，默认用内置样例",
    )
    parser.add_argument(
        "--limit", type=int, default=int(os.getenv("JEV_INFERENCE_LIMIT", "3"))
    )
    args = parser.parse_args()

    service_client = trio.ServiceClient()
    sampling_client = service_client.create_sampling_client(
        base_model=args.base_model, model_path=args.weights
    )
    encoder = Encoder(sampling_client.get_tokenizer())

    if args.input:
        records = load_jsonl(args.input)[: args.limit]
    else:
        records = [
            {
                "id": "demo-sarcasm-2",
                "state": {
                    "customer_message": (
                        "真是太感谢你们了，订单晚到了三天，还顺便多扣了一笔钱，服务真周到。"
                        "我倒也不是非要你们马上处理，毕竟等账单越滚越大也挺有意思的。"
                    )
                },
                "questions": {
                    "team": {
                        "type": "choice",
                        "instructions": "应该由哪个团队处理这条投诉？",
                        "criteria": {
                            "billing": "账单、重复扣款和退款",
                            "delivery": "订单延迟、配送和物流问题",
                            "technical": "产品或技术故障",
                        },
                    },
                    "urgency": {
                        "type": "score",
                        "instructions": "这条投诉的处理优先级是什么？",
                        "criteria": [
                            "低：一般咨询或不紧急的问题",
                            "中：需要跟进，但暂时没有明显紧急风险",
                            "高：涉及资金损失或需要尽快处理的问题",
                        ],
                    },
                    "refund_requested": {
                        "type": "noul",
                        "instructions": "客户是否明确要求退还多扣的钱？",
                    },
                },
            },
            {
                "id": "demo-1",
                "state": {
                    "customer_message": "I bought headphones yesterday and noticed "
                    "today that I was charged twice. Please refund the extra charge "
                    "as soon as possible."
                },
                "questions": {
                    "team": {
                        "type": "choice",
                        "instructions": "Which team should handle this ticket?",
                        "criteria": {
                            "billing": "Payments and refunds",
                            "delivery": "Shipping and delivery",
                            "technical": "Product technical support",
                        },
                    },
                    "urgency": {
                        "type": "score",
                        "instructions": "Choose the priority for handling this ticket.",
                        "criteria": [
                            "Low: general inquiry",
                            "Medium: an issue requiring follow-up",
                            "High: a financial issue such as a duplicate charge",
                        ],
                    },
                    "refund_requested": {
                        "type": "noul",
                        "instructions": "Does the customer explicitly request a refund?",
                    },
                },
            },
        ]

    for record in records:
        result = decide(sampling_client, encoder, record, args.temperature)
        # print()
        print(f"\n=== {record['id']} ===")
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
