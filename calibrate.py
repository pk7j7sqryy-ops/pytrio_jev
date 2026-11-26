"""温度校准：在 calibration 划分上拟合标量温度 T，使候选分布的 NLL 最小。

p_T = softmax(logprob / T)，T 只改概率形状、不改 argmax。
搜索用 log T 上的黄金分割（NLL 关于 1/T 是凸的，这里直接做无导数搜索）。

用法：
uv run python 09-pytrio-jev/calibrate.py --weights trio://...
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from pathlib import Path

import numpy as np
import pytrio as trio
from dotenv import load_dotenv
from tqdm import tqdm

from data import Encoder, compile_record, load_jsonl
from inference import candidate_logprobs_async

SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / "jev" / ".env")
DEFAULT_DATA = SCRIPT_DIR / "datasets" / "calibration.jsonl"
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs" / "calibration.json"
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"


async def collect_instances(
    sampling_client: trio.SamplingClient,
    encoder: Encoder,
    records: list[dict],
    concurrency: int,
) -> list[tuple[np.ndarray, int]]:
    """逐字段收集 (候选 logprob 向量, gold 下标)。"""
    semaphore = asyncio.Semaphore(concurrency)
    per_record: list = [None] * len(records)

    with tqdm(total=len(records), desc="收集校准预测", unit="条") as progress:

        async def run_one(index: int, record: dict) -> None:
            targets = compile_record(record, include_targets=True)["targets"]
            async with semaphore:
                collected = await candidate_logprobs_async(
                    sampling_client, encoder, record
                )
            encoded, logprobs = collected["encoded"], collected["logprobs"]
            instances = []
            for field, options in encoded["options"].items():
                symbols = [symbol for symbol, _, _ in options]
                gold = symbols.index(targets[field])
                instances.append(
                    (np.array([logprobs[field][s] for s in symbols]), gold)
                )
            per_record[index] = instances
            progress.update(1)

        await asyncio.gather(
            *(run_one(index, record) for index, record in enumerate(records))
        )
    return [instance for instances in per_record for instance in instances]


def mean_nll(instances: list[tuple[np.ndarray, int]], temperature: float) -> float:
    total = 0.0
    for logits, gold in instances:
        z = logits / temperature
        total += float(np.log(np.exp(z - z.max()).sum()) + z.max() - z[gold])
    return total / len(instances)


def fit_temperature(instances: list[tuple[np.ndarray, int]]) -> float:
    """在 log T ∈ [log 0.01, log 100] 上做黄金分割搜索。"""
    lo, hi = math.log(0.01), math.log(100.0)
    inv_phi = (math.sqrt(5) - 1) / 2
    for _ in range(80):
        c = hi - inv_phi * (hi - lo)
        d = lo + inv_phi * (hi - lo)
        if mean_nll(instances, math.exp(c)) < mean_nll(instances, math.exp(d)):
            hi = d
        else:
            lo = c
    return math.exp((lo + hi) / 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="温度校准")
    parser.add_argument(
        "--base-model", default=os.getenv("JEV_BASE_MODEL", DEFAULT_MODEL)
    )
    weights = os.getenv("JEV_CALIBRATION_WEIGHTS") or None
    parser.add_argument(
        "--weights",
        default=weights,
        required=weights is None,
        help="trio:// LoRA 权重路径",
    )
    data_path = Path(os.getenv("JEV_CALIBRATION_DATA") or DEFAULT_DATA)
    if not data_path.is_absolute():
        data_path = SCRIPT_DIR / data_path
    output_path = Path(os.getenv("JEV_CALIBRATION_OUTPUT") or DEFAULT_OUTPUT)
    if not output_path.is_absolute():
        output_path = SCRIPT_DIR / output_path
    parser.add_argument("--data", type=Path, default=data_path)
    parser.add_argument("--output", type=Path, default=output_path)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.getenv("JEV_CALIBRATION_CONCURRENCY", "15")),
        help="并发收集条数",
    )
    return parser.parse_args()


async def main(args: argparse.Namespace) -> None:
    service_client = trio.ServiceClient()
    sampling_client = await service_client.create_sampling_client_async(
        base_model=args.base_model, model_path=args.weights
    )
    encoder = Encoder(sampling_client.get_tokenizer())

    records = load_jsonl(args.data)
    instances = await collect_instances(
        sampling_client, encoder, records, args.concurrency
    )
    print(f"校准实例：{len(instances)} 个字段（来自 {len(records)} 条记录）")

    nll_before = mean_nll(instances, 1.0)
    temperature = fit_temperature(instances)
    nll_after = mean_nll(instances, temperature)
    print(f"NLL: T=1 → {nll_before:.4f} | T={temperature:.6f} → {nll_after:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "temperature": temperature,
                "weights": args.weights,
                "instances": len(instances),
                "nll_t1": nll_before,
                "nll_fitted": nll_after,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"写出 {args.output}")


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
