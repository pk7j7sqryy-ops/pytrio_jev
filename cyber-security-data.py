import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com uv run python jev/cyber-security-data.py"
from datasets import load_dataset, Dataset

# 流式，不缓存全部分片
ds_stream = load_dataset("Vyber07/cyber-security", split="train", streaming=True)

# 只拿前200条，这200条才会落到你本地
samples = list(ds_stream.take(200))
ds_sub = Dataset.from_list(samples)

# 保存这一小部分到本地
ds_sub.save_to_disk("./datasets/cyber_200_stream")
