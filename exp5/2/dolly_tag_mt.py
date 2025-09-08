# =========================================================
# 实验名称（多线程）：分片并发打标（复用单线程函数 tag_records）
# 面向对象：大二学生（了解线程池、列表分片与结果合并）
#
# 实验目标：
# 1) 读取 JSONL，按 --threads 将样本平均分片。
# 2) 使用 ThreadPoolExecutor 并发地调用 dolly_tag_worker.tag_records(...)。
# 3) 合并所有线程的返回结果，按 id 排序，写出 JSONL 与 CSV，并打印标签分布。
#
# 使用说明：
#   先完成并确保 dolly_tag_worker.py 中的单线程函数 tag_records 可用，
#   然后运行本脚本：
#     export DASHSCOPE_API_KEY=sk-xxx
#     python dolly_tag_mt.py --in_jsonl ./your_dataset.jsonl --model qwen-plus --n 200 --threads 6
# =========================================================

# dolly_tag_mt.py
# 依赖：pip install openai==1.* pandas tqdm
import argparse, json, math
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from dolly_tag_worker import tag_records  # 复用单线程函数

def _load_jsonl(path, limit=None):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if line.strip() == "":
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            records.append({
                "id": obj.get("id", idx),
                "instruction": obj.get("instruction", "") or "",
                "context": obj.get("context", "") or "",
                "dolly_category": obj.get("category", obj.get("dolly_category", "")) or ""
            })
            if limit is not None and len(records) >= limit:
                break
    return records

def _chunk_even(lst, parts):
    n = max(1, parts)
    size = math.ceil(len(lst) / n)
    return [lst[i:i+size] for i in range(0, len(lst), size)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_jsonl", required=True, help="输入数据集（JSONL，每行一个样本）")
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--n", type=int, default=None, help="最多处理前 N 条（默认全量）")
    parser.add_argument("--threads", type=int, default=4, help="并发线程数")
    parser.add_argument("--out_jsonl", default="tag_results_mt.jsonl")
    parser.add_argument("--out_csv", default="tag_results_mt.csv")
    args = parser.parse_args()

    # 读数
    records = _load_jsonl(args.in_jsonl, limit=args.n)
    if not records:
        print("未读取到样本，请检查 JSONL 格式与路径。")
        return

    # 分片
    shards = _chunk_even(records, args.threads)

    # 并发执行：每个线程调用“单线程函数”处理一个分片
    results = []
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futures = [ex.submit(tag_records, shard, args.model, False) for shard in shards]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="并发分片完成"):
            results.extend(fut.result())

    # 合并 & 排序
    results.sort(key=lambda x: x["id"])

    # 落盘
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pd.DataFrame(results).to_csv(args.out_csv, index=False)
    print(f"✅ 并发结果已写入 {args.out_jsonl} 和 {args.out_csv}")

    # 简单统计
    dist = pd.Series([r["tag"] for r in results]).value_counts().to_dict()
    print("标签分布：", dist)

if __name__ == "__main__":
    main()
