import argparse
import json
import os
from typing import Dict, List

from ir_system import InvertedIndex, VectorSpaceSearcher


def load_existing(path: str) -> Dict[str, List[str]]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_qrels(path: str, qrels: Dict[str, List[str]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(qrels, f, ensure_ascii=False, indent=2)


def annotate(query_file: str, out_file: str, topk: int = 20):
    idx = InvertedIndex()
    idx.load("data/index")
    searcher = VectorSpaceSearcher(idx)

    with open(query_file, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]

    qrels = load_existing(out_file)

    print("\n=== 人工标注工具 ===")
    print("输入格式：")
    print("- 相关文档序号，如: 1 3 5")
    print("- 跳过本查询: s")
    print("- 保存并退出: q\n")

    for q in queries:
        if q in qrels and qrels[q]:
            print(f"[跳过] 查询已存在标注: {q}")
            continue

        print("\n" + "=" * 80)
        print(f"Query: {q}")
        results = searcher.search(q, topk=topk)

        if not results:
            print("无检索结果，记为空标注。")
            qrels[q] = []
            save_qrels(out_file, qrels)
            continue

        for i, (score, doc, snippet) in enumerate(results, start=1):
            print(f"[{i:02d}] score={score:.4f} | {doc.doc_id} | {doc.title}")
            print(f"     date={doc.date} | lang={doc.lang}")
            print(f"     url={doc.url}")
            print(f"     snippet={snippet}\n")

        ans = input("请输入相关序号(如 1 2 8)，或 s 跳过，q 退出: ").strip().lower()
        if ans == "q":
            save_qrels(out_file, qrels)
            print(f"已保存到: {out_file}")
            return
        if ans == "s" or not ans:
            qrels[q] = []
            save_qrels(out_file, qrels)
            continue

        selected_ids = []
        for token in ans.split():
            if token.isdigit():
                idx_num = int(token)
                if 1 <= idx_num <= len(results):
                    selected_ids.append(results[idx_num - 1][1].doc_id)

        qrels[q] = sorted(set(selected_ids))
        save_qrels(out_file, qrels)
        print(f"已标注 {len(qrels[q])} 条相关文档。")

    print(f"\n全部查询处理完成，标注文件: {out_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="data/eval/queries.txt", help="每行一个查询")
    parser.add_argument("--out", default="data/eval/qrels.json", help="输出标注文件")
    parser.add_argument("--topk", type=int, default=20, help="每个查询显示前k条结果")
    args = parser.parse_args()

    annotate(args.queries, args.out, args.topk)


if __name__ == "__main__":
    main()
