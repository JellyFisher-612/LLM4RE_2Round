import json
import os
import argparse
from tqdm import tqdm
from collections import Counter
from typing import List, Dict, Any
from rag_utils import Retriever, detect_language, separate_by_language

def load_json_or_jsonl(path: str) -> List[Dict[Any, Any]]:
    """加载 JSON 或 JSONL 文件"""
    assert os.path.exists(path), f"File not found: {path}"
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
    else:
        raise ValueError(f"Unsupported file type: {path}")
    print(f"加载完成 {path}, 样本数: {len(data)}")
    return data

def filter_empty_outputs(samples: List[Dict[Any, Any]], key: str = "output") -> List[Dict[Any, Any]]:
    """过滤掉指定字段为空列表的样本"""
    before = len(samples)
    filtered = [s for s in samples if not (isinstance(s.get(key), list) and len(s.get(key)) == 0)]
    after = len(filtered)
    print(f"   🔎 已过滤掉 {before - after} 条 {key} 为空的样本，剩余 {after} 条。")
    return filtered

def main():
    parser = argparse.ArgumentParser(description="基于向量检索的样本增强 (仅需合并知识库)")
    parser.add_argument("--knowledge_base_path", type=str, required=True, help="合并后的中英文知识库路径 (.json/.jsonl)")
    parser.add_argument("--data_path", type=str, required=True, help="待增强数据路径 (.json/.jsonl)")
    parser.add_argument("--output_path", type=str, required=True, help="增强结果输出路径 (.json)")
    parser.add_argument("--text_key", type=str, default="input", help="用于检索的文本字段")
    parser.add_argument("--similarity_threshold", type=float, default=0.5, help="相似度阈值")
    args = parser.parse_args()

    print(f"\n📚 加载合并知识库: {args.knowledge_base_path}")
    combined_kb = load_json_or_jsonl(args.knowledge_base_path)
    print(f"   样本总数: {len(combined_kb)}")

    print(f"🔀 按语言分离知识库 (字段: '{args.text_key}')...")
    kb_samples_zh, kb_samples_en, kb_samples_other = separate_by_language(combined_kb, text_key=args.text_key)

    # ✅ 新增：过滤 output 为空的样本
    kb_samples_zh = filter_empty_outputs(kb_samples_zh, key="output")
    kb_samples_en = filter_empty_outputs(kb_samples_en, key="output")

    # === 构建 Retriever ===
    print("\n🚀 构建向量索引...")
    retriever_zh = Retriever(kb_samples_zh, key=args.text_key)
    retriever_en = Retriever(kb_samples_en, key=args.text_key)
    print(f"   ✅ 中文索引构建完成 ({len(kb_samples_zh)} 条样本)")
    print(f"   ✅ 英文索引构建完成 ({len(kb_samples_en)} 条样本)")

    print(f"\n📂 加载待增强样本: {args.data_path}")
    samples = load_json_or_jsonl(args.data_path)
    print(f"   待增强样本数: {len(samples)}")

    print(f"\n🎯 开始检索相似样本 (每条样本选取 2 个: 一个 output 为空，一个 output 非空)")
    augmented = []
    stats = Counter()

    for s in tqdm(samples, desc="Processing queries"):
        query = s.get(args.text_key, "")
        if not query.strip():
            stats["zero"] += 1
            continue

        detected_lang = detect_language(query)
        s["detected_language"] = detected_lang

        retriever = retriever_zh if detected_lang == 'zh' else retriever_en
        try:
            # 多取一些结果，用于筛选
            examples, sims = retriever.retrieve(query=query, top_k=20, threshold=args.similarity_threshold)
        except Exception as e:
            print(f"⚠️ 检索失败: {query[:50]}... Error: {e}")
            stats["zero"] += 1
            continue

        if not examples:
            stats["zero"] += 1
            continue

        # 新的筛选逻辑：选 1 个 output 非空 + 1 个 output 为空
        selected_examples = []
        selected_sims = []
        has_empty = False
        has_nonempty = False

        for ex, sim in zip(examples, sims):
            ex_text = ex.get(args.text_key, "").strip()
            ex_output = ex.get("output", [])

            # 排除自身
            if sim > 0.95 or ex_text == query.strip():
                continue

            # 优先选一个 output 非空，一个 output 空
            if not has_nonempty and ex_output:
                selected_examples.append(ex)
                selected_sims.append(sim)
                has_nonempty = True
            elif not has_empty and (isinstance(ex_output, list) and len(ex_output) == 0):
                selected_examples.append(ex)
                selected_sims.append(sim)
                has_empty = True

            # 两类都找到了就停止
            if has_nonempty and has_empty:
                break

        if not selected_examples:
            stats["zero"] += 1
            continue

        s["similar_samples"] = selected_examples
        s["similarity_scores"] = selected_sims

        # 统计信息
        if has_nonempty and has_empty:
            stats["full"] += 1
        else:
            stats["partial"] += 1

        augmented.append(s)

    # === 输出统计 ===
    total = len(samples)
    print(f"\n🔍 检索统计：")
    print(f"  完整结果 (找到空+非空样本): {stats['full']}")
    print(f"  部分结果: {stats['partial']}")
    print(f"  零结果: {stats['zero']}")
    print(f"  覆盖率: {(1 - stats['zero']/total)*100:.1f}%")

    # === 保存结果 ===
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(augmented, f, ensure_ascii=False, indent=2)
    print(f"\n💾 增强结果已保存到: {args.output_path}")


if __name__ == "__main__":
    main()

# python rag_enhance.py \
#   --knowledge_base_path /root/autodl-tmp/LLM4RE_2Round/data/train2.json \
#   --data_path /root/autodl-tmp/LLM4RE_2Round/data/train2.json \
#   --output_path /home/users/lhy/LLM4RE_2Round/data/rag_train2.json \
#   --k 2 \
#   --similarity_threshold 0.6