import json
import os
import argparse

def load_json_or_jsonl(path):
    """加载 JSON 或 JSONL 文件"""
    assert os.path.exists(path), f"❌ File not found: {path}"
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
    else:
        raise ValueError(f"Unsupported file type: {path}")
    print(f"✅ 已加载 {path}, 样本数: {len(data)}")
    return data

def extract_predicts(data):
    """提取 predict 字段"""
    predicts = []
    for i, item in enumerate(data):
        if "predict" in item:
            predicts.append(item["predict"])
        else:
            print(f"⚠️ 第 {i} 条数据缺少 'predict' 字段，已跳过。")
    print(f"✅ 共提取 {len(predicts)} 条 predict")
    return predicts

def save_json(data, path):
    """保存为 JSON 文件"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"💾 已保存到 {path}")

def main():
    parser = argparse.ArgumentParser(description="提取 JSON/JSONL 文件中的 predict 字段")
    parser.add_argument("--input_path", type=str, required=True, help="输入文件路径 (.json 或 .jsonl)")
    parser.add_argument("--output_path", type=str, required=True, help="输出 JSON 文件路径")
    args = parser.parse_args()

    data = load_json_or_jsonl(args.input_path)
    predicts = extract_predicts(data)
    save_json(predicts, args.output_path)

if __name__ == "__main__":
    main()


# python /root/autodl-tmp/LLM4RE_2Round/extract_step1.py \
#   --input_path /root/autodl-tmp/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/dev_2025-11-9/generated_predictions.jsonl \
#   --output_path /root/autodl-tmp/LLM4RE_2Round/data/dev_predict_yes_list.json
