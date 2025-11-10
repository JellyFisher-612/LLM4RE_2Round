import json
from pathlib import Path
import argparse

# ----------------------------
# 复用你的 postprocess 函数（仅保留 ensure_parsed_output 及依赖）
# ----------------------------

import re
from typing import Any, Dict, List

def normalize_generation_text(text: str) -> str:
    cleaned = text.strip().replace("\u200b", "")
    cleaned = re.sub(r"\s*\[/INST\]\s*$", "", cleaned)
    cleaned = re.sub(r"^(?:<s>|</s>|assistant:|Assistant:)\s*", "", cleaned)
    if "[/INST]" in cleaned:
        cleaned = cleaned.split("[/INST]")[-1]
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned)
        cleaned = cleaned.rstrip("`")
    return cleaned.strip()

def parse_completion(text: str, sample: Dict[str, Any]) -> Dict[str, Any]:
    stripped = text.strip()
    candidates = [stripped]

    if "```" in stripped:
        for match in re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.S):
            candidates.append(match)

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and start < end:
        candidates.append(stripped[start : end + 1])

    entities_match = re.search(r"entities\s*:\s*(\[.*\])", stripped, flags=re.S)
    if entities_match:
        try:
            entities_obj = json.loads(entities_match.group(1))
            constructed = json.dumps(
                {
                    "id": sample.get("id"),
                    "sentence": sample.get("sentence", ""),
                    "entities": entities_obj,
                },
                ensure_ascii=False,
            )
            candidates.append(constructed)
        except json.JSONDecodeError:
            pass

    kv_pattern = re.compile(r'"?(id|sentence|entities)"?\s*:\s*(.+)')
    if kv_pattern.search(stripped) and "{" not in stripped:
        parts = {}
        for line in stripped.splitlines():
            match = kv_pattern.match(line.strip().rstrip(","))
            if not match:
                continue
            key, value = match.groups()
            parts[key] = value
        obj = {
            "id": sample.get("id"),
            "sentence": sample.get("sentence", ""),
            "entities": [],
        }
        if "entities" in parts:
            try:
                obj["entities"] = json.loads(parts["entities"])
            except json.JSONDecodeError:
                pass
        candidates.append(json.dumps(obj, ensure_ascii=False))

    errors: List[str] = []
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError as err:
            errors.append(str(err))
            continue

        if isinstance(obj, list):
            obj = {"entities": obj}
        if (
            isinstance(obj, dict)
            and {"name", "coarse_type", "fine_type"}.issubset(obj.keys())
            and "entities" not in obj
        ):
            obj = {"entities": [obj]}

        if not isinstance(obj, dict):
            errors.append("解析后类型非字典")
            continue

        if "id" not in obj and "id" in sample:
            obj["id"] = sample["id"]
        if "sentence" not in obj:
            obj["sentence"] = sample.get("sentence", "")
        if "entities" not in obj:
            obj["entities"] = []

        entities = obj["entities"]
        if isinstance(entities, dict):
            entities = [entities]
        if not isinstance(entities, list):
            entities = []
        obj["entities"] = entities

        return obj

    raise ValueError("; ".join(errors) if errors else "无法解析模型输出")

def ensure_parsed_output(text: str, sample: Dict[str, Any]) -> Dict[str, Any]:
    try:
        parsed = parse_completion(text, sample)
        entities = parsed.get("entities", [])
        if isinstance(entities, dict):
            entities = [entities]
        if not isinstance(entities, list):
            entities = []
        return {
            "id": sample.get("id"),
            "sentence": sample.get("sentence", ""),
            "entities": entities,
        }
    except ValueError:
        return {
            "id": sample.get("id"),
            "sentence": sample.get("sentence", ""),
            "entities": [],
        }

# ----------------------------
# 主逻辑：按顺序对齐处理（支持外部参数）
# ----------------------------

def main(predictions_path: str, test_data_path: str, output_path: str):
    # 1. 加载测试数据（保持顺序）
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_samples = json.load(f)  # list of dicts

    # 2. 逐行读取预测结果（保持顺序）
    predict_strings = []
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                predict_strings.append(item.get("predict", ""))
            except json.JSONDecodeError:
                predict_strings.append("")  # 解析失败则为空

    # 3. 对齐并处理
    if len(test_samples) != len(predict_strings):
        print(f"⚠️ 警告：测试样本数 ({len(test_samples)}) 与预测行数 ({len(predict_strings)}) 不一致！")

    final_results = []
    for i, sample in enumerate(test_samples):
        if i >= len(predict_strings):
            predict_str = ""
        else:
            predict_str = predict_strings[i]

        # 使用你的 postprocess 函数解析
        result = ensure_parsed_output(predict_str, sample)
        final_results.append(result)

    # 4. 保存结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    print(f"✅ 成功处理 {len(final_results)} 条样本，结果已保存至 {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse and align model predictions with test data.")
    parser.add_argument("--predictions_path", type=str, required=True, help="Path to generated_predictions.jsonl")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to test data JSON file")
    parser.add_argument("--output_path", type=str, required=True, help="Output JSON file path")

    args = parser.parse_args()
    main(
        predictions_path=args.predictions_path,
        test_data_path=args.test_data_path,
        output_path=args.output_path
    )