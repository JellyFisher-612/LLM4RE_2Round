import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def normalize_generation_text(text: Optional[str]) -> str:
    if not text:
        return ""
    cleaned = text.strip()
    fence = re.compile(r"^```(?:json|python|text)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)
    match = fence.match(cleaned)
    if match:
        cleaned = match.group(1).strip()
    return cleaned


def extract_output(text: str) -> List[Dict[str, Any]]:
    cleaned = normalize_generation_text(text)
    if not cleaned:
        return []

    candidates: List[Any] = []
    try:
        candidates.append(json.loads(cleaned))
    except json.JSONDecodeError:
        pass

    if not candidates:
        start, end = cleaned.find("["), cleaned.rfind("]")
        if start != -1 and end != -1:
            snippet = cleaned[start : end + 1]
            try:
                candidates.append(json.loads(snippet))
            except json.JSONDecodeError:
                try:
                    candidates.append(ast.literal_eval(snippet))
                except (ValueError, SyntaxError):
                    pass

    if not candidates:
        try:
            candidates.append(ast.literal_eval(cleaned))
        except (ValueError, SyntaxError):
            return []

    for candidate in candidates:
        if isinstance(candidate, dict) and "output" in candidate:
            payload = candidate["output"]
        elif isinstance(candidate, list):
            payload = candidate
        else:
            continue

        if isinstance(payload, dict):
            payload = [payload]
        if isinstance(payload, list):
            triples = [item for item in payload if isinstance(item, dict)]
            return triples
    return []


def ensure_parsed_output(text: str, sample: Dict[str, Any], index: int) -> Dict[str, Any]:
    output = extract_output(text)
    sample_id = sample.get("id") or f"sample_{index:05d}"
    sentence = sample.get("sentence", "")
    return {"id": sample_id, "sentence": sentence, "output": output}


def load_test_data(path: Path) -> List[Dict[str, Any]]:
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in content.splitlines() if line.strip()]
    return json.loads(content)


def load_predictions(path: Path) -> List[str]:
    lines = []
    with path.open("r", encoding="utf-8") as fp:
        for raw in fp:
            raw = raw.strip()
            if not raw:
                continue
            record = json.loads(raw)
            if isinstance(record, str):
                lines.append(record)
                continue
            for key in ("generation", "text", "output_text", "response", "predict"):
                if key in record and isinstance(record[key], str):
                    lines.append(record[key])
                    break
            else:
                lines.append("")
    return lines


def write_results(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    data = list(rows)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract LLM predictions into eval-ready format.")
    parser.add_argument("--predictions_path", required=True, type=Path, help="LLM prediction JSONL path.")
    parser.add_argument("--test_data_path", required=True, type=Path, help="Original test data JSON/JSONL path.")
    parser.add_argument("--output_path", required=True, type=Path, help="Where to save converted results (JSON).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_samples = load_test_data(args.test_data_path)
    predict_strings = load_predictions(args.predictions_path)

    final_results: List[Dict[str, Any]] = []
    for idx, sample in enumerate(test_samples):
        prediction = predict_strings[idx] if idx < len(predict_strings) else ""
        final_results.append(ensure_parsed_output(prediction, sample, idx))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_results(args.output_path, final_results)
    print(f"[INFO] 总计 {len(final_results)} 条结果写入 {args.output_path}")


if __name__ == "__main__":
    main()


# python /home/users/lhy/LLM4RE_2Round/get_predict.py \
#   --predictions_path /home/users/lhy/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/eval_2025-11-5-21-46/generated_predictions.jsonl \
#   --test_data_path /home/users/lhy/LLM4RE_2Round/data/test2.json \
#   --output_path /home/users/lhy/LLM4RE_2Round/data/test2_llm_outputs.json