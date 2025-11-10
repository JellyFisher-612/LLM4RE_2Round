import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class LanguageDetector:
    """轻量级字符比例检测器，用于判断中英文。"""

    def detect_language(self, sentence: str, threshold: float = 0.5) -> str:
        text = sentence or ""
        # 移除非字母字符（保留中英文）
        text = re.sub(r'[\d\W_]+', '', text)
        if not text:
            return "unknown"

        chinese_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        latin_chars = sum(1 for ch in text if ch.isascii() and ch.isalpha())
        total = chinese_chars + latin_chars

        if total == 0:
            return "unknown"

        ratio = chinese_chars / total
        return "zh" if ratio >= threshold else "en"


class BilingualREPromptFormatter:
    """双语关系抽取 Prompt 格式化器，输出格式为三元列表。"""

    TASK_INSTRUCTION_ZH = "根据给定的文本和候选关系类型，提取三元组，并标注其中实体的粗粒度和细粒度类型"
    TASK_INSTRUCTION_EN = "Extract triples from the given text and relation candidates, annotating coarse-grained and fine-grained types for entities."

    OUTPUT_HEADER_ZH = "请参考示例对当前输入进行抽取，输出 JSON 结果："
    OUTPUT_HEADER_EN = "Please refer to the examples and extract triples from the current input. Output the JSON result:"

    def __init__(self, detection_threshold: float = 0.5):
        self.detector = LanguageDetector()
        self.detection_threshold = detection_threshold
        self.stats = {"zh_count": 0, "en_count": 0, "total_count": 0}

    def infer_language(self, sentence: str) -> str:
        return self.detector.detect_language(sentence, self.detection_threshold)

    def _format_example_zh(self, idx: int, ex: Dict[str, Any]) -> str:
        sentence = ex.get("sentence", "").strip()
        schema = ex.get("schema", [])
        coarse_types = ex.get("coarse_types", [])
        output = ex.get("output", [])
        output_str = json.dumps(output, ensure_ascii=False, separators=(',', ':'))
        return (
            f"示例{idx}：\n"
            f"输入：\n"
            f"句子:{sentence}\n"
            f"关系候选:{'、'.join(schema)}\n"
            f"实体粗粒度候选:{'、'.join(coarse_types)}\n"
            f"输出:\n"
            f"{output_str}"
        )

    def _format_example_en(self, idx: int, ex: Dict[str, Any]) -> str:
        sentence = ex.get("sentence", "").strip()
        schema = ex.get("schema", [])
        coarse_types = ex.get("coarse_types", [])
        output = ex.get("output", [])
        output_str = json.dumps(output, separators=(',', ':'))
        return (
            f"Example {idx}:\n"
            f"Input:\n"
            f"Sentence: {sentence}\n"
            f"Relation Candidates: {', '.join(schema)}\n"
            f"Entity Coarse-Grained Types: {', '.join(coarse_types)}\n"
            f"Output:\n"
            f"{output_str}"
        )

    def format(
        self,
        sample: Dict[str, Any],
        *,
        similar_samples: Optional[List[Dict[str, Any]]] = None,
        include_default_example: bool = False,
    ) -> str:
        sentence = sample.get("sentence", "")
        lang = self.infer_language(sentence)
        self._update_stats(lang)

        parts: List[str] = []

        # 0. Task instruction
        parts.append(self.TASK_INSTRUCTION_ZH if lang == "zh" else self.TASK_INSTRUCTION_EN)

        # 1. Example section (Rules are moved to system prompt, so omitted here)
        if similar_samples and len(similar_samples) > 0:
            if lang == "zh":
                parts.append("#相似参考示例（供类比）:")
                for i, ex in enumerate(similar_samples, 1):
                    if "output" in ex:
                        parts.append(self._format_example_zh(i, ex))
            else:
                parts.append("#Similar Reference Examples (for analogy):")
                for i, ex in enumerate(similar_samples, 1):
                    if "output" in ex:
                        parts.append(self._format_example_en(i, ex))
        elif include_default_example:
            if lang == "zh":
                default_ex = {
                    "sentence": "因为要避讳隋文帝之父杨忠，中江便改名为内江了",
                    "schema": ["改编自", "人口数量", "国籍", "父亲"],
                    "coarse_types": ["人","位置","科学"],
                    "output": [
                        {
                            "subject": ["隋文帝", "人", "君主"],
                            "relationship": "父亲",
                            "object": ["杨忠", "人", "君主"]
                        }
                    ]    
                }
                parts.append("#相似参考示例（供类比）:")
                parts.append(self._format_example_zh(1, default_ex))
            else:
                default_ex = {
                    "sentence": "While southern France traditionally produces a galette in the shape of a crown and garnished with candied fruits , the chic bakery houses of Paris have dared to take liberties .",
                    "schema": [
                        "country of capital",
                        "neighborhood of",
                        "administrative division of country",
                        "geographic distribution"
                    ],
                    "coarse_types": ["music", "location", "literature"],
                    "output": [
                        {
                            "subject": ["France", "location", "country"],
                            "relationship": "country of capital",
                            "object": ["Paris", "location", "city"]
                        },
                        {
                            "subject": ["Paris", "location", "city"],
                            "relationship": "administrative division of country",
                            "object": ["France", "location", "country"]
                        }
                    ]
                }
                parts.append("#Similar Reference Examples (for analogy):")
                parts.append(self._format_example_en(1, default_ex))

        # 2. Current input
        schema = sample.get("schema", [])
        coarse_types = sample.get("coarse_types", [])
        if lang == "zh":
            parts.append("#当前输入:")
            parts.append(f"句子:{sentence}")
            parts.append(f"关系候选:{'、'.join(schema)}")
            parts.append(f"实体粗粒度候选:{'、'.join(coarse_types)}")
        else:
            parts.append("#Current Input:")
            parts.append(f"Sentence: {sentence}")
            parts.append(f"Relation Candidates: {', '.join(schema)}")
            parts.append(f"Entity Coarse-Grained Types: {', '.join(coarse_types)}")

        # 3. Output header
        parts.append(self.OUTPUT_HEADER_ZH if lang == "zh" else self.OUTPUT_HEADER_EN)

        return "\n".join(parts).strip()

    def _update_stats(self, lang: str) -> None:
        self.stats["total_count"] += 1
        if lang == "zh":
            self.stats["zh_count"] += 1
        elif lang == "en":
            self.stats["en_count"] += 1


# Global formatter instance
PROMPT_FORMATTER = BilingualREPromptFormatter()

# System prompts (with full rules embedded)
RE_SYSTEM_PROMPT_ZH = """你是一名关系抽取专家，请严格按照指定格式抽取实体关系三元组，标注其中实体的粗粒度和细粒度。规则如下:
1. 输出 JSON: [{"subject": ["实体名", "粗粒度", "细粒度"], "relationship": "关系名", "object": ["实体名", "粗粒度", "细粒度"]}]
2. relationship 必须严格来自给定的关系候选列表，不得自行推断、改写或编造
3. 粗粒度必须来自实体粗粒度候选候选列表
4. 细粒度需合理、具体
5. 若无有效三元组，返回 []"""

RE_SYSTEM_PROMPT_EN = """You are a relation extraction expert. Strictly extract entity-relation triples and annotate coarse-grained and fine-grained types for entities. Rules:
1. Output JSON: [{"subject": ["entity_name", "coarse_grained", "fine_grained"], "relationship": "relation", "object": ["entity_name", "coarse_grained", "fine_grained"]}]
2. relationship must exactly match one from the relation candidates — do NOT infer, paraphrase, or invent
3. coarse_grained type must be from the entity coarse-grained type candidates
4. fine_grained type should be reasonable and specific
5. If no valid triples exist, return []"""

INSTRUCTION_ZH = "根据给定的文本和候选关系类型，提取三元组，并标注其中实体的粗粒度和细粒度类型"
INSTRUCTION_EN = "Extract triples from the given text and relation candidates, annotating coarse-grained and fine-grained types for entities."


def _load_dataset(source: Union[str, Path, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if isinstance(source, list):
        return source

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Data source does not exist: {path}")

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as fp:
            return [json.loads(line) for line in fp if line.strip()]

    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
        if isinstance(data, list):
            return data
        raise ValueError(f"Expected a list in JSON file, got {type(data)} from {path}")


def _write_dataset(output_path: Union[str, Path], rows: List[Dict[str, Any]]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".jsonl":
        with path.open("w", encoding="utf-8") as fp:
            for row in rows:
                fp.write(json.dumps(row, ensure_ascii=False))
                fp.write("\n")
    else:
        with path.open("w", encoding="utf-8") as fp:
            json.dump(rows, fp, ensure_ascii=False, indent=2)


def convert_to_training_data_format(
    data_source: Union[str, Path, List[Dict[str, Any]]],
    *,
    output_path: Optional[Union[str, Path]] = None,
    include_default_example: bool = False,
) -> List[Dict[str, Any]]:
    """Convert raw data (list/JSON/JSONL) to supervised fine-tuning format."""

    dataset = _load_dataset(data_source)
    converted_data: List[Dict[str, Any]] = []

    for idx, item in enumerate(dataset, start=1):
        print(f"Processing item {idx}/{len(dataset)}...")

        similar_samples = item.get("similar_samples")
        if similar_samples and not isinstance(similar_samples, list):
            similar_samples = list(similar_samples)  # best-effort fallback

        prompt = PROMPT_FORMATTER.format(
            item,
            similar_samples=similar_samples,
            include_default_example=include_default_example,
        )

        output_content = item.get("output", [])
        output_str = json.dumps(output_content, ensure_ascii=False, separators=(',', ':'))

        sentence = item.get("sentence", "")
        language = PROMPT_FORMATTER.infer_language(sentence)
        system_prompt = RE_SYSTEM_PROMPT_ZH if language == "zh" else RE_SYSTEM_PROMPT_EN
        instruction_text = INSTRUCTION_ZH if language == "zh" else INSTRUCTION_EN

        converted_item = {
            "instruction": instruction_text,
            "input": prompt,
            "output": output_str,
            "system": system_prompt,
            "history": [],
        }
        converted_data.append(converted_item)

    if output_path is not None:
        _write_dataset(output_path, converted_data)

    return converted_data


if __name__ == "__main__":
    source_path = Path("/home/users/lhy/LLM4RE_2Round/data/test2_rag.jsonl")
    target_path = Path("/home/users/lhy/LLM4RE_2Round/data/test2_rag_converted.json")

    converted = convert_to_training_data_format(
        source_path,
        include_default_example=True,
    )

    _write_dataset(target_path, converted)
    print(f"已将 {len(converted)} 条样本写入 {target_path}")