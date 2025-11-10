import json
import re

RAW_DATA_PATH = "/root/autodl-tmp/LLM4RE_2Round/data/test2.json"
OUTPUT_PATH = "/root/autodl-tmp/LLM4RE_2Round/data/step1_test2.json"

class LanguageDetector:
    def detect_language(self, sentence: str, threshold: float = 0.5) -> str:
        text = sentence or ""
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

# 中文 Prompt（使用中文术语，输出 yes/no）
SYSTEM_PROMPT_ZH = (
    "你是一个信息抽取任务的前置过滤器。请判断句子中是否存在潜在的、可抽取的关系三元组。"
    "如果句子中至少提及了两个实体，并且这两个实体之间存在可能的语义关联（即使不确定具体关系或类型是否完全匹配候选列表），则输出“yes”；"
    "如果句子中实体数量少于两个，或者实体间明显不存在任何语义联系，则输出“no”。\n"
    "\n"
    "判断要点：\n"
    "- 优先考虑实体数量：至少要有两个可识别的实体（如人物、地点、组织、物品或概念等）。\n"
    "- 其次考虑实体间的关联：即使具体关系不在候选列表中，或者实体类型不完全匹配，只要“可能”存在某种关联，就倾向于输出“yes”。\n"
    "- 仅当实体间明显不存在关联时才输出“no”（例如：只出现单个实体、无关联的实体列表、或仅描述事件但无明确实体关系）。\n"
    "\n"
    "示例（供参考类比）：\n"
    "句子1：《夜曲》由周杰伦演唱。\n"
    "关系候选：[...]\n"
    "实体粗类型候选：[...]\n"
    "→ 《夜曲》（音乐）和 周杰伦（人物）存在明确关联 → 输出：yes\n"woshinidie
    "\n"
    "句子2：《青花瓷》很受欢迎。\n"
    "→ 只提及了《青花瓷》一个实体 → 输出：no\n"
    "\n"
    "句子3：张艺谋和巩俐曾合作多部电影。\n"
    "关系候选：[...]\n"
    "实体粗类型候选：[...]\n"
    "→ 张艺谋（人物）和 巩俐（人物）存在明确关联（合作） → 输出：yes（即使“合作”不在候选列表中，也存在潜在关系）\n"
    "\n"
    "请根据以上原则进行判断。只输出“yes”或“no”，不要解释，不要输出其他内容。"
)



# 英文 Prompt（使用英文术语，输出 yes/no）
SYSTEM_PROMPT_EN = (
    "You are a pre-filter for an information extraction task. Determine whether the sentence contains any *potentially extractable* relational triples. "
    "If the sentence mentions at least two entities and there is a possible semantic connection between them (even if the exact relation or entity types do not perfectly match the candidate lists), output 'yes'; "
    "if the sentence contains fewer than two entities, or if there is clearly no semantic relationship between the entities, output 'no'.\n"
    "\n"
    "Guidelines for judgment:\n"
    "- Prioritize the number of entities: there must be at least two identifiable entities (such as people, locations, organizations, objects, or concepts).\n"
    "- Then consider the connection between entities: even if the specific relation is not in the candidate list, or the entity types do not fully match, if there *might* be some kind of relationship, lean toward 'yes'.\n"
    "- Use 'no' only when there is clearly no connection between entities (e.g., a single entity, an unconnected list of entities, or a sentence describing an event without explicit entity relations).\n"
    "\n"
    "Examples (for reference and analogy):\n"
    "Sentence 1: 'Nocturne' was sung by Jay Chou.\n"
    "Relation candidates: [...]\n"
    "Entity coarse types: [...]\n"
    "→ 'Nocturne' (music) and 'Jay Chou' (person) have an explicit connection → Output: yes\n"
    "\n"
    "Sentence 2: 'Blue and White Porcelain' is very popular.\n"
    "→ Only one entity ('Blue and White Porcelain') is mentioned → Output: no\n"
    "\n"
    "Sentence 3: Zhang Yimou and Gong Li collaborated on several films.\n"
    "Relation candidates: [...]\n"
    "Entity coarse types: [...]\n"
    "→ Zhang Yimou (person) and Gong Li (person) have a clear connection (collaboration) → Output: yes (even if 'collaboration' is not in the candidate list, there is a potential relationship)\n"
    "\n"
    "Follow the above principles for judgment. Output only 'yes' or 'no' — do not explain or include any other content."
)




def convert_raw_to_filter(raw_data):
    detector = LanguageDetector()
    result = []
    for item in raw_data:
        sentence = item["sentence"]
        lang = detector.detect_language(sentence, threshold=0.5)

        # 选择 system prompt
        system_prompt = SYSTEM_PROMPT_EN if lang != "zh" else SYSTEM_PROMPT_ZH

        # 选择 input 的语言引导词
        if lang == "zh":
            prefix_sentence = "句子"
            prefix_relations = "关系候选"
            prefix_coarse = "实体粗类型候选"
        else:
            prefix_sentence = "Sentence"
            prefix_relations = "Relation candidates"
            prefix_coarse = "Entity coarse type candidates"

        # 构造 input（保持 schema 和 coarse_types 原样）
        relations = "、".join(item["schema"])
        coarse_types = "、".join(item["coarse_types"])

        input_text = (
            f"{prefix_sentence}：{sentence}\n"
            f"{prefix_relations}：{relations}\n"
            f"{prefix_coarse}：{coarse_types}"
        )

        # 输出标签统一为英文小写
        # 若 output 不存在，则设为空列表
        output_field = item.get("output", [])

        # 若为空则直接设为 []，否则转为小写
        if not output_field:
            label = []
        else:
            # 若原 output 不是列表（例如 "yes"/"no"），则标准化为小写字符串
            if isinstance(output_field, str):
                label = output_field.lower()
            else:
                label = output_field  # 保留原结构（一般是关系三元组列表）

        result.append({
            "system": system_prompt,
            "instruction": "",
            "input": input_text,
            "output": label,
            "history": [] 
        })
    return result

if __name__ == "__main__":
    with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    filter_data = convert_raw_to_filter(raw_data)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(filter_data, f, ensure_ascii=False, indent=2)

    num_yes = sum(1 for x in filter_data if x["output"] == "yes")
    print(f"✅ 转换完成！")
    print(f"   总样本: {len(filter_data)}")
    print(f"   正样本（yes）: {num_yes}")
    print(f"   负样本（no）: {len(filter_data) - num_yes}")
    print(f"   输出文件: {OUTPUT_PATH}")