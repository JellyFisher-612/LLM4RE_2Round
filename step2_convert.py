import json
import re

RAW_DATA_PATH = "/root/autodl-tmp/LLM4RE_2Round/data/train2.json"
OUTPUT_PATH = "/root/autodl-tmp/LLM4RE_2Round/data/step1_train2.json"

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
    "你是一个信息抽取任务的前置过滤器。只有当以下三个条件同时满足时，输出“yes”；否则输出“no”：\n"
    "1. 句子中至少明确提及了两个实体；\n"
    "2. 这两个实体之间存在于【关系候选】中的某一关系；\n"
    "3. 这两个实体的粗粒度类型均能合理映射到【实体粗类型候选】中（例如：人→生物，国家→组织机构，歌曲/专辑→音乐）。\n"
    "\n"
    "示例（供参考类比）：\n"
    "句子1：《夜曲》由周杰伦演唱。\n关系候选：演唱者\n实体粗类型候选：音乐、生物\n"
    "→ 《夜曲》→音乐 ✓；周杰伦→生物 ✓；关系“演唱者”存在 ✓ → 输出：yes\n"
    "\n"
    "句子2：《青花瓷》很受欢迎。\n关系候选：所属专辑\n实体粗类型候选：音乐、组织机构\n"
    "→ 仅出现一个实体 → 条件1不满足 → 输出：no\n"
    "\n"
    "句子3：张艺谋出生于1956年。\n关系候选：导演\n实体粗类型候选：生物、时间\n"
    "→ 张艺谋→生物 ✓；1956年→时间 ✓；但关系不是“导演” → 条件2不满足 → 输出：no"
    "\n"
    "请严格按上述规则判断。只输出“yes”或“no”，不要解释，不要输出其他内容。"
)


# 英文 Prompt（使用英文术语，输出 yes/no）
SYSTEM_PROMPT_EN = (
    "You are a pre-filter for an information extraction task. Output “yes” only when the following three conditions "
    "are all satisfied; otherwise output “no”:\n"
    "1. The sentence explicitly mentions at least two entities;\n"
    "2. There exists a relationship between these two entities that appears in the [candidate relations];\n"
    "3. The coarse-grained types of both entities can be reasonably mapped to the [candidate entity coarse types] "
    "(for example: person→organism, country→organization, song/album→music).\n"
    "\n"
    "Examples (for reference and analogy):\n"
    "Sentence 1: “Nocturne” was performed by Jay Chou.\nCandidate relations: performer\nCandidate entity coarse types: music, organism\n"
    "→ “Nocturne”→music ✓; Jay Chou→organism ✓; relation “performer” exists ✓ → Output: yes\n"
    "\n"
    "Sentence 2: “Blue and White Porcelain” is very popular.\nCandidate relations: album_of\nCandidate entity coarse types: music, organization\n"
    "→ Only one entity appears → Condition 1 not satisfied → Output: no\n"
    "\n"
    "Sentence 3: Zhang Yimou was born in 1956.\nCandidate relations: director\nCandidate entity coarse types: organism, time\n"
    "→ Zhang Yimou→organism ✓; 1956→time ✓; but the relation is not “director” → Condition 2 not satisfied → Output: no"
    "\n"
    "Please strictly follow the above rules. Output only “yes” or “no”. Do not provide explanations or any other content."
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
        label = "yes" if len(item["output"]) > 0 else "no"

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