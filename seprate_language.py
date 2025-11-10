#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 LLM4RE_v4 格式的 JSON 数据集按语言（中文/英文）分离
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm
from langdetect import detect, LangDetectException


def load_json_or_jsonl(path: str):
    """加载JSON或JSONL文件"""
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    else:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)


def write_jsonl(path: str, rows):
    """写入JSONL文件"""
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")


def separate_by_language(data, text_key="sentence"):
    """
    根据语言分离数据。

    Args:
        data (list): 包含数据样本的列表。
        text_key (str): 包含待检测文本的字段名。默认为 "sentence"。

    Returns:
        tuple: 包含三个列表的元组 (chinese_data, english_data, other_data)
    """
    chinese_samples = []
    english_samples = []
    other_samples = []

    for item in tqdm(data, desc="Detecting language"):
        try:
            text_to_detect = item.get(text_key, "")
            if not text_to_detect or not isinstance(text_to_detect, str) or len(text_to_detect.strip()) == 0:
                # 如果文本为空或非字符串，归入 other
                other_samples.append(item)
                continue

            # 检测语言
            detected_lang = detect(text_to_detect)

            # 根据 langdetect 的返回值进行分类
            if detected_lang.startswith('zh'): # 'zh', 'zh-cn', 'zh-hk', 'zh-tw', 'zh-yue' 等
                chinese_samples.append(item)
            elif detected_lang == 'en':
                english_samples.append(item)
            else:
                other_samples.append(item)

        except LangDetectException:
            # 如果检测失败，归入 other
            other_samples.append(item)
        except Exception as e:
            # 其他潜在错误，也归入 other
            print(f"Warning: Error processing item: {item.get(text_key, '')[:50]}... Error: {e}")
            other_samples.append(item)

    return chinese_samples, english_samples, other_samples


def main():
    # 设置默认路径
    default_input_path = "/home/users/lhy/LLM4RE_2Round/data/dev2.json"
    default_output_dir = "/home/users/lhy/LLM4RE_2Round/data" # 默认输出目录

    parser = argparse.ArgumentParser(description="将 LLM4RE_v4 格式的 JSON 数据集按语言（中文/英文）分离")
    parser.add_argument("--input_path", type=str, default=default_input_path, help=f"输入数据集路径 (JSON 或 JSONL), 默认: {default_input_path}")
    parser.add_argument("--output_dir", type=str, default=default_output_dir, help=f"输出目录路径, 默认: {default_output_dir}")
    parser.add_argument("--text_key", type=str, default="sentence", help="包含待检测文本的字段名，默认为 'sentence'")

    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)

    # 检查输入文件是否存在
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_path}")
        return

    print(f"📂 加载数据: {input_path}")
    try:
        data = load_json_or_jsonl(str(input_path))
    except json.JSONDecodeError as e:
        print(f"错误: 输入文件不是有效的 JSON 格式: {e}")
        return
    except Exception as e:
        print(f"错误: 读取输入文件时出现问题: {e}")
        return

    print(f"🔍 开始按语言分离数据 (文本字段: '{args.text_key}')...")
    chinese_data, english_data, other_data = separate_by_language(
        data, text_key=args.text_key
    )

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 定义输出文件名
    base_name = input_path.stem
    zh_output_path = output_dir / f"{base_name}_zh.jsonl"
    en_output_path = output_dir / f"{base_name}_en.jsonl"
    other_output_path = output_dir / f"{base_name}_other.jsonl"

    # 写入文件
    print(f"📝 保存中文数据到: {zh_output_path}")
    write_jsonl(str(zh_output_path), chinese_data)

    print(f"📝 保存英文数据到: {en_output_path}")
    write_jsonl(str(en_output_path), english_data)

    if other_data: # 如果 other_data 不为空
        print(f"📝 保存其他语言/无法识别数据到: {other_output_path}")
        write_jsonl(str(other_output_path), other_data)

    # 输出统计
    print("\n📊 分离结果统计:")
    print(f"  总样本数: {len(data)}")
    print(f"  中文样本数: {len(chinese_data)}")
    print(f"  英文样本数: {len(english_data)}")
    print(f"  其他/无法识别样本数: {len(other_data)}")
    print(f"✅ 数据分离完成！")


if __name__ == "__main__":
    main()