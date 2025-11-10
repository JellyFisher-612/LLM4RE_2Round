#!/usr/bin/env python3
# count_empty_output.py

import json
import sys
from pathlib import Path

def walk(obj, callback):
    """递归遍历 JSON 结构，每当遇到 output 字段就回调。"""
    if isinstance(obj, dict):
        if "output" in obj:
            callback(obj["output"])
        for value in obj.values():
            walk(value, callback)
    elif isinstance(obj, list):
        for item in obj:
            walk(item, callback)

def main(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    total = 0
    empty = 0

    def handle_output(value):
        nonlocal total, empty
        total += 1
        if isinstance(value, list) and len(value) == 0:
            empty += 1

    walk(data, handle_output)

    if total == 0:
        print("未在文件中找到任何 output 字段。")
        return

    percentage = empty / total * 100
    print(f"output 字段总数   : {total}")
    print(f"output 为 [] 的数量: {empty}")
    print(f"占比               : {percentage:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python count_empty_output.py <文件路径>")
        sys.exit(1)
    main(Path(sys.argv[1]))