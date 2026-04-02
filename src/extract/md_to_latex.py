"""
Markdown to LaTeX JSONL Conversion Module
从 Markdown 文件中提取行间公式（$$...$$）并保存为 JSONL 格式
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict

def mask_fenced_code_blocks_keep_len(md: str) -> str:
    """
    用空格掩码 ```...``` 代码块，保持长度不变，
    这样 match.start() 位置仍然对应原始内容。
    """
    def repl(m: re.Match) -> str:
        return " " * (m.end() - m.start())
    return re.sub(r"```.*?```", repl, md, flags=re.DOTALL)

def extract_md_to_latex(
    input_md: str = None,
    output_dir: str = None
) -> str:
    """
    从 Markdown 文件中提取行间公式（$$...$$），
    不进行额外处理，直接输出为 JSONL 格式。
    
    参数：
        input_md: 输入的 Markdown 文件路径
                 （默认为环境变量或预设路径）
        output_dir: 输出文件夹路径
                   （默认为环境变量或预设路径）
    
    返回：
        生成的 JSONL 文件路径
    """
    # 获取输入文件路径（优先级：参数 > 环境变量 > 默认值）
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    _DATA_DIR = _PROJECT_ROOT / "data"
    input_path = (
        input_md
        or os.environ.get("PDF_TO_EMBED_INPUT_MD")
        or str((_DATA_DIR / "md" / "tilt_rotor" / "tilt_rotor.md").resolve())
    )
    
    # 获取输出文件夹路径
    output_base = (
        output_dir
        or os.environ.get("PDF_TO_EMBED_OUTPUT_DIR")
        or str((_DATA_DIR / "latex").resolve())
    )

    # 确保输出目录存在
    os.makedirs(output_base, exist_ok=True)
    
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在 - {input_path}")

    # 读取输入文件
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 掩码代码块，避免提取代码块中的 $$
    masked_for_search = mask_fenced_code_blocks_keep_len(content)

    # 匹配行间公式 $$...$$
    pattern = r'\$\$(.*?)\$\$'
    
    formulas: List[Dict] = []
    
    for match in re.finditer(pattern, masked_for_search, re.DOTALL):
        # 从原始内容中提取公式（保持原样）
        formula_content = content[match.start()+2:match.end()-2].strip()
        
        if not formula_content:
            continue
        
        # 计算位置信息
        start_pos = match.start()
        text_before = content[:start_pos]
        line_number = text_before.count('\n') + 1
        last_newline_pos = text_before.rfind('\n')
        column_number = start_pos - last_newline_pos if last_newline_pos != -1 else start_pos + 1
        
        # 构建单条记录
        formula_record = {
            "formula": formula_content,
            "line": line_number,
            "column": column_number,
            "start_pos": start_pos,
            "end_pos": match.end()
        }
        
        formulas.append(formula_record)

    # 生成输出文件路径
    input_stem = Path(input_path).stem
    output_subdir = os.path.join(output_base, input_stem)
    os.makedirs(output_subdir, exist_ok=True)
    output_file = os.path.join(output_subdir, f"{input_stem}.jsonl")
    
    # 写入 JSONL 文件（每行一个 JSON 对象）
    with open(output_file, 'w', encoding='utf-8') as f:
        for formula_record in formulas:
            json_line = json.dumps(formula_record, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"行间公式提取完成！")
    print(f"输入文件: {input_path}")
    print(f"找到 {len(formulas)} 个行间公式")
    print(f"输出文件: {output_file}")
    
    return output_file

if __name__ == "__main__":
    extract_md_to_latex()
