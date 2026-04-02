#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Formula Fixer Module
从 JSONL 读取 LaTeX 公式，调用 LLM 进行处理：
  1) Split merged/concatenated formulas
  2) Normalize formula text (remove redundant styling/spacing)
  3) Detect and complete truncated formulas
写入处理结果到输出 JSONL
"""

import asyncio
import json
import re
from pathlib import Path
from typing import List, Dict, Optional

from src.clients.llm_client import LLMClient, LLMConfig

# ==================== Prompt 模板 ====================
PROMPT_TEMPLATE_JSONL = """
你是一位专业的公式处理专家，请分析文件中包含的所有公式，并完成以下三个功能：

功能1️⃣：公式拆分
- 遍历所有公式，如果公式之间是并列的，需要拆分为单独的公式。
- 示例：公式 "F = ma \\text{ 和 } P = W / t" 应拆分为 "F = ma" 和 "P = W / t"。

功能2️⃣：去除多余修饰
- 去掉公式中多余修饰，例如字符的样式、字符间的空格、位置控制相关的信息。
- 示例：公式 "W _ { \\scriptscriptstyle { e m p t y } }" 应转换为 "W_empty"。

功能3️⃣：公式补全（仅基于原始 Markdown 上下文）
- 仅依据每条公式的 md_context 字段内容判断是否残缺，并进行补全；不得凭空臆测或仅基于一般常识推断。
- 示例：若 md_context 中出现 "L = W0 * g"，则残缺 "L =" 可补全为 "L = W0 * g"。

【输入文件】: {file_name}
【输入公式与上下文列表】:
{context}

【输出格式要求】请返回合法 JSON 数组，每个对象包含：
- idx: 原始编号（对应输入条目）
- original: 原始公式字符串
- fixed: 处理后的公式数组（拆分并列、清理修饰后的每条公式）
- is_incomplete: 布尔值，是否判定为残缺（依据 md_context）
- completed: 若 is_incomplete 为 true，给出基于 md_context 的补全版本数组；否则为空数组
- notes: 简要说明（例如去修饰、拆分、补全依据）

请开始处理文件中的公式，并仅输出上述 JSON 数组：
"""

# ---- LLM client (shared) ----
llm_cfg_default = LLMConfig(system_prompt="你是一位精通 LaTeX 公式清洗与补全的专家。")

# ---- IO helpers ----
def read_jsonl(path: Path) -> List[Dict]:
    """读取 JSONL 文件"""
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def write_jsonl(path: Path, rows: List[Dict]) -> None:
    """写入 JSONL 文件"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

# ---- Prompt construction ----
def build_prompt(formulas: List[Dict], file_name: str) -> str:
    """构建 prompt"""
    context = json.dumps(formulas, ensure_ascii=False, indent=2)
    prompt = PROMPT_TEMPLATE_JSONL
    prompt = prompt.replace("{file_name}", file_name)
    prompt = prompt.replace("{context}", context)
    return prompt

# ---- JSON parse helpers ----
def strip_code_fences(text: str) -> str:
    """移除代码围栏"""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```\s*$", "", text)
    return text.strip()

def extract_json_array(text: str) -> List[Dict]:
    """从文本中提取 JSON 数组"""
    s = strip_code_fences(text)
    start = s.find('[')
    end = s.rfind(']')
    if start == -1 or end == -1 or end <= start:
        raise json.JSONDecodeError("No JSON array found", s, 0)
    candidate = s[start:end+1]
    return json.loads(candidate)

def chunk_list(lst: list, n: int):
    """分块列表"""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# ---- Markdown context helpers ----
def read_md_lines(md_path: Path) -> List[str]:
    """读取 Markdown 文件行"""
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312"]
    for enc in encodings:
        try:
            return md_path.read_text(encoding=enc).splitlines()
        except UnicodeDecodeError:
            continue
    return md_path.read_text(encoding="utf-8", errors="ignore").splitlines()

def build_md_context(item: Dict, md_lines: List[str], window: int = 3) -> str:
    """构建 Markdown 上下文"""
    line_no = item.get("line")
    if isinstance(line_no, int) and 1 <= line_no <= len(md_lines):
        start = max(1, line_no - window)
        end = min(len(md_lines), line_no + window)
        snippet = md_lines[start-1:end]
        return "\n".join(snippet)
    
    text = "\n".join(md_lines)
    start_pos = item.get("start_pos")
    end_pos = item.get("end_pos")
    if isinstance(start_pos, int) and isinstance(end_pos, int) and 0 <= start_pos < end_pos <= len(text):
        w = 120
        s = max(0, start_pos - w)
        e = min(len(text), end_pos + w)
        return text[s:e]
    return ""

# ---- Main function ----
async def process_and_fix_formulas(
    input_jsonl: str = None,
    output_dir: str = None,
    md_file: str = None,
    api_key: str = "ximu-llm-api-key",
    base_url: str = "http://www.science42.vip:40200/v1/chat/completions",
    batch_size: int = 20,
    max_retries: int = 2,
    llm_cfg: Optional[LLMConfig] = None,
) -> str:
    """
    处理并修复公式
    
    参数：
        input_jsonl: 输入 JSONL 文件路径
        output_dir: 输出目录
        md_file: Markdown 文件路径（用于提取上下文）
        api_key: LLM API 密钥
        base_url: LLM API 基础 URL
        batch_size: 批处理大小
        max_retries: 最大重试次数
    
    返回：
        输出文件路径
    """
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    _DATA_DIR = _PROJECT_ROOT / "data"
    input_path = Path(input_jsonl or (_DATA_DIR / "latex" / "tilt_rotor" / "tilt_rotor.jsonl"))
    output_root = Path(output_dir or (_DATA_DIR / "fixed_latex"))
    md_path = Path(md_file or (_DATA_DIR / "md" / "tilt_rotor" / "tilt_rotor.md"))

    # Configure LLM
    llm_config = llm_cfg or LLMConfig(
        api_key=api_key,
        base_url=base_url,
        system_prompt="你是一位精通 LaTeX 公式清洗与补全的专家。",
    )
    llm = LLMClient(llm_config)

    # Read input
    items = read_jsonl(input_path)
    if not items:
        raise SystemExit("输入为空")

    # Read markdown for context
    md_lines = read_md_lines(md_path)

    # Assign global idx for stable mapping
    for i, it in enumerate(items, start=1):
        if "idx" not in it:
            it["idx"] = i

    output_rows: List[Dict] = []

    for batch_no, batch in enumerate(chunk_list(items, batch_size), start=1):
        attempt = 0
        while True:
            attempt += 1
            # Enrich batch items with md_context
            batch_with_ctx = []
            for it in batch:
                ctx = build_md_context(it, md_lines)
                obj = dict(it)
                obj["md_context"] = ctx
                batch_with_ctx.append(obj)
            
            prompt = build_prompt(batch_with_ctx, input_path.name)
            response_text = await llm.acompletion_text(
                user_prompt=prompt,
                system_prompt=llm_config.system_prompt,
                max_tokens=4096,
                model=llm_config.model,
                temperature=0.2,
                timeout=120,
            )
            try:
                parsed_batch = extract_json_array(response_text)
                break
            except Exception as e:
                raw_path = output_root / input_path.stem / f"{input_path.name}_batch{batch_no}_raw.txt"
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_text(response_text, encoding="utf-8")
                print(f"⚠️ 批次 {batch_no} 第 {attempt} 次解析失败: {e}")
                if attempt > max_retries:
                    print(f"❌ 批次 {batch_no} 多次失败，跳过该批。")
                    parsed_batch = []
                    break

        # Map parsed batch results
        for entry in parsed_batch:
            idx = entry.get("idx")
            original = entry.get("original", "")
            fixed_list = entry.get("fixed", []) or []
            is_incomplete = bool(entry.get("is_incomplete", False))
            completed_list = entry.get("completed", []) or []
            notes = entry.get("notes", "")
            if not isinstance(fixed_list, list):
                fixed_list = [str(fixed_list)]
            if not isinstance(completed_list, list):
                completed_list = [str(completed_list)] if completed_list else []
            output_rows.append({
                "idx": idx,
                "original": original,
                "fixed": fixed_list,
                "is_incomplete": is_incomplete,
                "completed": completed_list,
                "notes": notes,
            })

    # Prepare output path
    out_dir = output_root / input_path.stem
    out_file = out_dir / f"{input_path.stem}_fixed.jsonl"
    write_jsonl(out_file, output_rows)
    print(f"✅ 已写出: {out_file}")
    
    return str(out_file)

if __name__ == "__main__":
    asyncio.run(process_and_fix_formulas())
