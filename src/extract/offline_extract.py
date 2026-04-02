#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Formula Extraction Pipeline - 本地文件处理
整合四个处理步骤：
    1. PDF to Markdown (pdf_to_md)
    2. Markdown to LaTeX JSONL (md_to_latex)
    3. Formula Division and Fixing (devide_and_fix)
    4. Formula to YAML (llm_4_extract)

并支持按输入类型自动选择起点：
    - PDF: 从步骤 1 开始
    - Markdown: 从步骤 2 开始
    - LaTeX JSONL: 从步骤 3 开始
    - Fixed JSONL: 从步骤 4 开始

本文件仅包含本地文件处理逻辑，FastAPI 接口请使用 main.py
"""

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import asyncpg
import yaml

from src.clients.config import db_config
from src.clients.llm_client import LLMConfig

# 导入四个模块的主函数
from src.pdf.pdf_to_md import extract_pdf_to_md_async
try:
    from src.extract.md_to_latex import extract_md_to_latex
    from src.extract.devide_and_fix import process_and_fix_formulas
    from src.extract.llm_4_extract import extract_formulas_to_yaml
except ImportError:
    # 兼容直接在 src/extract 目录执行脚本
    from md_to_latex import extract_md_to_latex
    from devide_and_fix import process_and_fix_formulas
    from llm_4_extract import extract_formulas_to_yaml

# ==================== 配置 ====================
class Config:
    """全局配置（基于仓库根目录）"""
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    _DATA_DIR = _PROJECT_ROOT / "data"
    # PDF 输入目录
    PDF_INPUT_DIR = str((_PROJECT_ROOT / "src" / "mypdf").resolve())
    # Markdown 输出目录
    MD_OUTPUT_DIR = str((_DATA_DIR / "md").resolve())
    # Markdown 输入目录（与 chunk 流程一致）
    MARKDOWN_INPUT_DIR = str((_DATA_DIR / "markdown").resolve())
    # LaTeX JSONL 输出目录
    LATEX_OUTPUT_DIR = str((_DATA_DIR / "latex").resolve())
    # Fixed LaTeX 输出目录
    FIXED_LATEX_OUTPUT_DIR = str((_DATA_DIR / "fixed_latex").resolve())
    # 提取结果 YAML 输出目录（保存到 data 下）
    EXTRACT_YAML_OUTPUT_DIR = str((_DATA_DIR / "extract_yaml").resolve())
    RUNTIME_DIR = str((_DATA_DIR / "runtime").resolve())
    # MinerU API 地址
    MINERU_API_BASE = "http://www.science42.vip:40093"
    # LLM API 配置
    LLM_API_KEY = "ximu-llm-api-key"
    LLM_API_BASE = "http://www.science42.vip:40200/v1/chat/completions"
    # 上传文件临时目录（放在 data/uploads）
    UPLOAD_DIR = str((_DATA_DIR / "uploads").resolve())
    # 数据库 DSN
    DB_DSN = db_config.url.replace("postgresql+asyncpg://", "postgresql://", 1)


# 统一的 LLM 配置实例
LLM_CFG = LLMConfig(api_key=Config.LLM_API_KEY, base_url=Config.LLM_API_BASE)


def _yaml_output_dir_for_stem(source_stem: str) -> Path:
    base = Path(Config.EXTRACT_YAML_OUTPUT_DIR)
    out = base / source_stem
    out.mkdir(parents=True, exist_ok=True)
    return out


def _load_yaml_items(path: Path, root_key: str) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        return []
    items = obj.get(root_key) or []
    if not isinstance(items, list):
        return []
    return [it for it in items if isinstance(it, dict)]


def _detect_jsonl_type(path: Path) -> str:
    """
    判断 JSONL 类型：
      - latex_jsonl: 原始公式 JSONL（包含 formula 字段）
      - fixed_jsonl: 清洗后 JSONL（包含 fixed 字段）
    """
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict) and "fixed" in obj:
                return "fixed_jsonl"
            if isinstance(obj, dict) and "formula" in obj:
                return "latex_jsonl"
            raise ValueError(f"无法识别 JSONL 字段: {path}")
    raise ValueError(f"JSONL 文件为空: {path}")


async def process_local_input(input_path: str, doc_id: Optional[str] = None) -> Dict[str, Any]:
    """按输入类型自动选择起点并执行提取流程。"""
    src = Path(input_path).resolve()
    if not src.exists():
        return {"success": False, "error": f"文件不存在: {src}"}

    print("\n" + "=" * 60)
    print(f"开始处理输入: {src}")
    print("=" * 60)

    md_path: Optional[str] = None
    jsonl_path: Optional[str] = None
    fixed_jsonl_path: Optional[str] = None

    try:
        suffix = src.suffix.lower()

        if suffix == ".pdf":
            print("\n[步骤 1/4] PDF 转 Markdown...")
            md_path = await extract_pdf_to_md_async(
                input_path=str(src),
                output_dir=Config.MD_OUTPUT_DIR,
                docker_url=Config.MINERU_API_BASE,
            )
            if not md_path:
                return {"success": False, "error": "PDF 转 Markdown 失败"}
            print(f"✅ Markdown 文件: {md_path}")

            print("\n[步骤 2/4] Markdown 转 LaTeX JSONL...")
            jsonl_path = extract_md_to_latex(input_md=str(md_path), output_dir=Config.LATEX_OUTPUT_DIR)
            print(f"✅ JSONL 文件: {jsonl_path}")

            print("\n[步骤 3/4] 公式清洗与修复...")
            fixed_jsonl_path = await process_and_fix_formulas(
                input_jsonl=jsonl_path,
                output_dir=Config.FIXED_LATEX_OUTPUT_DIR,
                md_file=str(md_path),
                api_key=Config.LLM_API_KEY,
                base_url=Config.LLM_API_BASE,
                llm_cfg=LLM_CFG,
            )
            print(f"✅ 清洗后的 JSONL: {fixed_jsonl_path}")

        elif suffix == ".md":
            md_path = str(src)

            print("\n[步骤 2/4] Markdown 转 LaTeX JSONL...")
            jsonl_path = extract_md_to_latex(input_md=md_path, output_dir=Config.LATEX_OUTPUT_DIR)
            print(f"✅ JSONL 文件: {jsonl_path}")

            print("\n[步骤 3/4] 公式清洗与修复...")
            fixed_jsonl_path = await process_and_fix_formulas(
                input_jsonl=jsonl_path,
                output_dir=Config.FIXED_LATEX_OUTPUT_DIR,
                md_file=md_path,
                api_key=Config.LLM_API_KEY,
                base_url=Config.LLM_API_BASE,
                llm_cfg=LLM_CFG,
            )
            print(f"✅ 清洗后的 JSONL: {fixed_jsonl_path}")

        elif suffix == ".jsonl":
            jsonl_kind = _detect_jsonl_type(src)
            source_stem = src.stem.replace("_fixed", "")

            if jsonl_kind == "latex_jsonl":
                jsonl_path = str(src)

                # 尝试定位同名 Markdown 作为上下文，找不到则让下游使用默认路径。
                possible_md = Path(Config.MD_OUTPUT_DIR) / source_stem / f"{source_stem}.md"
                md_arg = str(possible_md) if possible_md.exists() else None

                print("\n[步骤 3/4] 公式清洗与修复...")
                fixed_jsonl_path = await process_and_fix_formulas(
                    input_jsonl=jsonl_path,
                    output_dir=Config.FIXED_LATEX_OUTPUT_DIR,
                    md_file=md_arg,
                    api_key=Config.LLM_API_KEY,
                    base_url=Config.LLM_API_BASE,
                    llm_cfg=LLM_CFG,
                )
                print(f"✅ 清洗后的 JSONL: {fixed_jsonl_path}")
            else:
                fixed_jsonl_path = str(src)
                print(f"\n跳过步骤 1-3，直接使用 Fixed JSONL: {fixed_jsonl_path}")

        else:
            return {
                "success": False,
                "error": f"不支持的输入类型: {suffix}。仅支持 .pdf / .md / .jsonl",
            }

        source_stem = Path(fixed_jsonl_path).stem.replace("_fixed", "")
        formulas_items: List[Dict[str, Any]] = []
        quantity_items: List[Dict[str, Any]] = []
        formulas_yaml_path: Optional[str] = None
        quantities_yaml_path: Optional[str] = None
        write_stats: Dict[str, int] = {"formulas_written": 0, "quantities_written": 0}

        if doc_id:
            print("\n[步骤 4/4] 生成 YAML 并写入数据库...")
            yaml_output_dir = _yaml_output_dir_for_stem(source_stem)
            await extract_formulas_to_yaml(
                input_jsonl=fixed_jsonl_path,
                output_dir=str(yaml_output_dir),
                api_key=Config.LLM_API_KEY,
                base_url=Config.LLM_API_BASE,
                llm_cfg=LLM_CFG,
            )

            formulas_yaml = yaml_output_dir / "formulas.yaml"
            quantities_yaml = yaml_output_dir / "quantities.yaml"
            formulas_yaml_path = str(formulas_yaml)
            quantities_yaml_path = str(quantities_yaml)

            formulas_items = _load_yaml_items(formulas_yaml, "formulas")
            quantity_items = _load_yaml_items(quantities_yaml, "quantities")
            write_stats = await _upsert_extraction_to_db(
                doc_id=doc_id,
                formulas_items=formulas_items,
                quantity_items=quantity_items,
            )
            print(
                f"✅ 已写库: formulas={write_stats['formulas_written']}, "
                f"quantities={write_stats['quantities_written']}"
            )
        else:
            print("\n跳过步骤 4：未提供 doc_id，不执行数据库写入。")

        print("\n" + "=" * 60)
        print("✅ 处理完成！")
        print("=" * 60)

        return {
            "success": True,
            "input_path": str(src),
            "doc_id": doc_id,
            "md_path": md_path,
            "jsonl_path": jsonl_path,
            "fixed_jsonl_path": fixed_jsonl_path,
            "formulas_yaml_path": formulas_yaml_path,
            "quantities_yaml_path": quantities_yaml_path,
            "formulas_count": len(formulas_items),
            "quantities_count": len(quantity_items),
            **write_stats,
        }

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        return {"success": False, "error": str(e), "input_path": str(src)}


def _normalize_filename_stem(filename: str) -> str:
    stem = Path(filename).stem
    return stem.strip()


def _find_markdown_by_filename(filename: str) -> Path:
    """根据输入 filename 在 data/markdown 下查找对应 markdown 文件。"""
    stem = _normalize_filename_stem(filename)
    md_root = Path(Config.MARKDOWN_INPUT_DIR)
    if not md_root.exists():
        raise FileNotFoundError(f"Markdown 根目录不存在: {md_root}")

    exact_path = md_root / stem / f"{stem}.md"
    if exact_path.exists():
        return exact_path

    candidates = sorted([p for p in md_root.rglob("*.md") if p.stem == stem])
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # 优先 {stem}/{stem}.md，其次最短路径
        preferred = [p for p in candidates if p.parent.name == stem]
        if preferred:
            return preferred[0]
        candidates.sort(key=lambda x: len(str(x)))
        return candidates[0]

    raise FileNotFoundError(f"在 {md_root} 下未找到与 filename 匹配的 markdown: {filename}")


def _stable_row_id(prefix: str, doc_id: str, source_id: str, idx: int) -> str:
    raw = f"{prefix}|{doc_id}|{source_id}|{idx}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


async def _upsert_extraction_to_db(
    doc_id: str,
    formulas_items: List[Dict[str, Any]],
    quantity_items: List[Dict[str, Any]],
) -> Dict[str, int]:
    conn = await asyncpg.connect(Config.DB_DSN)
    formulas_written = 0
    quantities_written = 0

    try:
        async with conn.transaction():
            for i, it in enumerate(formulas_items):
                source_id = str(it.get("id") or f"f_{i}")
                row_id = _stable_row_id("formula", doc_id, source_id, i)
                extractid = it.get("extractid")
                if isinstance(extractid, list):
                    extractid_str = ",".join([str(x) for x in extractid])
                elif extractid is None:
                    extractid_str = None
                else:
                    extractid_str = str(extractid)

                await conn.execute(
                    """
                    INSERT INTO formulas (
                        id,
                        doc_id,
                        name_zh,
                        expr,
                        extractid,
                        category,
                        page
                    ) VALUES ($1,$2,$3,$4,$5,$6,$7)
                    ON CONFLICT (id) DO UPDATE SET
                        doc_id = EXCLUDED.doc_id,
                        name_zh = EXCLUDED.name_zh,
                        expr = EXCLUDED.expr,
                        extractid = EXCLUDED.extractid,
                        category = EXCLUDED.category,
                        page = EXCLUDED.page
                    """,
                    row_id,
                    doc_id,
                    str(it.get("name_zh") or source_id),
                    str(it.get("expr") or ""),
                    extractid_str,
                    source_id,
                    None,
                )
                formulas_written += 1

            for i, it in enumerate(quantity_items):
                source_id = str(it.get("id") or f"q_{i}")
                row_id = _stable_row_id("quantity", doc_id, source_id, i)

                await conn.execute(
                    """
                    INSERT INTO physical_quantities (
                        id,
                        doc_id,
                        symbol,
                        symbol_latex,
                        name_zh,
                        unit,
                        page
                    ) VALUES ($1,$2,$3,$4,$5,$6,$7)
                    ON CONFLICT (id) DO UPDATE SET
                        doc_id = EXCLUDED.doc_id,
                        symbol = EXCLUDED.symbol,
                        symbol_latex = EXCLUDED.symbol_latex,
                        name_zh = EXCLUDED.name_zh,
                        unit = EXCLUDED.unit,
                        page = EXCLUDED.page
                    """,
                    row_id,
                    doc_id,
                    str(it.get("symbol") or source_id),
                    str(it.get("symbol_latex") or ""),
                    str(it.get("name_zh") or ""),
                    str(it.get("unit") or ""),
                    None,
                )
                quantities_written += 1
    finally:
        await conn.close()

    return {
        "formulas_written": formulas_written,
        "quantities_written": quantities_written,
    }


async def run_offline_extract_for_doc(doc_id: str, filename: str) -> Dict[str, Any]:
    """
    从 data/markdown 按 filename 定位 markdown，执行提取链路并将结果写入数据库。

    流程：md_to_latex -> devide_and_fix -> llm_4_extract(yaml落盘) -> 读取yaml -> DB
    """
    md_path = _find_markdown_by_filename(filename)
    source_stem = md_path.stem

    jsonl_path = extract_md_to_latex(input_md=str(md_path), output_dir=Config.LATEX_OUTPUT_DIR)
    fixed_jsonl_path = await process_and_fix_formulas(
        input_jsonl=jsonl_path,
        output_dir=Config.FIXED_LATEX_OUTPUT_DIR,
        md_file=str(md_path),
        api_key=Config.LLM_API_KEY,
        base_url=Config.LLM_API_BASE,
        llm_cfg=LLM_CFG,
    )

    yaml_output_dir = _yaml_output_dir_for_stem(source_stem)
    await extract_formulas_to_yaml(
        input_jsonl=fixed_jsonl_path,
        output_dir=str(yaml_output_dir),
        api_key=Config.LLM_API_KEY,
        base_url=Config.LLM_API_BASE,
        llm_cfg=LLM_CFG,
    )

    formulas_yaml = yaml_output_dir / "formulas.yaml"
    quantities_yaml = yaml_output_dir / "quantities.yaml"
    formulas_items = _load_yaml_items(formulas_yaml, "formulas")
    quantity_items = _load_yaml_items(quantities_yaml, "quantities")

    write_stats = await _upsert_extraction_to_db(
        doc_id=doc_id,
        formulas_items=formulas_items,
        quantity_items=quantity_items,
    )

    return {
        "success": True,
        "doc_id": doc_id,
        "filename": filename,
        "markdown_path": str(md_path),
        "jsonl_path": jsonl_path,
        "fixed_jsonl_path": fixed_jsonl_path,
        "formulas_yaml_path": str(formulas_yaml),
        "quantities_yaml_path": str(quantities_yaml),
        "formulas_count": len(formulas_items),
        "quantities_count": len(quantity_items),
        **write_stats,
    }


# ==================== 兼容入口（保留原函数名） ====================
async def process_local_pdf(
    pdf_path: str,
    output_suffix: str = None,
) -> Dict[str, Any]:
    """兼容旧调用，内部转发到新流程。"""
    _ = output_suffix  # 预留参数，保持兼容
    return await process_local_input(pdf_path)

# ==================== 命令行接口 ====================
async def main_cli():
    """命令行入口"""
    import sys
    
    print("\n" + "="*60)
    print("Formula Extraction Pipeline - 本地文件处理")
    print("="*60)
    print("\n使用方法：")
    print("  python src/extract/offline_extract.py <输入文件路径>")
    print("\n注意：")
    print("  支持输入: .pdf / .md / .jsonl")
    print("  FastAPI 接口服务请使用 main.py")
    print("="*60)
    
    if len(sys.argv) < 2:
        print("\n❌ 请指定输入文件路径！")
        print("示例: python src/extract/offline_extract.py /path/to/file.pdf")
        return

    input_path = sys.argv[1]

    if not os.path.exists(input_path):
        print(f"❌ 文件不存在: {input_path}")
        return

    result = await process_local_input(input_path)
    
    # 打印结果
    print("\n📋 处理结果：")
    for key, value in result.items():
        print(f"  {key}: {value}")

# ==================== 示例函数 ====================
async def example_pipeline():
    """示例：完整流程处理"""
    pdf_file = str((Config._DATA_DIR / "raw" / "tilt_rotor" / "tilt_rotor.pdf").resolve())
    
    print("\n示例：处理 tilt_rotor.pdf")
    result = await process_local_pdf(pdf_file)
    
    if result["success"]:
        print("\n📋 处理成功，生成的文件：")
        print(f"  Markdown: {result['md_path']}")
        print(f"  JSONL: {result['jsonl_path']}")
        print(f"  Fixed JSONL: {result['fixed_jsonl_path']}")
        print(f"  Formulas 数量: {result.get('formulas_count', 0)}")
        print(f"  Quantities 数量: {result.get('quantities_count', 0)}")
    else:
        print(f"\n❌ 处理失败: {result['error']}")

if __name__ == "__main__":
    import sys
    
    # 如果有命令行参数，使用 CLI 模式
    if len(sys.argv) > 1:
        asyncio.run(main_cli())
    else:
        # 否则运行示例
        print("运行示例流程...")
        asyncio.run(example_pipeline())
