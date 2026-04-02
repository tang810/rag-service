from __future__ import annotations

import os
import re
from typing import Any, Dict, List

import requests
import streamlit as st

API_BASE = os.getenv("UI_API_BASE", "http://127.0.0.1:1688")


def _normalize_latex_for_streamlit(expr_latex: str) -> str:
    txt = str(expr_latex or "").strip()
    if not txt:
        return ""
    if txt.startswith("$$") and txt.endswith("$$") and len(txt) >= 4:
        txt = txt[2:-2].strip()
    if txt.startswith(r"\(") and txt.endswith(r"\)") and len(txt) >= 4:
        txt = txt[2:-2].strip()
    if txt.startswith(r"\[") and txt.endswith(r"\]") and len(txt) >= 4:
        txt = txt[2:-2].strip()
    return txt



def _expr_to_latex_fallback(expr: str) -> str:
    s = str(expr or "").strip()
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s)
    s = s.replace("**", "^")
    s = re.sub(r"([A-Za-z]+)_([A-Za-z0-9]+)", r"\1_{\2}", s)
    s = re.sub(r"\^(-?\d+(?:\.\d+)?)", r"^{\1}", s)
    s = s.replace("Omega", r"\Omega").replace("omega", r"\omega")
    s = s.replace("rho", r"\rho").replace("eta", r"\eta").replace("chi", r"\chi")
    s = s.replace("pi", r"\pi")
    s = s.replace("*", r" \cdot ")
    return s


def _to_math_symbol(token: str) -> str:
    t = str(token or "").strip()
    if not t:
        return t
    t = re.sub(r"\s+", "", t)
    t = t.replace("Omega", r"\Omega").replace("omega", r"\omega")
    t = t.replace("rho", r"\rho").replace("eta", r"\eta").replace("chi", r"\chi")
    t = t.replace("pi", r"\pi")
    # t_min_n -> t_{min,n}
    if re.match(r"^[A-Za-z]+(?:_[A-Za-z0-9]+)+$", t):
        parts = t.split("_")
        base = parts[0]
        sub = ",".join(parts[1:])
        t = rf"{base}_{{{sub}}}"
    else:
        t = re.sub(r"([A-Za-z]+)_([A-Za-z0-9]+)", r"\1_{\2}", t)
    return t


def _pythonish_expr_to_latex(expr: str) -> str:
    s = str(expr or "").strip()
    if not s:
        return ""

    s = re.sub(r"\s+", " ", s)

    # sum([body for i in range(1, m+1)]) -> \sum_{i=1}^{m}(body)
    sum_comp_pat = re.compile(
        r"sum\(\s*\[\s*(?P<body>.+?)\s+for\s+(?P<var>[A-Za-z_]\w*)\s+in\s+range\(\s*1\s*,\s*(?P<upper>[^)]+)\)\s*\]\s*\)"
    )

    def _sum_comp_repl(m: re.Match[str]) -> str:
        body = m.group("body").strip()
        var = m.group("var").strip()
        upper = m.group("upper").strip()
        upper = re.sub(r"\+\s*1\b", "", upper)
        upper = _to_math_symbol(upper)

        body = body.replace("**", "^")
        body = re.sub(r"([A-Za-z]+)_([A-Za-z0-9]+)", r"\1_{\2}", body)
        body = re.sub(r"\^(-?\d+(?:\.\d+)?)", r"^{\1}", body)
        body = body.replace("*", r" \cdot ")
        return rf"\sum_{{{var}=1}}^{{{upper}}}\left({body}\right)"

    for _ in range(6):
        new_s = re.sub(sum_comp_pat, _sum_comp_repl, s)
        if new_s == s:
            break
        s = new_s

    # min(x for i in range(1, N+1)) -> \min_{i=1}^{N}(x)
    min_comp_pat = re.compile(
        r"min\(\s*(?P<body>.+?)\s+for\s+(?P<var>[A-Za-z_]\w*)\s+in\s+range\(\s*1\s*,\s*(?P<upper>[^)]+)\)\s*\)"
    )

    def _min_comp_repl(m: re.Match[str]) -> str:
        body = m.group("body").strip()
        var = m.group("var").strip()
        upper = m.group("upper").strip()
        upper = re.sub(r"\+\s*1\b", "", upper)
        upper = _to_math_symbol(upper)
        body = _to_math_symbol(body)
        return rf"\min_{{{var}=1}}^{{{upper}}}\left({body}\right)"

    s = re.sub(min_comp_pat, _min_comp_repl, s)

    # sum(expr, (i,1,N)) -> \sum_{i=1}^{N}(expr)
    sum_tuple_pat = re.compile(
        r"sum\(\s*(?P<body>.+)\s*,\s*\(\s*(?P<var>[A-Za-z_]\w*)\s*,\s*1\s*,\s*(?P<upper>[^)]+)\)\s*\)"
    )

    def _sum_tuple_repl(m: re.Match[str]) -> str:
        body = m.group("body").strip()
        var = m.group("var").strip()
        upper = _to_math_symbol(m.group("upper").strip())
        body = body.replace("**", "^")
        body = re.sub(r"\^(-?\d+(?:\.\d+)?)", r"^{\1}", body)
        # convert variables like t_min_n
        body = re.sub(r"[A-Za-z]+(?:_[A-Za-z0-9]+)+", lambda mm: _to_math_symbol(mm.group(0)), body)
        body = re.sub(r"([A-Za-z]+)_([A-Za-z0-9]+)", r"\1_{\2}", body)
        body = body.replace("*", r" \cdot ")
        return rf"\sum_{{{var}=1}}^{{{upper}}}\left({body}\right)"

    for _ in range(10):
        new_s = re.sub(sum_tuple_pat, _sum_tuple_repl, s)
        if new_s == s:
            break
        s = new_s

    # abs(x) -> \left|x\right|
    for _ in range(6):
        new_s = re.sub(r"abs\(\s*([^()]+?)\s*\)", r"\\left|\1\\right|", s)
        if new_s == s:
            break
        s = new_s

    # sqrt(x) -> \sqrt{x}
    s = re.sub(r"sqrt\((.+)\)", r"\\sqrt{\1}", s)

    # Basic symbol/operator cleanup.
    s = s.replace("**", "^")
    s = re.sub(r"[A-Za-z]+(?:_[A-Za-z0-9]+)+", lambda m: _to_math_symbol(m.group(0)), s)
    s = re.sub(r"([A-Za-z]+)_([A-Za-z0-9]+)", r"\1_{\2}", s)
    s = re.sub(r"\^(-?\d+(?:\.\d+)?)", r"^{\1}", s)
    s = s.replace("Omega", r"\Omega").replace("omega", r"\omega")
    s = s.replace("rho", r"\rho").replace("eta", r"\eta").replace("chi", r"\chi")
    s = s.replace("pi", r"\pi")
    s = s.replace("*", r" \cdot ")
    return s


def _looks_plain_text_latex(expr_latex: str) -> bool:
    txt = str(expr_latex or "").strip()
    if not txt:
        return True
    # Backend fallback often wraps raw text as \text{...}, which is readable but not math-visual.
    if txt.startswith(r"\text{") or txt.startswith(r"\mathrm{text"):
        return True
    if r"\textbackslash{}" in txt:
        return True
    return False


def _has_balanced_braces(text: str) -> bool:
    depth = 0
    for ch in str(text or ""):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _is_latex_render_safe(latex_body: str) -> bool:
    txt = str(latex_body or "").strip()
    if not txt:
        return False
    if not _has_balanced_braces(txt):
        return False
    # Sympy-like function calls frequently break KaTeX in this pipeline.
    risky_calls = (r"sqrt\(", r"sum\(", r"min\(", r"max\(", r"abs\(", r"forlinrange\(")
    if any(re.search(p, txt) for p in risky_calls):
        return False
    # Tuple-index style often fails in latex parser.
    if re.search(r"\([a-zA-Z]\s*,\s*\d+\s*,\s*[^)]+\)", txt):
        return False
    return True

def _clean_conclusion_start(text: str) -> str:
    txt = str(text or "").strip()
    txt = re.sub(r"^[\s\-\.\,;:]+", "", txt)
    txt = re.sub(r"^\d+\.\s*", "", txt)
    txt = re.sub(r"^[a-zA-Z]\s+(?=[a-z])", "", txt)
    return txt


def _sentence_case_start(text: str) -> str:
    txt = str(text or "").strip()
    if not txt:
        return txt
    return txt[0].upper() + txt[1:] if txt[0].islower() else txt


st.set_page_config(page_title="Material RAG", layout="wide")
st.title("Material RAG 可视化")
st.caption("按论文展示：题目 / 作者 / 时间 / 摘要 / 公式(LaTeX) / 结论 / 文章链接")

with st.sidebar:
    st.subheader("检索参数")
    api_base_input = st.text_input("API Base", API_BASE)
    top_k = st.slider("top_k", min_value=1, max_value=20, value=10)
    search_mode = st.selectbox("search_mode", ["hybrid", "embedding", "bm25", "adaptive"], index=2)
    include_formulas = st.checkbox("include_formulas", value=True)
    include_conclusions = st.checkbox("include_conclusions", value=True)
    include_links = st.checkbox("include_article_links", value=True)

query = st.text_area("问题", value="展弦比", height=110)

if st.button("检索", type="primary"):
    payload = {
        "query": query,
        "top_k": top_k,
        "search_mode": search_mode,
        "include_formulas": include_formulas,
        "include_conclusions": include_conclusions,
        "include_article_links": include_links,
    }
    url = f"{api_base_input.rstrip('/')}/api/v1/chat/sync"

    try:
        with st.spinner("请求中..."):
            resp = requests.post(url, json=payload, timeout=300)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        st.error(f"请求失败: {exc}")
        st.stop()

    st.success("请求成功")

    papers: List[Dict[str, Any]] = data.get("papers", [])

    st.subheader("论文结果")
    if not papers:
        st.info("没有检索到论文（请确认后端已返回 papers 字段）")
    else:
        for idx, p in enumerate(papers, 1):
            title = p.get("title") or "未知"
            authors = p.get("authors") or "未知"
            year = p.get("publish_year") if p.get("publish_year") is not None else "未知"
            abstract = p.get("abstract") or "暂无"
            link = p.get("article_link") or ""

            with st.container(border=True):
                st.markdown(f"### 论文 {idx}: {title}")
                st.write(f"**作者**: {authors}")
                st.write(f"**时间**: {year}")
                st.write(f"**摘要**: {abstract}")

                if link:
                    st.markdown(f"**文章链接**: [{link}]({link})")
                else:
                    st.write("**文章链接**: 暂无")

                st.markdown("**公式**")
                doc_formulas = p.get("formulas", [])
                if not doc_formulas:
                    st.write("暂无")
                else:
                    for i, f in enumerate(doc_formulas, 1):
                        name = f.get("name_zh") or f.get("id") or "未命名公式"
                        expr = str(f.get("expr") or "")
                        expr_latex = str(f.get("expr_latex") or "")
                        st.markdown(f"{i}. {name}")

                        latex_body = _normalize_latex_for_streamlit(expr_latex)
                        if not latex_body or _looks_plain_text_latex(latex_body):
                            latex_body = _pythonish_expr_to_latex(expr)
                        if not latex_body or _looks_plain_text_latex(latex_body):
                            latex_body = _expr_to_latex_fallback(expr)
                        if latex_body and _is_latex_render_safe(latex_body):
                            try:
                                st.latex(latex_body)
                            except Exception:
                                st.code(expr)
                        else:
                            st.code(expr)

                        with st.expander("查看原始表达式"):
                            st.code(expr)

                st.markdown("**结论**")
                doc_conclusions = p.get("conclusions", [])
                if not doc_conclusions:
                    st.write("暂无")
                else:
                    has_tail_raw = any(str(c.get("source") or "") == "tail_raw_original" for c in doc_conclusions)
                    merged = " ".join(_clean_conclusion_start(str(c.get("content") or "")) for c in doc_conclusions)
                    merged = re.sub(r"\s+", " ", merged).strip()
                    merged = _sentence_case_start(merged)
                    if has_tail_raw:
                        st.caption("当前结论为原文文末片段（未识别到明确结论标题）。")
                    st.write(merged or "暂无")

    with st.expander("查看原始 answer 文本"):
        st.text(data.get("answer", ""))

    with st.expander("查看原始 JSON"):
        st.json(data)
