# Material RAG 项目详细介绍

## 1. 项目定位与目标

`material-rag-service` 是一个面向论文/技术文档的检索增强问答（RAG）系统，重点解决以下问题：

- 将 PDF 文档自动转换为可检索的知识库（文档、分块、向量、公式、物理量）。
- 支持多种检索策略（向量、BM25、混合、自动策略）。
- 在问答返回中结构化输出论文信息：题目、作者、年份、摘要、公式、结论、文章链接。
- 提供 API 与 Streamlit 可视化界面，支持开发联调和业务展示。

项目适用于材料、航空、工程等需要“从论文中提炼公式与结论并可追溯原文”的场景。

---

## 2. 技术栈

- Web/API
  - FastAPI
  - Uvicorn
  - Pydantic
- 数据库
  - PostgreSQL
  - pgvector
  - SQLAlchemy（异步）+ asyncpg
- 检索相关
  - Embedding 服务（外部 HTTP）
  - Reranker 服务（外部 HTTP）
  - BM25（PostgreSQL 全文检索）
  - RRF（Reciprocal Rank Fusion）混合召回
- 文档处理
  - MinerU（PDF->Markdown）
  - 自定义 Markdown 分块器
  - 离线公式/物理量抽取链路
- 前端
  - Streamlit

---

## 3. 项目结构（核心）

```text
material-rag-service/
├─ main.py                                  # FastAPI 主入口
├─ ui_app.py                                # Streamlit 可视化入口
├─ batch_upload.py                          # 本地 PDF 批量入库脚本
├─ requirements.txt
├─ src/
│  ├─ clients/
│  │  ├─ config.py                          # 全局配置（DB/API/服务地址等）
│  │  ├─ embedding_client.py                # Embedding HTTP 客户端
│  │  ├─ reranker_client.py                 # Reranker HTTP 客户端
│  │  ├─ llm_client.py                      # LLM 客户端
│  │  └─ MinerUParser.py                    # MinerU 相关
│  ├─ rag/
│  │  ├─ router.py                          # /api/v1 路由
│  │  ├─ schemas.py                         # 请求/响应模型
│  │  ├─ search_service.py                  # 检索服务（embedding/bm25/hybrid/adaptive）
│  │  ├─ rag_service.py                     # RAG 聚合与结构化输出
│  │  ├─ database.py                        # Async SQLAlchemy 引擎/会话
│  │  └─ models.py                          # ORM 模型（documents/chunks/embeddings/...）
│  ├─ chunk/
│  │  └─ chunk_processor.py                 # Markdown 分块与 chunks 入库
│  ├─ embedding/
│  │  └─ embedding_service.py               # embeddings 入库服务
│  ├─ pdf/
│  │  ├─ pdf_to_md.py                       # PDF->Markdown
│  │  ├─ metadata_extractor.py              # 元数据提取
│  │  ├─ document_writer.py                 # documents 入库
│  │  ├─ doc_id_generator.py                # doc_id 生成
│  │  └─ ...
│  ├─ extract/
│  │  ├─ offline_extract.py                 # 离线抽取总流程
│  │  ├─ md_to_latex.py                     # Markdown->LaTeX JSONL
│  │  ├─ devide_and_fix.py                  # 公式清洗修复
│  │  └─ llm_4_extract.py                   # 公式/物理量结构化输出
│  └─ service/
│     └─ pdf_chunks_embedding_extract_service.py  # 统一处理服务
└─ test/                                    # 各阶段测试脚本
```

---

## 4. 核心能力总览

### 4.1 检索能力

`src/rag/search_service.py` 实现四种模式：

- `embedding`：纯向量召回。
- `bm25`：基于 PostgreSQL 全文检索与模糊匹配。
- `hybrid`：向量 + BM25，使用 RRF 融合后可选 rerank。
- `adaptive`：按 query 特征自动选择检索路径。

关键特性：

- 支持过滤条件（doc_id/source_type/年份区间等）。
- 记录检索日志（耗时、结果数、策略）。
- `TEST_MODE=true` 时可生成伪向量用于联调。

### 4.2 RAG 聚合与结构化输出

`src/rag/rag_service.py` 负责：

- 组装论文元数据（title/authors/year/abstract/link）。
- 拉取公式（`formulas` 表）并附加 LaTeX 字段。
- 多策略提取结论（正文命中 > section 命中 > 文末回退）。
- 返回结构化 `papers` 列表，前端可直接渲染。

结论策略（当前实现）偏向“原文优先”：

- 优先从明显结论段提取并去噪。
- 若仍无结果，最终回退到原文文末片段（`tail_raw_original`）。

### 4.3 文档入库 Pipeline

统一接口在 `main.py` 的：

- `POST /api/v1/pdf-chunks-embedding-extract`

处理链路包括：

1. 下载/读取 PDF。
2. PDF -> Markdown。
3. 元数据提取并写入 `documents`。
4. Markdown 分块并写入 `chunks`。
5. 生成向量并写入 `embeddings`。
6. 可选：对航空等 source_type 触发离线公式抽取，写入 `formulas` 与 `physical_quantities`。

---

## 5. 主要 API 说明

路由入口：`src/rag/router.py`，统一前缀 `/api/v1`。

### 5.1 健康检查

- `GET /api/v1/health`
- 返回数据库连通状态。

### 5.2 检索接口

- `POST /api/v1/search`
- 入参模型：`SearchRequest`
- 返回：`SearchResponse`（含耗时与策略信息）

### 5.3 RAG 问答（流式）

- `POST /api/v1/chat`
- SSE 输出 token 片段，适合打字机效果。

### 5.4 RAG 问答（同步）

- `POST /api/v1/chat/sync`
- 一次性返回结构化结果，典型字段：
  - `answer`
  - `papers`
  - `sources`
  - `formulas`
  - `conclusions`
  - `article_links`

### 5.5 全链路文档处理

- `POST /api/v1/pdf-chunks-embedding-extract`
- 适合运维/批处理/自动化任务调度。

---

## 6. 数据模型（关键表）

定义位置：`src/rag/models.py`。

### 6.1 `documents`

核心字段：

- `doc_id`（业务唯一标识）
- `title/authors/keywords`
- `source_url/source_type`
- `publish_year/abstract/doc_type`

### 6.2 `chunks`

核心字段：

- `chunk_id`（唯一）
- `doc_id`（关联 documents）
- `content`
- `page`
- `section_path`（JSONB）
- `chunk_index`

### 6.3 `embeddings`

核心字段：

- `chunk_id`（唯一）
- `doc_id`
- `embedding`（`Vector(1024)`）

### 6.4 `search_logs`

用于观测检索性能和策略效果。

### 6.5 抽取相关表

在离线抽取流程中写入：

- `formulas`
- `physical_quantities`

用于支撑公式和物理量可视化。

---

## 7. 配置与环境变量

位置：`src/clients/config.py`。

重点变量：

- 数据库
  - `DB_HOST`
  - `DB_PORT`
  - `DB_USER`
  - `DB_PASSWORD`
  - `DB_NAME`
- 检索服务
  - `EMBEDDING_BASE_URL`
  - `RERANKER_BASE_URL`
- API
  - `API_HOST`（默认 `0.0.0.0`）
  - `API_PORT`（代码默认 `1469`）
- 文件服务
  - `PDF_SERVE_DIR`
  - `ARTICLE_LINK_BASE_URL`

建议：

- 使用 `.env` 管理敏感配置。
- 生产环境不要使用默认数据库密码。
- UI 默认 URL 可能与 API 默认端口不一致，部署时统一。

---

## 8. 启动与运行

### 8.1 安装依赖

```bash
pip install -r requirements.txt
```

### 8.2 启动 API

```bash
python main.py
```

### 8.3 启动前端

```bash
streamlit run ui_app.py
```

### 8.4 常用联调请求

```bash
curl -sS -X POST "http://127.0.0.1:1688/api/v1/chat/sync" \
  -H "Content-Type: application/json" \
  -d '{"query":"展弦比","top_k":10,"search_mode":"hybrid","include_formulas":true,"include_conclusions":true,"include_article_links":true}'
```

---

## 9. 前端可视化设计说明（`ui_app.py`）

前端目标：将后端 `papers` 结构化结果按“论文”维度展示。

展示模块：

- 题目、作者、时间、摘要、文章链接。
- 公式：优先后端 LaTeX，失败时尝试表达式自动转 LaTeX，再退回原始代码块。
- 结论：展示原文提取内容；文末回退数据会有来源提示。

公式渲染策略（当前）：

1. 读取 `expr_latex`，做 `$$`/`\(\)`/`\[\]` 去壳。
2. 如果是纯文本包装，尝试把 Python/Sympy 风格表达式转为 LaTeX。
3. 校验大括号平衡，再调用 `st.latex`。
4. 渲染失败时回退 `st.code(expr)`，保证至少“可见”。

---

## 10. 端到端数据流（E2E）

### 10.1 入库阶段

`PDF -> Markdown -> metadata/documents -> chunks -> embeddings -> (optional) formulas/quantities`

### 10.2 问答阶段

`query -> SearchService召回 -> RAGService组装论文/公式/结论 -> /chat/sync 返回 papers -> UI 渲染`

---

## 11. 典型使用场景

- 场景 1：论文知识库构建
  - 批量扫描本地 PDF，完成全量入库。
- 场景 2：面向公式与结论的检索问答
  - 用户输入主题关键词（如“展弦比”），系统返回相关论文公式和结论。
- 场景 3：外部演示
  - API + Streamlit 组合用于业务汇报与技术评审。

---

## 12. 常见问题与排查

### 12.1 返回结果为空

检查：

- `documents/chunks/embeddings` 是否有数据。
- 查询词是否过于冷门，尝试 `search_mode=bm25`。
- `min_score` 是否设置过高。

### 12.2 公式显示不稳定

检查：

- 后端 `expr_latex` 是否为可渲染 LaTeX。
- 原始 `expr` 是否是复杂 Python 风格表达式。
- 前端是否已走到回退渲染路径（`st.code`）。

建议：

- 在后端逐步提升 `_to_latex_expr` 的规则覆盖率。
- 保留前端“失败即回退代码”的策略，避免空白。

### 12.3 结论看起来不像结论

当前系统是“原文优先 + 多级回退”，部分文档可能没有显式 `Conclusion` 段。

建议：

- 结合 `section_path` 与文末位置进行人工抽检。
- 在 UI 中展示 `source` 字段帮助甄别来源。

### 12.4 外部机器访问

确保：

- API 用 `0.0.0.0` 监听。
- 防火墙放行对应端口（API 与 Streamlit）。
- 使用服务器局域网 IP（如 `10.x.x.x`）进行访问。

---

## 13. 安全与工程建议

- 配置安全
  - 将数据库口令、API Key 放入 `.env`，避免硬编码。
- 数据治理
  - 建立文档版本控制与重复入库策略。
- 可观测性
  - 利用 `search_logs` 分析 query 命中率、耗时、策略质量。
- 稳定性
  - 对外部依赖（Embedding/Reranker/MinerU）增加熔断与重试。

---

## 14. 后续可扩展方向

- 增强公式解析
  - 引入 AST/符号解析器，把 Python 数学表达式稳定转 LaTeX。
- 多语言检索
  - 对中文/英文分别优化 BM25 分词与 rerank 提示。
- 评测体系
  - 构建离线评测集，按 Recall/MRR/答案结构完整性评估。
- 权限与多租户
  - 按项目/部门隔离文档与索引。
- 部署升级
  - Docker Compose 一键部署 API + DB + 前端 + 外部模型服务。

---

## 15. 

1. 先启动 PostgreSQL + pgvector，并确认连接可用。
2. 启动 API（`main.py`）并检查 `/api/v1/health`。
3. 用 `batch_upload.py` 或 pipeline 接口导入一批 PDF。
4. 调用 `/api/v1/chat/sync` 验证 `papers` 返回结构。
5. 启动 `ui_app.py` 观察公式与结论可视化效果。

这个顺序可以最快打通“数据入库 -> 检索问答 -> 前端展示”的完整闭环。

---

## 16. 文件与入口索引

- API 主入口：`main.py`
- 路由：`src/rag/router.py`
- 检索服务：`src/rag/search_service.py`
- RAG 聚合：`src/rag/rag_service.py`
- 前端：`ui_app.py`（只为了可视化展示的）
- 全链路处理服务：`src/service/pdf_chunks_embedding_extract_service.py`
- 批处理脚本：`batch_upload.py`

---
