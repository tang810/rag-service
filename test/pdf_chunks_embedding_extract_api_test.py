from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any


def _http_json(method: str, url: str, payload: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    body = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(url=url, method=method.upper(), headers=headers, data=body)

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            status = resp.getcode()
            data = resp.read().decode("utf-8")
            return status, json.loads(data)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"raw": raw}
        return exc.code, parsed


def _assert_response_shape(result: dict[str, Any]) -> None:
    required_top_keys = {
        "input",
        "doc_id",
        "generated_doc_id",
        "source_type",
        "pdf_workflow",
        "chunks_workflow",
        "embedding_workflow",
        "extract_workflow",
        "summary",
    }
    missing = sorted(required_top_keys - set(result.keys()))
    if missing:
        raise AssertionError(f"响应缺少关键字段: {missing}")


def main() -> int:
    parser = argparse.ArgumentParser(description="API test for /api/v1/pdf-chunks-embedding-extract")
    parser.add_argument("--base-url", default="http://127.0.0.1:1400", help="API base url")
    parser.add_argument("--filename", default="轻型倾转旋翼机总体设计与参数优化.pdf", help="PDF filename")
    parser.add_argument(
        "--preview-url",
        default="http://36.103.203.113:2300/alpha/pdf/2026/03/05/bf89d0e8a84543859072cdf4f38344e0.pdf",
        help="PDF preview url",
    )
    parser.add_argument("--module", default="aircraft", help="Source module")
    parser.add_argument("--skip-pdf-db", action="store_true", help="Skip writing documents table")
    parser.add_argument("--skip-chunks-db", action="store_true", help="Skip writing chunks table")
    parser.add_argument("--skip-embeddings-db", action="store_true", help="Skip writing embeddings table")
    parser.add_argument("--skip-extract", action="store_true", help="Skip extract stage")
    parser.add_argument("--embedding-batch-size", type=int, default=32, help="Embedding batch size")
    parser.add_argument("--embedding-limit", type=int, default=1000, help="Embedding fetch limit")
    parser.add_argument("--check-health", action="store_true", help="Check /api/v1/health before running")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")

    if args.check_health:
        health_url = f"{base_url}/api/v1/health"
        health_status, health_resp = _http_json("GET", health_url)
        print(f"[health] status={health_status}")
        print(json.dumps(health_resp, ensure_ascii=False, indent=2))
        if health_status != 200:
            print("健康检查失败，终止测试。", file=sys.stderr)
            return 2

    endpoint = f"{base_url}/api/v1/pdf-chunks-embedding-extract"
    payload = {
        "filename": args.filename,
        "preview_url": args.preview_url,
        "module": args.module,
        "write_pdf_db": not args.skip_pdf_db,
        "write_chunks_db": not args.skip_chunks_db,
        "write_embeddings_db": not args.skip_embeddings_db,
        "run_extract_if_aircraft": not args.skip_extract,
        "embedding_batch_size": args.embedding_batch_size,
        "embedding_limit": args.embedding_limit,
    }

    print(f"[request] POST {endpoint}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    status, response = _http_json("POST", endpoint, payload=payload)
    print(f"[response] status={status}")
    print(json.dumps(response, ensure_ascii=False, indent=2))

    if status != 200:
        print("接口测试失败：HTTP 非 200。", file=sys.stderr)
        return 1

    try:
        _assert_response_shape(response)
    except AssertionError as exc:
        print(f"接口测试失败：{exc}", file=sys.stderr)
        return 1

    print("接口测试通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())