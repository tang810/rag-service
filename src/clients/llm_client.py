#!/usr/bin/env python3
"""Unified LLM client for reuse across scripts.

Features:
- Env-based defaults (LLM_API_KEY, LLM_API_BASE, LLM_MODEL)
- Sync completion_text plus async wrapper
- Simple retry with backoff
"""

from __future__ import annotations
import json
import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests


@dataclass
class LLMConfig:
    api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", "ximu-llm-api-key"))
    base_url: str = field(default_factory=lambda: os.getenv("LLM_API_BASE", "http://www.science42.vip:40200/v1/chat/completions"))
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "SE_V0.0"))
    system_prompt: str = "You are a helpful assistant."
    timeout: int = 180
    max_retries: int = 3
    retry_backoff: float = 2.0  # seconds multiplier


class LLMClient:
    def __init__(self, cfg: Optional[LLMConfig] = None):
        self.cfg = cfg or LLMConfig()
        self.headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }

    def _build_messages(self, user_prompt: str, system_prompt: Optional[str]) -> List[Dict[str, str]]:
        sys_prompt = system_prompt or self.cfg.system_prompt
        return [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _request(self, payload: Dict[str, Any], timeout: Optional[int]) -> Dict[str, Any]:
        delay = 1.0
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                resp = requests.post(
                    self.cfg.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=timeout or self.cfg.timeout,
                )
                if resp.status_code == 200:
                    return resp.json()
                last_err = f"status={resp.status_code} body={resp.text[:512]}"
            except Exception as exc:  # network or JSON error
                last_err = str(exc)

            if attempt >= self.cfg.max_retries:
                raise RuntimeError(f"LLM request failed after {attempt} attempts: {last_err}")

            time.sleep(delay)
            delay *= self.cfg.retry_backoff

        raise RuntimeError("Unreachable retry loop")

    def completion_text(
        self,
        *,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> str:
        payload = {
            "model": model or self.cfg.model,
            "messages": self._build_messages(user_prompt, system_prompt),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = self._request(payload, timeout)
        try:
            return data["choices"][0]["message"]["content"]
        except Exception as exc:
            raise RuntimeError(f"Invalid LLM response: {data}") from exc

    async def acompletion_text(self, **kwargs: Any) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.completion_text(**kwargs))

    def completion_stream(
        self,
        *,
        user_prompt: str,  # 用户输入的提示文本
        temperature: float = 0.2,  # 控制生成文本的随机性，默认0.2
        max_tokens: int = 4096,  # 生成的最大token数，默认4096
        model: Optional[str] = None,  # 可选的模型名称，不指定则使用配置中的默认模型
        system_prompt: Optional[str] = None,  # 可选的系统提示文本
        timeout: Optional[int] = None,  # 请求超时时间，不指定则使用配置中的默认超时时间
    ):
        """逐 token 生成器，与业务无关，纯底层能力"""
        payload = {
            "model": model or self.cfg.model,
            "messages": self._build_messages(user_prompt, system_prompt),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        resp = requests.post(
            self.cfg.base_url,
            headers=self.headers,
            json=payload,
            timeout=timeout or self.cfg.timeout,
            stream=True,
        )
        resp.raise_for_status()

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data_str = line[len("data: "):]
            if data_str.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content
            except (json.JSONDecodeError, KeyError, IndexError):
                continue


__all__ = ["LLMClient", "LLMConfig"]
