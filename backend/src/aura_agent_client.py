"""
Aura Agent REST client for Module 3.
Uses OAuth 2.0 client credentials to obtain a bearer token, then invokes the Agent API.
"""
import asyncio
import base64
import json
import logging
import time
import uuid
from typing import Any

import httpx

from src.config import Config, logger

AURA_LOGGER = logging.getLogger(f"{Config.LOGGER_NAME}.aura")

_token_cache: dict[str, Any] = {"token": None, "expires_at": 0}
_token_lock = asyncio.Lock()


async def _get_access_token(force_refresh: bool = False) -> str:
    """Exchange client credentials for an OAuth access token. Caches token until expiry."""
    async with _token_lock:
        now = time.time()
        if not force_refresh and _token_cache["token"] and now < _token_cache["expires_at"]:
            return _token_cache["token"]

        auth = base64.b64encode(
            f"{Config.AURA_CLIENT_ID}:{Config.AURA_CLIENT_SECRET}".encode()
        ).decode()
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                Config.AURA_TOKEN_URL,
                headers={
                    "Authorization": f"Basic {auth}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data={"grant_type": "client_credentials"},
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()
            token = data.get("access_token")
            if not token:
                raise ValueError("OAuth response missing access_token")
            expires_in = int(data.get("expires_in", 3600))
            _token_cache["token"] = token
            _token_cache["expires_at"] = now + expires_in - 60
        return token


async def invoke_agent(
    query: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    """
    Invoke the Aura Agent with a user query.
    Returns normalized response dict with keys: answer, reasoning, evidence, grounding_ok, raw.
    On failure or missing evidence, grounding_ok is False.
    """
    if not Config.aura_configured():
        raise ValueError("Aura Agent not configured: missing endpoint or credentials")

    sid = session_id or f"{Config.SESSION_PREFIX}{uuid.uuid4().hex[:12]}"
    token = await _get_access_token()
    # Aura Agent API accepts "input" (tutorial) or "query"; send both for compatibility
    payload = {"input": query, "query": query, "session_id": sid}

    max_retries = Config.AURA_AGENT_MAX_RETRIES
    retry_backoff = Config.AURA_AGENT_RETRY_BACKOFF
    resp = None
    last_401 = False
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    Config.AURA_AGENT_ENDPOINT,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=float(Config.AURA_AGENT_TIMEOUT_SECONDS),
                )
            if resp.status_code == 401 and not last_401:
                last_401 = True
                token = await _get_access_token(force_refresh=True)
                continue
            if resp.status_code >= 500 and attempt < max_retries - 1:
                await asyncio.sleep(retry_backoff * (2 ** attempt))
                continue
            break
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_backoff * (2 ** attempt))
            else:
                AURA_LOGGER.warning("Aura Agent failed after %d retries: %s", max_retries, e)
                return {
                    "answer": "",
                    "reasoning": [],
                    "evidence": [],
                    "grounding_ok": False,
                    "raw": None,
                    "error": str(e),
                }

    if resp is None:
        return {
            "answer": "",
            "reasoning": [],
            "evidence": [],
            "grounding_ok": False,
            "raw": None,
            "error": "Request failed",
        }

    if resp.status_code != 200:
        AURA_LOGGER.warning(
            "Aura Agent non-2xx: status=%s body=%s",
            resp.status_code,
            resp.text[:500],
        )
        return {
            "answer": "",
            "reasoning": [],
            "evidence": [],
            "grounding_ok": False,
            "raw": None,
            "error": f"Agent returned {resp.status_code}",
        }

    try:
        data = resp.json()
    except json.JSONDecodeError as e:
        AURA_LOGGER.warning("Aura Agent invalid JSON: %s", e)
        return {
            "answer": "",
            "reasoning": [],
            "evidence": [],
            "grounding_ok": False,
            "raw": None,
            "error": "Invalid JSON response",
        }

    # Normalize: Aura returns content[] with type: thinking|text|cypher_template_tool_use|cypher_template_tool_result
    content = data.get("content") or data.get("messages") or []
    if not isinstance(content, list):
        content = []

    reasoning = []
    evidence = []
    answer_text = ""

    for item in content:
        if not isinstance(item, dict):
            continue
        kind = item.get("type") or item.get("role") or ""
        kind_str = str(kind).lower()

        # Answer text: type "text" with item.text
        if kind in ("text", "assistant", "message"):
            text = item.get("text") or item.get("content") or ""
            if text:
                answer_text = (answer_text + " " + text).strip()

        # Reasoning: type "thinking"
        if kind == "thinking":
            thinking = item.get("thinking") or item.get("text") or ""
            if thinking:
                reasoning.append({"type": "thinking", "content": thinking})

        # Evidence: tool_use, tool_result, or any type containing "tool"
        if "tool" in kind_str or kind in ("tool_result", "tool_use", "evidence"):
            evidence.append(item)

    # Fallback: top-level answer/text/output
    if not answer_text:
        answer_text = data.get("answer") or data.get("text") or data.get("output") or ""
        if isinstance(answer_text, dict):
            answer_text = answer_text.get("text", "") or answer_text.get("content", "") or ""

    # Trust mandate: any answer text or evidence counts as grounded
    has_valid_data = bool(
        (answer_text and answer_text.strip())
        or (evidence and len(evidence) > 0)
        or (reasoning and len(reasoning) > 0)
    )

    if not has_valid_data:
        AURA_LOGGER.debug(
            "Aura response lacked grounded content: keys=%s content_len=%s",
            list(data.keys()) if isinstance(data, dict) else "n/a",
            len(content),
        )

    return {
        "answer": answer_text.strip(),
        "reasoning": reasoning,
        "evidence": evidence,
        "grounding_ok": has_valid_data,
        "raw": data,
    }
