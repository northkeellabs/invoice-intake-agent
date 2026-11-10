"""Model provider factory and structured-output calls.

The model is only ever called with plain messages (text, plus page images for the vision model). It is
never bound to tools, so nothing in a document can make it call anything.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import re
from typing import Any, TypeVar

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from PIL import Image
from pydantic import BaseModel, ValidationError

from .config import Settings

T = TypeVar("T", bound=BaseModel)
MAX_OUTPUT_TOKENS = 2048  # an extraction with line items fits; stops runaway generation
LLM_TIMEOUT_S = 180
VISION_MAX_SIDE = 1400   # longest image side sent to the vision model (A4 at about 120 dpi)


class StructuredOutputError(Exception):
    def __init__(self, message: str, attempts: int):
        super().__init__(message)
        self.attempts = attempts


class OllamaJsonChat:
    """Minimal direct client for Ollama /api/chat, used for the vision model.

    qwen3-vl:8b on Ollama is a thinking build: with think=false and a JSON schema format, Ollama returns the
    JSON in the `thinking` field and leaves `content` empty, and langchain-ollama drops that field. This client
    reads `content`, falling back to `thinking`. Same interface as a LangChain chat model for structured_call.
    """

    def __init__(self, model: str, base_url: str, num_ctx: int = 16384, keep_alive: str | int = "10m"):
        self.model, self.base_url, self.num_ctx, self.keep_alive = model, base_url.rstrip("/"), num_ctx, keep_alive

    @staticmethod
    def _convert(messages: list) -> list[dict]:
        out = []
        for m in messages:
            role = {"system": "system", "human": "user", "ai": "assistant"}.get(m.type, "user")
            if isinstance(m.content, list):
                text = "".join(p.get("text", "") for p in m.content if p.get("type") == "text")
                imgs = [p["image_url"]["url"].split(",", 1)[1] for p in m.content if p.get("type") == "image_url"]
                out.append({"role": role, "content": text, "images": imgs})
            else:
                out.append({"role": role, "content": str(m.content)})
        return out

    async def ainvoke(self, messages: list, format: dict | None = None, **_: Any) -> AIMessage:
        body = {"model": self.model, "messages": self._convert(messages), "stream": False, "think": False,
                "keep_alive": self.keep_alive,
                "options": {"temperature": 0, "num_ctx": self.num_ctx, "num_predict": MAX_OUTPUT_TOKENS}}
        if format:
            body["format"] = format
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT_S, trust_env=False) as c:
            r = await c.post(f"{self.base_url}/api/chat", json=body)
            r.raise_for_status()
            msg = r.json().get("message") or {}
        return AIMessage(content=(msg.get("content") or "").strip() or (msg.get("thinking") or ""))


def make_model(settings: Settings, vision: bool = False):
    p = settings.model_provider
    name = settings.vision_model if vision else settings.model_name
    if p == "ollama" and vision:
        return OllamaJsonChat(name, settings.ollama_base_url)
    if p == "ollama":
        from langchain_ollama import ChatOllama
        # reasoning=False turns off qwen3 "thinking" so outputs are plain JSON and fast.
        return ChatOllama(model=name, base_url=settings.ollama_base_url, temperature=0, reasoning=False,
                          num_ctx=16384 if vision else 8192, num_predict=MAX_OUTPUT_TOKENS, keep_alive="10m",
                          client_kwargs={"timeout": LLM_TIMEOUT_S})
    if p == "anthropic":  # optional, untested without a key
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=name, temperature=0, max_tokens=MAX_OUTPUT_TOKENS)
    if p == "openai":  # optional, untested without a key
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=name, temperature=0)
    raise ValueError(f"unknown MODEL_PROVIDER {p!r} (use ollama, anthropic or openai)")


def unload(settings: Settings, model_name: str) -> None:
    """Ask Ollama to drop a model from GPU memory now (keep_alive 0), so two 8B models are never loaded together."""
    if settings.model_provider != "ollama":
        return
    try:
        with httpx.Client(timeout=60, trust_env=False) as c:
            c.post(f"{settings.ollama_base_url}/api/generate", json={"model": model_name, "keep_alive": 0}).raise_for_status()
    except httpx.HTTPError:
        pass


def image_part(img: Image.Image) -> dict:
    im = img.convert("RGB")
    scale = VISION_MAX_SIDE / max(im.size)
    if scale < 1:
        im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}


def _parse(text: str, schema: type[T]) -> T:
    text = re.sub(r"(?s)<think>.*?</think>", "", text or "").strip()
    m = re.search(r"(?s)```(?:json)?\s*(.*?)```", text)
    if m:
        text = m.group(1).strip()
    if not text.startswith("{"):
        i, j = text.find("{"), text.rfind("}")
        if i >= 0 and j > i:
            text = text[i:j + 1]
    return schema.model_validate(json.loads(text))


def _content(msg: Any) -> str:
    c = msg.content if isinstance(msg, AIMessage) else getattr(msg, "content", msg)
    if isinstance(c, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return str(c)


async def structured_call(model, system: str, user: str, schema: type[T], provider: str = "ollama",
                          images: list[Image.Image] | None = None) -> tuple[T, int]:
    """Call the model for JSON matching `schema`. Retry once on invalid output, then raise."""
    content: Any = user
    if images:
        content = [{"type": "text", "text": user}] + [image_part(i) for i in images]
    messages: list = [SystemMessage(system), HumanMessage(content=content)]
    kwargs: dict[str, Any] = {}
    if provider == "ollama":
        kwargs["format"] = schema.model_json_schema()
    last_err = ""
    for attempt in (1, 2):
        try:
            resp = await asyncio.wait_for(model.ainvoke(messages, **kwargs), timeout=LLM_TIMEOUT_S + 5)
        except Exception as exc:  # timeouts and transport errors count as a failed attempt
            last_err = f"model call failed: {type(exc).__name__}"
            continue
        raw = _content(resp)
        try:
            return _parse(raw, schema), attempt
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_err = f"{type(exc).__name__}: {str(exc)[:300]}"
            messages = messages + [AIMessage(raw[:4000]), HumanMessage(
                "That was not valid JSON for the required schema. Error: " + last_err +
                "\nReply again with only one JSON object that matches the schema exactly.")]
    raise StructuredOutputError(last_err, attempts=2)
