"""LLM adapters + a lenient JSON helper.

* ``EchoLLM`` — returns the fused evidence so the search pipeline runs end-to-end
  with no API key; build stages that expect JSON simply get nothing back and
  no-op (so the deterministic CSV path needs no LLM at all).
* ``CallableLLM`` — wrap any ``f(prompt, system) -> str`` (your OpenAI / Anthropic /
  vLLM call) and you have a drop-in LLM.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional


class EchoLLM:
    """No-op LLM: echoes the evidence. Lets the pipeline run with zero credentials."""

    def generate(self, prompt: str, *, system: str = "", timeout: Optional[float] = None) -> str:
        return prompt


class CallableLLM:
    """Adapt ``f(prompt, system=...)`` (or ``f(prompt)``) into the LLM protocol."""

    def __init__(self, fn: Callable[..., str]):
        self._fn = fn

    def generate(self, prompt: str, *, system: str = "", timeout: Optional[float] = None) -> str:
        try:
            return self._fn(prompt, system=system)
        except TypeError:
            return self._fn(prompt)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def invoke_json(llm: Any, system: str, user: str) -> dict:
    """Call ``llm.generate`` and parse a JSON object out of the reply, leniently.

    Returns ``{}`` on any failure (no LLM, non-JSON echo, parse error) so build
    stages degrade gracefully to their rule-based behavior."""
    if llm is None:
        return {}
    try:
        raw = llm.generate(user, system=system)
    except Exception:
        return {}
    if not raw or not isinstance(raw, str):
        return {}
    # strip code fences if present
    m = _FENCE.search(raw)
    if m:
        raw = m.group(1)
    raw = raw.strip()
    # direct parse, else first balanced {...}
    for candidate in (raw, _first_object(raw)):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            continue
    return {}


def _first_object(s: str) -> str:
    start = s.find("{")
    if start < 0:
        return ""
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return ""
