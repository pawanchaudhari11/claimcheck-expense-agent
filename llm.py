"""Language-model client: Groq's chat-completions API (OpenAI-compatible).

The agent loop (agent.py) uses four methods:

    turn = llm.call(system, tools)   -> Turn (reasoning, text, tool_calls, stop)
    llm.add_user(text)
    llm.add_assistant(turn)
    llm.add_tool_results([(call, output_json, is_error), ...])

The API key is read from the GROQ_API_KEY environment variable, or from a `.env`
file in the project folder (GROQ_API_KEY=...). `.env` is git-ignored.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: Path = Path(__file__).parent / ".env") -> None:
    """Minimal .env loader: KEY=value lines; real environment variables win."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        os.environ.setdefault(key, val.strip().strip('"').strip("'"))


_load_dotenv()

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")


class LLMError(Exception):
    """The API could not be reached or returned something unusable."""


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class Turn:
    reasoning: str = ""
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop: str = "end_turn"          # end_turn | tool_use | max_tokens
    model: str = ""
    usage: dict = field(default_factory=dict)


class GroqClient:
    name = "groq"

    def __init__(self, model: str | None = None, timeout: int = 120, max_retries: int = 4):
        self.key = os.environ.get("GROQ_API_KEY", "").strip()
        if not self.key:
            raise LLMError("GROQ_API_KEY is not set. Put it in a .env file in the project "
                           "folder (GROQ_API_KEY=gsk_...) or export it in your terminal.")
        self.model = model or DEFAULT_MODEL
        self.timeout, self.max_retries = timeout, max_retries
        self.messages: list[dict] = []

    @staticmethod
    def _tools(schemas: list[dict]) -> list[dict]:
        return [{"type": "function", "function": {
            "name": s["name"], "description": s["description"],
            "parameters": s["input_schema"]}} for s in schemas]

    def _post(self, payload: dict) -> dict:
        body = json.dumps(payload).encode()
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(GROQ_URL, data=body, headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.key}",
                "User-Agent": "expense-policy-agent/1.0"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                text = e.read().decode(errors="replace")
                if e.code == 429 and attempt < self.max_retries:   # free-tier rate limit
                    wait = float(e.headers.get("retry-after") or 2 ** (attempt + 1))
                    time.sleep(min(wait, 60))
                    continue
                if e.code >= 500 and attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue
                if e.code == 400 and "tool_use_failed" in text:
                    # The model produced a malformed tool call; let the loop remind it.
                    return {"_tool_use_failed": text[:300]}
                if e.code == 401:
                    raise LLMError("Groq rejected the API key (401). Check GROQ_API_KEY.") from e
                raise LLMError(f"Groq HTTP {e.code}: {text[:300]}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise LLMError(f"Cannot reach Groq ({e}).") from e
        raise LLMError("Groq: retries exhausted.")

    def call(self, system: str, tools: list[dict]) -> Turn:
        data = self._post({
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + self.messages,
            "tools": self._tools(tools),
            "tool_choice": "auto",
            "temperature": 0,
            "max_completion_tokens": 4096,
        })
        if "_tool_use_failed" in data:
            return Turn(text="(model produced a malformed tool call)", model=self.model)

        choice = data["choices"][0]
        msg = choice["message"]
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"__unparseable__": fn.get("arguments")}
            calls.append(ToolCall(id=tc["id"], name=fn.get("name", ""),
                                  input=args if isinstance(args, dict) else {}))
        stop = "tool_use" if calls else (
            "max_tokens" if choice.get("finish_reason") == "length" else "end_turn")
        usage = data.get("usage", {})
        return Turn(reasoning=msg.get("reasoning") or "", text=msg.get("content") or "",
                    tool_calls=calls, stop=stop, model=data.get("model", self.model),
                    usage={"input": usage.get("prompt_tokens"),
                           "output": usage.get("completion_tokens")})

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, turn: Turn) -> None:
        m: dict = {"role": "assistant", "content": turn.text or ""}
        if turn.tool_calls:
            m["tool_calls"] = [{"id": c.id, "type": "function", "function": {
                "name": c.name, "arguments": json.dumps(c.input)}} for c in turn.tool_calls]
        self.messages.append(m)

    def add_tool_results(self, results: list[tuple[ToolCall, str, bool]]) -> None:
        for call, content, _is_err in results:
            self.messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
