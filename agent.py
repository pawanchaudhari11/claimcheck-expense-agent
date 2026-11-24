"""The agent loop.

The language model decides which tool to call next. This file only runs the tool
the model picked, sends the result back, and logs everything. There is no fixed
sequence of steps in the code: the order of tool calls comes from the model.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from llm import GroqClient, LLMError, ToolCall
from tools import TOOL_SCHEMAS, ToolError, build_registry, load_claim

MAX_TURNS = 15
MAX_NUDGES = 3
LOG_DIR = Path(__file__).parent / "logs"

SYSTEM_PROMPT = """\
You are an expense-reimbursement review assistant for a small fictional company.
You review one employee expense claim and RECOMMEND one decision for a human
finance reviewer: APPROVE, FLAG_FOR_REVIEW, REJECT, or NEEDS_INFO. You never
approve payments yourself.

Work from evidence, not assumptions:
- Look up the policy rules you need with lookup_policy (including decision_rules).
  Do not rely on limits you have not read from the policy.
- Check receipt completeness with check_receipts.
- Do all arithmetic with the calculate tool, never in your head: the claim total
  (sum of every line-item amount), the sum of meal receipts for EACH date, and every
  comparison against a limit.
- If a required receipt field is missing, ask the employee with ask_employee. If
  they cannot answer, recommend NEEDS_INFO. Never invent or guess missing values.
  An answer from the employee without a receipt does not complete the receipt.
- Apply the policy's decision_rules in their stated order.

Choose each next step based on what the previous tool returned. When you have
enough evidence, call submit_recommendation exactly once, with reasons that cite
the tool results (amounts, limits, differences)."""


# --------------------------------------------------------------------------- #
# Tool-argument validation (the API does not enforce the JSON schema)
# --------------------------------------------------------------------------- #
_SCHEMAS = {s["name"]: s["input_schema"] for s in TOOL_SCHEMAS}
_TYPES = {"string": str, "number": (int, float), "array": list, "object": dict}


def validate_args(name: str, args: dict) -> str | None:
    """Return an error message if args don't match the tool's schema, else None."""
    schema = _SCHEMAS[name]
    props = schema["properties"]
    missing = [k for k in schema["required"] if k not in args]
    if missing:
        return f"Missing required argument(s): {missing}"
    extra = [k for k in args if k not in props]
    if extra:
        return f"Unknown argument(s): {extra}. Allowed: {list(props)}"
    for k, v in args.items():
        spec = props[k]
        if not isinstance(v, _TYPES[spec["type"]]) or isinstance(v, bool):
            return f"Argument '{k}' must be of type {spec['type']}."
        if "enum" in spec and v not in spec["enum"]:
            return f"Argument '{k}' must be one of {spec['enum']}."
        if spec["type"] == "array" and not all(isinstance(i, str) for i in v):
            return f"Argument '{k}' must be a list of strings."
    return None


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
@dataclass
class RunLog:
    claim_id: str
    tag: str = ""
    events: list[dict] = field(default_factory=list)
    path: Path | None = None

    def __post_init__(self) -> None:
        LOG_DIR.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = LOG_DIR / f"{self.claim_id}_{self.tag}_{stamp}.jsonl"

    def add(self, kind: str, **data: Any) -> None:
        event = {"t": datetime.now().isoformat(timespec="seconds"), "event": kind, **data}
        self.events.append(event)
        with self.path.open("a") as f:
            f.write(json.dumps(event, default=str) + "\n")


def _short(obj: Any, limit: int = 500) -> str:
    s = obj if isinstance(obj, str) else json.dumps(obj, default=str)
    s = " ".join(s.split())
    return s if len(s) <= limit else s[:limit] + " ...(full text in log)"


@dataclass
class ReviewResult:
    claim_id: str
    recommendation: dict | None
    tool_calls: list[dict]
    log_path: Path
    stop_reason: str
    turns: int


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
def review_claim(claim_id: str, model: str | None = None, interactive: bool = False,
                 verbose: bool = True, llm=None) -> ReviewResult:
    """Run the agent on one claim. `llm` can be replaced by a fake in tests."""
    llm = llm or GroqClient(model)
    registry = build_registry(interactive)
    log = RunLog(claim_id, tag=llm.name)
    say = print if verbose else (lambda *a, **k: None)

    claim = load_claim(claim_id)
    user_msg = ("Please review this expense claim and recommend a decision.\n\n"
                f"```json\n{json.dumps(claim, indent=2)}\n```")
    llm.add_user(user_msg)
    log.add("start", claim_id=claim_id, api=llm.name, model=llm.model,
            interactive=interactive, system_prompt=SYSTEM_PROMPT, user_message=user_msg)
    say(f"\n=== Reviewing {claim_id} ({claim.get('purpose', '')}) "
        f"with {llm.name}:{llm.model} ===")

    tool_calls: list[dict] = []
    recommendation: dict | None = None
    stop_reason, nudges, turn = "max_turns", 0, 0

    for turn in range(1, MAX_TURNS + 1):
        t0 = time.time()
        try:
            t = llm.call(SYSTEM_PROMPT, TOOL_SCHEMAS)
        except LLMError as e:
            log.add("llm_error", turn=turn, error=str(e))
            say(f"\n[turn {turn}] MODEL ERROR: {e}")
            stop_reason = "llm_error"
            break
        llm.add_assistant(t)
        log.add("model_response", turn=turn, stop=t.stop, model=t.model,
                seconds=round(time.time() - t0, 1), usage=t.usage,
                reasoning=t.reasoning, text=t.text,
                tool_calls=[{"id": c.id, "name": c.name, "input": c.input}
                            for c in t.tool_calls])
        if t.reasoning.strip():
            say(f"\n[turn {turn}] model reasoning: {_short(t.reasoning, 350)}")
        if t.text.strip():
            say(f"[turn {turn}] model says: {_short(t.text, 400)}")

        if t.stop == "max_tokens":
            stop_reason = t.stop
            say(f"[turn {turn}] stopped: {t.stop}")
            break

        if not t.tool_calls:
            # Model ended its turn without a final decision -> remind it (bounded).
            if nudges >= MAX_NUDGES:
                stop_reason = "no_recommendation"
                break
            nudges += 1
            reminder = ("You have not submitted a decision yet. Continue the review using "
                        "the tools; when finished, call submit_recommendation.")
            llm.add_user(reminder)
            log.add("nudge", turn=turn, message=reminder)
            say(f"[turn {turn}] (no tool call - reminding the model to use tools / submit)")
            continue

        results: list[tuple[ToolCall, str, bool]] = []
        for c in t.tool_calls:
            say(f"\n[turn {turn}] TOOL CALL   -> {c.name}({json.dumps(c.input)})")
            output, is_error = _run_tool(c, registry)
            if c.name == "submit_recommendation" and not is_error:
                recommendation = dict(c.input)
            say(f"[turn {turn}] TOOL RESULT {'(ERROR) ' if is_error else ''}<- {_short(output)}")
            call = {"turn": turn, "tool": c.name, "input": c.input,
                    "output": output, "is_error": is_error}
            tool_calls.append(call)
            log.add("tool_call", **call)
            results.append((c, json.dumps(output), is_error))
        llm.add_tool_results(results)

        if recommendation is not None:
            stop_reason = "submitted"
            break

    log.add("end", stop_reason=stop_reason, turns=turn, recommendation=recommendation)
    _print_recommendation(say, claim_id, recommendation, stop_reason)
    say(f"  full log: {log.path}")
    return ReviewResult(claim_id, recommendation, tool_calls, log.path, stop_reason, turn)


def _run_tool(c: ToolCall, registry: dict) -> tuple[dict, bool]:
    if c.name not in _SCHEMAS:
        return {"error": f"Unknown tool '{c.name}'. Available: {list(_SCHEMAS)}"}, True
    err = validate_args(c.name, c.input)
    if err:
        return {"error": err}, True
    if c.name == "submit_recommendation":
        return {"status": "recorded"}, False
    try:
        return registry[c.name](**c.input), False
    except ToolError as e:
        return {"error": str(e)}, True


def _print_recommendation(say, claim_id, rec, stop_reason) -> None:
    if not rec:
        say(f"\n=== No recommendation submitted for {claim_id} (stop: {stop_reason}) ===")
        return
    say(f"\n=== RECOMMENDATION for {claim_id}: {rec['decision']} ===")
    say(f"  claim total: ${rec['claim_total']:.2f}")
    for r in rec["reasons"]:
        say(f"  - {r}")
    for v in rec["policy_violations"]:
        say(f"  ! violation: {v}")
    for m in rec["missing_information"]:
        say(f"  ? missing: {m}")
    say("  (Recommendation only - a human reviewer makes the final approval.)")
