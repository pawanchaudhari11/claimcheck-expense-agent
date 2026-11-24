"""Tools the expense-review agent can call.

Each tool is a plain, deterministic Python function. The language model never
does the work itself: it decides WHICH tool to call and WHAT to pass, and the
results come from this code. That makes every decision checkable.

Tools
-----
lookup_policy      read rules from data/policy.json
calculate          safe arithmetic (sums, differences, comparisons)
check_receipts     inspect a claim's receipts for missing required fields
ask_employee       action: ask the human for missing details instead of guessing
submit_recommendation  action: final recommendation (ends the run)
"""

from __future__ import annotations

import ast
import json
import operator
from pathlib import Path
from typing import Any, Callable

DATA_DIR = Path(__file__).parent / "data"
POLICY_PATH = DATA_DIR / "policy.json"
CLAIMS_DIR = DATA_DIR / "claims"

DECISIONS = ["APPROVE", "FLAG_FOR_REVIEW", "REJECT", "NEEDS_INFO"]


class ToolError(Exception):
    """Raised for bad tool input; reported back to the model as is_error=True."""


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_policy() -> dict:
    with POLICY_PATH.open() as f:
        return json.load(f)


def load_claim(claim_id: str) -> dict:
    path = CLAIMS_DIR / f"{claim_id}.json"
    if not path.is_file():
        raise ToolError(f"No claim file found for claim_id '{claim_id}'.")
    with path.open() as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Tool 1: policy lookup
# --------------------------------------------------------------------------- #
def lookup_policy(section: str) -> dict:
    """Return one section of the policy JSON."""
    policy = load_policy()
    if section == "all":
        return policy
    if section in policy["categories"]:
        return {"category": section, "rules": policy["categories"][section],
                "currency": policy["currency"]}
    if section in ("receipts", "decision_rules"):
        return {section: policy[section]}
    raise ToolError(
        f"Unknown policy section '{section}'. Valid: "
        f"{sorted(policy['categories']) + ['receipts', 'decision_rules', 'all']}"
    )


# --------------------------------------------------------------------------- #
# Tool 2: calculator (safe AST evaluation - no eval())
# --------------------------------------------------------------------------- #
_BIN_OPS: dict[type, Callable] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_CMP_OPS: dict[type, Callable] = {
    ast.Gt: operator.gt, ast.GtE: operator.ge, ast.Lt: operator.lt,
    ast.LtE: operator.le, ast.Eq: operator.eq, ast.NotEq: operator.ne,
}
_FUNCS: dict[str, Callable] = {"sum": lambda *a: sum(a), "round": round,
                               "max": max, "min": min, "abs": abs}


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _eval_node(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _CMP_OPS:
        return _CMP_OPS[type(node.ops[0])](_eval_node(node.left),
                                           _eval_node(node.comparators[0]))
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS and not node.keywords):
        return _FUNCS[node.func.id](*[_eval_node(a) for a in node.args])
    raise ToolError(f"Unsupported expression element: {ast.dump(node)[:80]}")


def calculate(expression: str) -> dict:
    """Evaluate an arithmetic expression, e.g. '14.75 + 48.30 + 39.95 > 75'."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ToolError(f"Could not parse expression: {e.msg}") from e
    try:
        value = _eval_node(tree)
    except ZeroDivisionError as e:
        raise ToolError("Division by zero.") from e
    if isinstance(value, float):
        value = round(value, 2)
    return {"expression": expression, "result": value}


# --------------------------------------------------------------------------- #
# Tool 3: receipt checker
# --------------------------------------------------------------------------- #
def check_receipts(claim_id: str) -> dict:
    """Check every receipt in a claim for the policy's required fields."""
    claim = load_claim(claim_id)
    required = load_policy()["receipts"]["required_fields"]
    items = []
    for li in claim["line_items"]:
        r = li.get("receipt") or {}
        missing = [f for f in required if r.get(f) in (None, "")]
        items.append({"item_id": li["item_id"], "merchant": r.get("merchant"),
                      "category": r.get("category"), "amount": r.get("amount"),
                      "date": r.get("date"), "missing_fields": missing,
                      "complete": not missing})
    incomplete = [i for i in items if not i["complete"]]
    return {"claim_id": claim_id, "required_fields": required,
            "receipt_count": len(items), "all_complete": not incomplete,
            "incomplete_items": [{"item_id": i["item_id"], "merchant": i["merchant"],
                                  "missing_fields": i["missing_fields"]} for i in incomplete],
            "items": items}


# --------------------------------------------------------------------------- #
# Action: ask the employee (human-in-the-loop)
# --------------------------------------------------------------------------- #
def make_ask_employee(interactive: bool) -> Callable[[str], dict]:
    def ask_employee(question: str) -> dict:
        if not interactive:
            return {"answered": False,
                    "response": "No response: this is a non-interactive run, so the "
                                "employee cannot answer now. Do not guess the value."}
        print(f"\n  [agent asks the employee] {question}")
        answer = input("  your answer (blank = no answer): ").strip()
        if not answer:
            return {"answered": False, "response": "Employee did not provide an answer."}
        return {"answered": True, "response": answer,
                "note": "Self-reported by the employee; not backed by a receipt."}
    return ask_employee


# --------------------------------------------------------------------------- #
# Tool schemas sent to the model
# --------------------------------------------------------------------------- #
TOOL_SCHEMAS = [
    {
        "name": "lookup_policy",
        "description": "Read company travel-policy rules from the policy JSON file. "
                       "Use it to get spending limits for a category, the required "
                       "receipt fields, or the decision rules. Never assume a limit "
                       "you have not looked up.",
        "input_schema": {
            "type": "object",
            "properties": {"section": {"type": "string", "enum": [
                "meals", "lodging", "ground_transport", "airfare",
                "receipts", "decision_rules", "all"]}},
            "required": ["section"], "additionalProperties": False,
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression exactly. Supports + - * / , "
                       "parentheses, comparisons (> >= < <= == !=), and sum(), round(), "
                       "max(), min(), abs(). Use this for EVERY total, daily sum, or "
                       "limit comparison instead of doing math yourself. "
                       "Example: '14.75 + 48.30 + 39.95 > 75'.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"], "additionalProperties": False,
        },
    },
    {
        "name": "check_receipts",
        "description": "Inspect every receipt in a claim and report any that are "
                       "missing required fields (date, merchant, amount, category).",
        "input_schema": {
            "type": "object",
            "properties": {"claim_id": {"type": "string"}},
            "required": ["claim_id"], "additionalProperties": False,
        },
    },
    {
        "name": "ask_employee",
        "description": "Ask the employee who submitted the claim for missing "
                       "information. Use this instead of guessing when a receipt is "
                       "missing a required field.",
        "input_schema": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"], "additionalProperties": False,
        },
    },
    {
        "name": "submit_recommendation",
        "description": "Submit the final recommendation for the claim. This ends the "
                       "review. It is a recommendation for a human reviewer, not a "
                       "payment approval. Only call it after gathering evidence.",
        "input_schema": {
            "type": "object",
            "properties": {
                "decision": {"type": "string", "enum": DECISIONS},
                "claim_total": {"type": "number",
                                "description": "Sum of all line-item amounts, from the calculate tool."},
                "reasons": {"type": "array", "items": {"type": "string"},
                            "description": "Evidence-based reasons, citing tool results."},
                "policy_violations": {"type": "array", "items": {"type": "string"},
                                      "description": "Each limit exceeded and by how much; empty if none."},
                "missing_information": {"type": "array", "items": {"type": "string"},
                                        "description": "What the employee must still provide; empty if none."},
            },
            "required": ["decision", "claim_total", "reasons",
                         "policy_violations", "missing_information"],
            "additionalProperties": False,
        },
    },
]


def build_registry(interactive: bool) -> dict[str, Callable[..., dict]]:
    return {
        "lookup_policy": lookup_policy,
        "calculate": calculate,
        "check_receipts": check_receipts,
        "ask_employee": make_ask_employee(interactive),
    }
