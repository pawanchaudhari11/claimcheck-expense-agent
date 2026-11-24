"""Check the agent's work against evidence, not against what the model claims.

For each claim we compare the agent's recommendation with:
  1. the hand-written answer key (data/answer_key.json), and
  2. a deterministic rule-based reference checker (below) that computes the
     correct decision and totals straight from policy.json + the claim file.
We also check the log: the required tools were really called and the
claim_total the agent reported matches the total computed by code.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from tools import DATA_DIR, load_claim, load_policy

ANSWER_KEY_PATH = DATA_DIR / "answer_key.json"


def load_answer_key() -> dict:
    with ANSWER_KEY_PATH.open() as f:
        return json.load(f)


def reference_review(claim_id: str) -> dict:
    """Rule-based ground truth, independent of the language model."""
    policy, claim = load_policy(), load_claim(claim_id)
    cats = policy["categories"]
    required = policy["receipts"]["required_fields"]
    receipts = [li.get("receipt") or {} for li in claim["line_items"]]

    total = round(sum(r.get("amount") or 0 for r in receipts), 2)
    missing = [f"item {li['item_id']}: {f}" for li, r in zip(claim["line_items"], receipts)
               for f in required if r.get(f) in (None, "")]

    prohibited, violations = [], []
    meals_by_day: dict[str, float] = defaultdict(float)
    for li, r in zip(claim["line_items"], receipts):
        cat, amt = r.get("category"), r.get("amount") or 0
        if cat == "meals":
            if r.get("alcohol") and not cats["meals"]["alcohol_reimbursable"]:
                prohibited.append(f"item {li['item_id']}: alcohol")
            if r.get("date"):
                meals_by_day[r["date"]] += amt
        elif cat == "lodging" and amt > cats["lodging"]["nightly_limit"]:
            violations.append(f"item {li['item_id']}: lodging {amt} > {cats['lodging']['nightly_limit']}")
        elif cat == "ground_transport" and amt > cats["ground_transport"]["per_trip_limit"]:
            violations.append(f"item {li['item_id']}: ground {amt} > {cats['ground_transport']['per_trip_limit']}")
        elif cat == "airfare":
            if r.get("class", "economy") != cats["airfare"]["allowed_class"]:
                prohibited.append(f"item {li['item_id']}: {r.get('class')} class airfare")
            elif amt > cats["airfare"]["per_trip_limit"]:
                violations.append(f"item {li['item_id']}: airfare {amt} > {cats['airfare']['per_trip_limit']}")
    for day, amt in sorted(meals_by_day.items()):
        if round(amt, 2) > cats["meals"]["daily_limit"]:
            violations.append(f"meals on {day}: {round(amt, 2)} > {cats['meals']['daily_limit']}")

    if missing:
        decision = "NEEDS_INFO"
    elif prohibited:
        decision = "REJECT"
    elif violations:
        decision = "FLAG_FOR_REVIEW"
    else:
        decision = "APPROVE"
    return {"decision": decision, "claim_total": total, "missing": missing,
            "prohibited": prohibited, "violations": violations,
            "meals_by_day": {d: round(a, 2) for d, a in meals_by_day.items()}}


def grade(result, answer_key: dict) -> dict:
    """Grade one ReviewResult. Returns a dict of named pass/fail checks."""
    expected = answer_key[result.claim_id]
    ref = reference_review(result.claim_id)
    rec = result.recommendation or {}
    called = {c["tool"] for c in result.tool_calls}

    checks = {
        "submitted_recommendation": result.recommendation is not None,
        "answer_key_matches_reference": expected["expected_decision"] == ref["decision"]
                                        and abs(expected["expected_claim_total"] - ref["claim_total"]) < 0.01,
        "decision_matches_answer_key": rec.get("decision") == expected["expected_decision"],
        "claim_total_matches_code": rec.get("claim_total") is not None
                                    and abs(rec["claim_total"] - ref["claim_total"]) < 0.01,
        "required_tools_called": all(t in called for t in expected["required_tools"]),
    }
    if expected["expected_decision"] == "NEEDS_INFO":
        checks["listed_missing_info"] = bool(rec.get("missing_information"))
    if expected["expected_decision"] in ("FLAG_FOR_REVIEW", "REJECT"):
        checks["listed_violation"] = bool(rec.get("policy_violations"))

    return {"claim_id": result.claim_id,
            "description": expected["description"],
            "expected": expected["expected_decision"],
            "reference": ref["decision"],
            "agent": rec.get("decision"),
            "agent_total": rec.get("claim_total"),
            "reference_total": ref["claim_total"],
            "tools_called_in_order": [c["tool"] for c in result.tool_calls],
            "checks": checks,
            "passed": all(checks.values()),
            "log": str(result.log_path)}


def print_report(rows: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("TEST REPORT (agent vs. answer key vs. rule-based reference)")
    print("=" * 78)
    for r in rows:
        print(f"\n{r['claim_id']}  {'PASS' if r['passed'] else 'FAIL'}  - {r['description']}")
        print(f"  expected={r['expected']}  reference={r['reference']}  agent={r['agent']}")
        print(f"  total: agent={r['agent_total']}  computed-by-code={r['reference_total']}")
        print(f"  tools called: {' -> '.join(r['tools_called_in_order'])}")
        for name, ok in r["checks"].items():
            print(f"    [{'x' if ok else ' '}] {name}")
        print(f"  log: {r['log']}")
    n = sum(r["passed"] for r in rows)
    print(f"\n{n}/{len(rows)} claims passed.")


def save_report(rows: list[dict], path: Path) -> None:
    path.write_text(json.dumps(rows, indent=2))
