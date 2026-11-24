"""Offline tests: tools, reference checker, and the agent loop with a fake model.

No API key or network needed:  python -m unittest discover -s tests -v
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent  # noqa: E402
from evaluate import grade, load_answer_key, reference_review  # noqa: E402
from tools import ToolError, calculate, check_receipts, lookup_policy  # noqa: E402


class ToolTests(unittest.TestCase):
    def test_policy_lookup(self):
        self.assertEqual(lookup_policy("meals")["rules"]["daily_limit"], 75.0)
        with self.assertRaises(ToolError):
            lookup_policy("spa")

    def test_calculator(self):
        self.assertEqual(calculate("14.75 + 48.30 + 39.95")["result"], 103.0)
        self.assertTrue(calculate("14.75 + 48.30 + 39.95 > 75")["result"])
        self.assertEqual(calculate("sum(1, 2, 3.5)")["result"], 6.5)
        for bad in ("__import__('os')", "1/0", "2 ** 10", "x + 1"):
            with self.assertRaises(ToolError):
                calculate(bad)

    def test_receipt_checker(self):
        self.assertTrue(check_receipts("CLM-001")["all_complete"])
        r = check_receipts("CLM-003")
        self.assertFalse(r["all_complete"])
        self.assertEqual(r["incomplete_items"], [
            {"item_id": "2", "merchant": "Congress Ave Hotel", "missing_fields": ["date"]}])
        with self.assertRaises(ToolError):
            check_receipts("CLM-999")


class ReferenceTests(unittest.TestCase):
    def test_answer_key_agrees_with_rules(self):
        for cid, exp in load_answer_key().items():
            ref = reference_review(cid)
            self.assertEqual(ref["decision"], exp["expected_decision"], cid)
            self.assertAlmostEqual(ref["claim_total"], exp["expected_claim_total"], 2)


# ---- fake model client that replays a scripted sequence of tool calls ----
from llm import ToolCall, Turn  # noqa: E402


def _turn(*calls, text=""):
    tcs = [ToolCall(f"c{i}", n, inp) for i, (n, inp) in enumerate(calls)]
    return Turn(text=text, tool_calls=tcs, stop="tool_use" if tcs else "end_turn",
                model="fake")


class FakeLLM:
    """Stands in for the Groq API. Records what the agent sends back."""
    name, model = "fake", "scripted"

    def __init__(self, script):
        self.script, self.users, self.tool_results = list(script), [], []

    def call(self, system, tools):
        return self.script.pop(0)

    def add_user(self, text):
        self.users.append(text)

    def add_assistant(self, turn):
        pass

    def add_tool_results(self, results):
        self.tool_results.append(results)


class LoopTests(unittest.TestCase):
    def test_loop_feeds_results_and_grades(self):
        fake = FakeLLM([
            _turn(("check_receipts", {"claim_id": "CLM-003"})),
            _turn(("ask_employee", {"question": "What date was the hotel stay?"})),
            _turn(("calculate", {"expression": "289.00 + 165.00 + 22.40 + 29.80"})),
            _turn(("submit_recommendation", {
                "decision": "NEEDS_INFO", "claim_total": 506.2,
                "reasons": ["Item 2 has no date"], "policy_violations": [],
                "missing_information": ["Date on hotel receipt (item 2)"]})),
        ])
        result = agent.review_claim("CLM-003", verbose=False, llm=fake)
        # the receipt-check result was sent back to the model
        first = json.loads(fake.tool_results[0][0][1])
        self.assertFalse(first["all_complete"])
        # non-interactive ask_employee says "don't guess"
        self.assertFalse(result.tool_calls[1]["output"]["answered"])
        row = grade(result, load_answer_key())
        self.assertTrue(row["passed"], row["checks"])
        self.assertTrue(result.log_path.exists())
        result.log_path.unlink()

    def test_bad_tool_calls_are_reported_not_raised(self):
        fake = FakeLLM([
            _turn(("lookup_policy", {"section": "spa"})),          # invalid enum
            _turn(("calculate", {"expr": "1+1"})),                 # wrong arg name
            _turn(("fly_to_moon", {})),                            # unknown tool
            _turn(text="I think it's fine."),                      # no tool -> nudge
            _turn(("submit_recommendation", {
                "decision": "APPROVE", "claim_total": 1.0, "reasons": [],
                "policy_violations": [], "missing_information": []})),
        ])
        result = agent.review_claim("CLM-001", verbose=False, llm=fake)
        self.assertEqual([c["is_error"] for c in result.tool_calls],
                         [True, True, True, False])
        self.assertEqual(len(fake.users), 2)   # claim + one reminder
        row = grade(result, load_answer_key())
        self.assertFalse(row["checks"]["claim_total_matches_code"])  # wrong total caught
        self.assertFalse(row["passed"])
        result.log_path.unlink()

    def test_gives_up_without_tools(self):
        fake = FakeLLM([_turn(text="ok")] * (agent.MAX_NUDGES + 1))
        result = agent.review_claim("CLM-001", verbose=False, llm=fake)
        self.assertIsNone(result.recommendation)
        self.assertEqual(result.stop_reason, "no_recommendation")
        result.log_path.unlink()


class ValidationTests(unittest.TestCase):
    def test_validate_args(self):
        v = agent.validate_args
        self.assertIsNone(v("calculate", {"expression": "1+1"}))
        self.assertIn("Missing", v("calculate", {}))
        self.assertIn("one of", v("submit_recommendation", {
            "decision": "PAY_NOW", "claim_total": 1, "reasons": [],
            "policy_violations": [], "missing_information": []}))
        self.assertIn("type number", v("submit_recommendation", {
            "decision": "APPROVE", "claim_total": "1", "reasons": [],
            "policy_violations": [], "missing_information": []}))


if __name__ == "__main__":
    unittest.main()
