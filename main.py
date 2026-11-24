"""Command-line entry point for the Expense Reimbursement Policy Checker Agent.

  python main.py list                     list available claims
  python main.py review CLM-001           review one claim
  python main.py review CLM-003 -i        let the agent ask you for missing details
  python main.py test                     run all claims and grade against the answer key
  python main.py reference CLM-002        rule-based expected result (no LLM)

Uses the Groq API (set GROQ_API_KEY in .env). Add  --model <name>  to pick a model.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

from llm import DEFAULT_MODEL, LLMError
from tools import CLAIMS_DIR


def _claim_ids() -> list[str]:
    return sorted(p.stem for p in CLAIMS_DIR.glob("*.json"))


def cmd_list(_args) -> int:
    for cid in _claim_ids():
        claim = json.loads((CLAIMS_DIR / f"{cid}.json").read_text())
        print(f"{cid}  {claim['employee']:<26} {claim['purpose']}  "
              f"({len(claim['line_items'])} items)")
    return 0


def cmd_review(args) -> int:
    from agent import review_claim
    result = review_claim(args.claim_id, model=args.model, interactive=args.interactive)
    return 0 if result.recommendation else 1


def cmd_test(args) -> int:
    from agent import LOG_DIR, review_claim
    from evaluate import grade, load_answer_key, print_report, save_report
    key = load_answer_key()
    rows = []
    for cid in args.claims or list(key):
        result = review_claim(cid, model=args.model, interactive=False,
                              verbose=not args.quiet)
        rows.append(grade(result, key))
    print_report(rows)
    out = LOG_DIR / f"test_report_groq_{datetime.now():%Y%m%d-%H%M%S}.json"
    save_report(rows, out)
    print(f"report saved: {out}")
    return 0 if all(r["passed"] for r in rows) else 1


def cmd_reference(args) -> int:
    from evaluate import reference_review
    print(json.dumps(reference_review(args.claim_id), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="expense-agent",
                                description="Expense Reimbursement Policy Checker Agent")
    sub = p.add_subparsers(dest="command", required=True)

    def add_llm_opts(sp):
        sp.add_argument("--model", default=None,
                        help=f"Groq model name (default: {DEFAULT_MODEL})")

    sub.add_parser("list", help="list sample claims").set_defaults(func=cmd_list)

    r = sub.add_parser("review", help="review one claim with the agent")
    r.add_argument("claim_id", choices=_claim_ids())
    r.add_argument("-i", "--interactive", action="store_true",
                   help="let the agent ask you (the employee) for missing details")
    add_llm_opts(r)
    r.set_defaults(func=cmd_review)

    t = sub.add_parser("test", help="run claims and grade against the answer key")
    t.add_argument("claims", nargs="*", help="claim IDs (default: all in answer key)")
    t.add_argument("-q", "--quiet", action="store_true", help="only print the report")
    add_llm_opts(t)
    t.set_defaults(func=cmd_test)

    ref = sub.add_parser("reference", help="rule-based expected result (no LLM)")
    ref.add_argument("claim_id", choices=_claim_ids())
    ref.set_defaults(func=cmd_reference)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except LLMError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
