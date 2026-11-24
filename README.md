# ClaimCheck: Expense Reimbursement Policy Checker Agent

CSYE 7374 Agentic AI mini project by Pawan Chaudhari.

ClaimCheck is a command-line AI agent that reviews **fictional** employee expense claims
against a small company travel policy. It **recommends** one of four decisions for a human
finance reviewer: `APPROVE`, `FLAG_FOR_REVIEW`, `REJECT` or `NEEDS_INFO`. It never approves
a payment itself.

## How it works

```
claim JSON ──► LLM (openai/gpt-oss-20b on Groq) ──decides next step──► one of the tools
                    ▲                                                     │
                    └────────────── tool result (JSON) ◄──────────────────┘
               (the loop repeats until the model calls submit_recommendation)
```

The code has no fixed sequence of steps. On each turn the model reads the results so far
and picks the next action:

| Tool / action | What it does |
|---|---|
| `lookup_policy` | Reads rules from `data/policy.json`: category limits, required receipt fields, decision rules |
| `calculate` | Safe arithmetic evaluator (AST based, no `eval`) for totals, daily sums and limit comparisons |
| `check_receipts` | Finds receipts missing required fields (date, merchant, amount, category) |
| `ask_employee` | Asks the human for missing details instead of guessing. With `-i` you answer it; otherwise it returns "no answer" |
| `submit_recommendation` | Final structured recommendation; this ends the run |

Policy decision rules, applied in order: missing receipt field → `NEEDS_INFO`;
non-reimbursable expense (alcohol, non-economy airfare) → `REJECT`; limit exceeded →
`FLAG_FOR_REVIEW`; otherwise → `APPROVE`.

Every model response (its reasoning, text and tool calls) and every tool result is written
to `logs/<CLAIM>_groq_<timestamp>.jsonl` as evidence of what the agent did.

## Setup

Requires **Python 3.10+**. No packages need to be installed: the project uses only the
Python standard library.

1. Create a free API key at [console.groq.com](https://console.groq.com) (no credit card).
2. Save it in a `.env` file in this folder:

   ```bash
   echo 'GROQ_API_KEY=gsk_your_key_here' > .env
   chmod 600 .env
   ```

   `.env` is git-ignored, so the key is never committed. (Alternatively, `export GROQ_API_KEY=...`.)

Optional: set `GROQ_MODEL` or pass `--model <name>` to use a different Groq model
(default `openai/gpt-oss-20b`).

## Commands

```bash
python3 main.py list                  # show the sample claims
python3 main.py review CLM-001        # run the agent on one claim
python3 main.py review CLM-003 -i     # interactive: the agent may ask you for missing info
python3 main.py test                  # run all 4 test cases and grade them
python3 main.py test -q               # same, report only
python3 main.py reference CLM-002     # rule-based expected result (no API call)
python3 -m unittest discover -s tests -v   # offline tests (no API key needed)
```

## Test cases

| Claim | Scenario | Expected |
|---|---|---|
| `CLM-001` | All receipts complete, all limits respected (total $856.65) | `APPROVE` |
| `CLM-002` | Three meal receipts on 2026-10-12 add up to $103.00, over the $75 daily limit, although no single receipt exceeds it (total $714.30) | `FLAG_FOR_REVIEW` |
| `CLM-003` | Hotel receipt (item 2) has no date (total $506.20) | `NEEDS_INFO` |
| `CLM-004` | Business-class flight ($640). It is under the $800 airfare limit, but only economy is reimbursable (total $907.75) | `REJECT` |

## How results are verified

`python3 main.py test` does not take the model's word for it. For each claim, `evaluate.py` checks:

1. **Decision** matches `data/answer_key.json`.
2. **Answer key** agrees with an independent rule-based checker (`reference_review`) that
   computes the correct decision from `policy.json` and the claim file, without the LLM.
3. **Claim total** the agent reported matches the total computed by code (±$0.01).
4. **Required tools** were actually called (taken from the logged tool calls).
5. For the flag and reject cases, a violation is listed. For the missing-info case, the
   missing item is listed.

The report is printed and saved to `logs/test_report_groq_<timestamp>.json`.

The offline tests (`tests/test_offline.py`) check the tools and the rule-based checker. They
also drive the agent loop with a scripted fake model, to show that bad tool calls (unknown
tool, wrong argument, invalid value) come back as errors instead of crashing, and that the
grader catches a wrong total.

## Project layout

```
main.py            command-line interface
agent.py           agent loop, system prompt, tool-argument validation, JSONL logging
llm.py             Groq API client (chat completions with tool calling), .env loader
tools.py           tool implementations + JSON schemas sent to the model
evaluate.py        grading: answer key + rule-based reference + log checks
data/policy.json   the fictional travel policy
data/claims/       the four fictional claims
data/answer_key.json
tests/test_offline.py   unit tests + loop tests with a scripted fake model
logs/              run logs (JSONL) and test reports
```

## Notes

- Model: `openai/gpt-oss-20b` on Groq, temperature 0. The model's reasoning is logged, so
  the log shows *why* each tool was picked.
- The Groq API does not enforce tool schemas, so `agent.py` checks every tool call's
  arguments itself (required fields, types, allowed values). A bad call is sent back to the
  model as an error so it can correct itself.
- If the model stops without calling a tool, it gets a reminder (at most 3). The run stops
  after 15 turns at most, so a confused model cannot loop forever.
- Groq free-tier limits on this account were 1,000 requests/day and 8,000 tokens/minute. A
  full test run uses about 25–30 requests. On a rate-limit response the client waits and
  retries automatically, so a run takes about 3–4 minutes.
- A model's own explanation is not always accurate. In one run, its reasons said a per-day
  meal sum came from the calculator when it had done that sum itself (the sum was correct).
  This is why results are graded against the logs and code-computed values, not the model's text.
- All people, companies and receipts are fictional.
