"""Eval harness: golden dataset + deterministic checks + LLM-as-judge, with
regression detection against a stored baseline.

Run:
    python -m eval.run_eval                # run + compare against baseline.json
    python -m eval.run_eval --update-baseline   # (re)write the baseline

Exits non-zero if any previously-passing case regresses, or overall pass rate
drops - suitable for wiring into CI.
"""
import argparse
import json
import sys
from pathlib import Path

from copilot.graph import build_graph, resume_request, run_request
from copilot.llm import get_backend
from eval.judge import judge

DATA_PATH = Path(__file__).parent / "golden_dataset.jsonl"
BASELINE_PATH = Path(__file__).parent / "baseline.json"
REPORT_PATH = Path(__file__).parent / "last_report.json"

_RISK_ORDER = {"low": 0, "medium": 1, "high": 2}


def load_cases() -> list[dict]:
    with open(DATA_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _risk_at_most(actual: str, maximum: str) -> bool:
    return _RISK_ORDER[actual] <= _RISK_ORDER[maximum]


def run_case(app, llm, case: dict) -> dict:
    request_id = f"eval-{case['id']}"
    _, result = run_request(app, query=case["query"], user_id="eval-harness", role=case["role"], request_id=request_id)
    interrupted = "__interrupt__" in result

    if interrupted:
        approved = case.get("resume_decision", True)
        result = resume_request(app, request_id=request_id, approved=approved, approver="eval-harness")

    checks: dict[str, bool] = {}
    if "expected_task_type" in case:
        checks["task_type_match"] = result.get("task_type") == case["expected_task_type"]
    if case.get("expected_keywords"):
        lowered = result.get("final_answer", "").lower()
        checks["keywords_present"] = all(kw.lower() in lowered for kw in case["expected_keywords"])
    if "max_risk" in case:
        checks["risk_within_bound"] = _risk_at_most(result.get("guardrail_risk", "low"), case["max_risk"])
    if "expect_blocked" in case:
        checks["blocked_as_expected"] = bool(result.get("blocked", False)) == case["expect_blocked"]
    if "expect_interrupt" in case:
        checks["interrupt_as_expected"] = interrupted == case["expect_interrupt"]
    if "expected_tool" in case:
        checks["tool_selection_correct"] = case["expected_tool"] in result.get("tools_used", [])
    if case.get("require_clean_citations"):
        issues = result.get("guardrail_issues", [])
        checks["citations_verified"] = not any("unverified citation" in issue for issue in issues)

    answer = result.get("final_answer", "")
    verdict = judge(llm, query=case["query"], answer=answer, criteria=case.get("criteria", ""))

    passed = all(checks.values()) and verdict.verdict == "pass"
    return {
        "id": case["id"],
        "passed": passed,
        "checks": checks,
        "judge_score": verdict.score,
        "judge_verdict": verdict.verdict,
        "judge_rationale": verdict.rationale,
        "final_answer": answer,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-baseline", action="store_true", help="write current results as the new baseline")
    args = parser.parse_args()

    llm = get_backend()
    app = build_graph(llm=llm)
    cases = load_cases()
    results = [run_case(app, llm, case) for case in cases]

    pass_rate = sum(r["passed"] for r in results) / len(results)
    report = {"pass_rate": pass_rate, "total": len(results), "results": results}
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print(f"\n{'=' * 60}")
    print(f"Eval run: {sum(r['passed'] for r in results)}/{len(results)} passed ({pass_rate:.0%})")
    print("=" * 60)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {r['id']:<40} judge={r['judge_score']}/5  checks={r['checks']}")
        if not r["passed"]:
            print(f"         rationale: {r['judge_rationale']}")

    regressed = False
    if BASELINE_PATH.exists() and not args.update_baseline:
        baseline = json.loads(BASELINE_PATH.read_text())
        baseline_by_id = {r["id"]: r["passed"] for r in baseline["results"]}
        for r in results:
            was_passing = baseline_by_id.get(r["id"])
            if was_passing and not r["passed"]:
                print(f"\nREGRESSION: case '{r['id']}' previously passed, now fails")
                regressed = True
        if pass_rate < baseline["pass_rate"] - 1e-9:
            print(f"\nREGRESSION: pass rate dropped from {baseline['pass_rate']:.0%} to {pass_rate:.0%}")
            regressed = True
        if not regressed:
            print("\nNo regressions detected against baseline.")

    if args.update_baseline or not BASELINE_PATH.exists():
        BASELINE_PATH.write_text(json.dumps(report, indent=2))
        print(f"\nBaseline written to {BASELINE_PATH}")

    sys.exit(1 if regressed else 0)


if __name__ == "__main__":
    main()
