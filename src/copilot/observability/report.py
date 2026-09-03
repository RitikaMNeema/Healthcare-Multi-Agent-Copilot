"""Aggregates the JSONL trace log into per-span latency percentiles and
token-cost totals, and renders a static HTML dashboard - no external
monitoring stack required to see what the traces already captured.

Run: python -m copilot.observability.report
"""
import json
import math
from pathlib import Path

from copilot.config import default_trace_log_path

DASHBOARD_PATH = Path(__file__).resolve().parents[3] / "data" / "observability_dashboard.html"


def load_spans(path: str) -> list[dict]:
    spans = []
    if not Path(path).exists():
        return spans
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                spans.append(json.loads(line))
    return spans


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    lower, upper = math.floor(k), math.ceil(k)
    if lower == upper:
        return ordered[int(k)]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (k - lower)


def summarize(spans: list[dict]) -> dict:
    by_name: dict[str, list[dict]] = {}
    for span in spans:
        by_name.setdefault(span["name"], []).append(span)

    latency_rows = []
    for name, group in sorted(by_name.items()):
        durations = [s["duration_ms"] for s in group if s.get("duration_ms") is not None]
        latency_rows.append({
            "name": name,
            "count": len(group),
            "p50_ms": round(_percentile(durations, 0.50), 2),
            "p95_ms": round(_percentile(durations, 0.95), 2),
            "p99_ms": round(_percentile(durations, 0.99), 2),
            "max_ms": round(max(durations), 2) if durations else 0.0,
        })

    cost_by_model: dict[str, dict] = {}
    for span in spans:
        attrs = span.get("attributes", {})
        model = attrs.get("model")
        if not model or "cost_usd" not in attrs:
            continue
        bucket = cost_by_model.setdefault(model, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
        bucket["calls"] += 1
        bucket["input_tokens"] += attrs.get("input_tokens", 0)
        bucket["output_tokens"] += attrs.get("output_tokens", 0)
        bucket["cost_usd"] += attrs.get("cost_usd", 0.0)

    total_cost = round(sum(b["cost_usd"] for b in cost_by_model.values()), 6)

    return {
        "total_spans": len(spans),
        "latency_by_span": latency_rows,
        "cost_by_model": {k: {**v, "cost_usd": round(v["cost_usd"], 6)} for k, v in cost_by_model.items()},
        "total_cost_usd": total_cost,
    }


def render_html(summary: dict) -> str:
    latency_rows = "".join(
        f"<tr><td>{r['name']}</td><td>{r['count']}</td><td>{r['p50_ms']}</td>"
        f"<td>{r['p95_ms']}</td><td>{r['p99_ms']}</td><td>{r['max_ms']}</td></tr>"
        for r in summary["latency_by_span"]
    )
    cost_rows = "".join(
        f"<tr><td>{model}</td><td>{v['calls']}</td><td>{v['input_tokens']}</td>"
        f"<td>{v['output_tokens']}</td><td>${v['cost_usd']:.6f}</td></tr>"
        for model, v in summary["cost_by_model"].items()
    ) or "<tr><td colspan='5'>No priced LLM calls recorded (mock backend reports $0).</td></tr>"

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Copilot Observability</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 2rem; color: #1a1a1a; }}
h1 {{ font-size: 1.4rem; }} h2 {{ font-size: 1.1rem; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; max-width: 900px; }}
th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; font-size: 0.9rem; }}
th {{ background: #f5f5f5; }}
.total {{ font-weight: 600; margin-top: 0.5rem; }}
</style></head>
<body>
<h1>Copilot Observability Dashboard</h1>
<p>{summary['total_spans']} spans captured.</p>

<h2>Latency by span (ms)</h2>
<table><tr><th>span</th><th>count</th><th>p50</th><th>p95</th><th>p99</th><th>max</th></tr>
{latency_rows}
</table>

<h2>Token cost by model</h2>
<table><tr><th>model</th><th>calls</th><th>input tokens</th><th>output tokens</th><th>cost</th></tr>
{cost_rows}
</table>
<p class="total">Total estimated cost: ${summary['total_cost_usd']:.6f}</p>
</body></html>"""


def main() -> None:
    trace_log_path = default_trace_log_path()
    spans = load_spans(trace_log_path)
    summary = summarize(spans)

    print(f"\nRead {summary['total_spans']} spans from {trace_log_path}\n")
    print(f"{'span':<28}{'count':>8}{'p50 ms':>10}{'p95 ms':>10}{'p99 ms':>10}{'max ms':>10}")
    for row in summary["latency_by_span"]:
        print(f"{row['name']:<28}{row['count']:>8}{row['p50_ms']:>10}{row['p95_ms']:>10}{row['p99_ms']:>10}{row['max_ms']:>10}")

    print(f"\n{'model':<20}{'calls':>8}{'input tok':>12}{'output tok':>12}{'cost':>12}")
    for model, v in summary["cost_by_model"].items():
        print(f"{model:<20}{v['calls']:>8}{v['input_tokens']:>12}{v['output_tokens']:>12}{v['cost_usd']:>12.6f}")
    print(f"\nTotal estimated cost: ${summary['total_cost_usd']:.6f}")

    DASHBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    DASHBOARD_PATH.write_text(render_html(summary), encoding="utf-8")
    print(f"\nDashboard written to {DASHBOARD_PATH}")


if __name__ == "__main__":
    main()
