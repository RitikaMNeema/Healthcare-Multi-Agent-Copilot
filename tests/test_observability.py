import json

from copilot.observability.cost import estimate_cost_usd
from copilot.observability.report import load_spans, summarize
from copilot.observability.tracing import Tracing


def test_estimate_cost_known_model():
    cost = estimate_cost_usd("claude-opus-5", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == 30.00  # $5 input + $25 output per 1M tokens


def test_estimate_cost_unknown_model_is_zero():
    assert estimate_cost_usd("some-future-model", 1000, 1000) == 0.0


def test_estimate_cost_missing_usage_is_zero():
    assert estimate_cost_usd("claude-opus-5", None, None) == 0.0


def test_tracing_writes_spans_with_expected_shape(tmp_path):
    log_path = str(tmp_path / "traces.jsonl")
    tracing = Tracing(trace_log_path=log_path)

    with tracing.span("node.plan", request_id="r1", role="operator") as span:
        span.set_attribute("model", "claude-opus-5")

    spans = load_spans(log_path)
    assert len(spans) == 1
    assert spans[0]["name"] == "node.plan"
    assert spans[0]["attributes"]["request_id"] == "r1"
    assert spans[0]["attributes"]["model"] == "claude-opus-5"
    assert spans[0]["duration_ms"] is not None and spans[0]["duration_ms"] >= 0


def test_tracing_captures_nested_spans_with_parent_link(tmp_path):
    log_path = str(tmp_path / "traces.jsonl")
    tracing = Tracing(trace_log_path=log_path)

    with tracing.span("outer") as outer_span:
        with tracing.span("inner"):
            pass

    spans = {s["name"]: s for s in load_spans(log_path)}
    assert spans["inner"]["parent_span_id"] == format(outer_span.get_span_context().span_id, "016x")


def test_summarize_computes_latency_percentiles_and_cost(tmp_path):
    log_path = tmp_path / "traces.jsonl"
    rows = [
        {"name": "node.plan", "duration_ms": 10.0, "attributes": {}},
        {"name": "node.plan", "duration_ms": 20.0, "attributes": {}},
        {
            "name": "llm.complete_with_tools", "duration_ms": 5.0,
            "attributes": {"model": "claude-opus-5", "input_tokens": 1000, "output_tokens": 500, "cost_usd": 0.0175},
        },
    ]
    log_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    spans = load_spans(str(log_path))
    summary = summarize(spans)

    plan_row = next(r for r in summary["latency_by_span"] if r["name"] == "node.plan")
    assert plan_row["count"] == 2
    assert plan_row["max_ms"] == 20.0

    assert summary["cost_by_model"]["claude-opus-5"]["calls"] == 1
    assert summary["cost_by_model"]["claude-opus-5"]["input_tokens"] == 1000
    assert summary["total_cost_usd"] == 0.0175
