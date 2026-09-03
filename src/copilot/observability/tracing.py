"""Real OpenTelemetry instrumentation with a local, dependency-free sink.

`Tracing` holds its own `TracerProvider` instance rather than registering it
as the process-wide global (`trace.set_tracer_provider`) - that global can
only be set once per process, which would make every test share whichever
trace log path the first test happened to construct. Holding the provider as
a plain instance is still 100% real OpenTelemetry (`TracerProvider`,
`SpanProcessor`, `SpanExporter`, `Span` - the standard SDK primitives); it
just skips the optional global-registration convenience, which a
single-process CLI/API server doesn't need anyway (there's exactly one
`Tracing` instance per `build_graph()` call).

Spans are exported as JSON lines to a local file - a lightweight sink for a
project with no hosted trace backend. Swap `JSONLSpanExporter` for
`OTLPSpanExporter` (pointed at Jaeger, Tempo, Honeycomb, or LangFuse's OTLP
endpoint) to ship the exact same spans to a real observability backend
without touching any instrumentation call site.
"""
import json
import os
import threading
from collections.abc import Sequence
from contextlib import contextmanager

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor, SpanExporter, SpanExportResult

from copilot.config import default_trace_log_path


class JSONLSpanExporter(SpanExporter):
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        lines = [json.dumps(_span_to_dict(span), default=str) for span in spans]
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


def _span_to_dict(span: ReadableSpan) -> dict:
    duration_ms = None
    if span.start_time is not None and span.end_time is not None:
        duration_ms = (span.end_time - span.start_time) / 1_000_000
    return {
        "name": span.name,
        "trace_id": format(span.context.trace_id, "032x") if span.context else None,
        "span_id": format(span.context.span_id, "016x") if span.context else None,
        "parent_span_id": format(span.parent.span_id, "016x") if span.parent else None,
        "start_time_unix_ns": span.start_time,
        "end_time_unix_ns": span.end_time,
        "duration_ms": duration_ms,
        "attributes": dict(span.attributes or {}),
        "status": span.status.status_code.name if span.status else None,
    }


class Tracing:
    """One instance per `build_graph()` call, mirroring the `AuditLog`/
    `ApprovalQueue` pattern - a fresh object bound to a specific log path,
    not a hidden process-wide singleton, so tests can isolate their own
    trace output the same way they isolate the audit and checkpoint DBs."""

    def __init__(self, trace_log_path: str | None = None, service_name: str = "healthcare-copilot"):
        self.trace_log_path = trace_log_path or default_trace_log_path()
        resource = Resource.create({"service.name": service_name})
        self.provider = TracerProvider(resource=resource)
        self.provider.add_span_processor(SimpleSpanProcessor(JSONLSpanExporter(self.trace_log_path)))
        if os.environ.get("COPILOT_TRACE_CONSOLE"):
            self.provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        self._tracer = self.provider.get_tracer("copilot")

    @contextmanager
    def span(self, name: str, **attributes):
        clean_attrs = {k: v for k, v in attributes.items() if v is not None}
        with self._tracer.start_as_current_span(name, attributes=clean_attrs) as span:
            yield span
