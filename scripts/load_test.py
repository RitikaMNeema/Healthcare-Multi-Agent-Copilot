"""Self-contained load test for the API server.

Starts a fresh copy of `api.server:app` as a subprocess with isolated state
(its own audit/checkpoint/trace files) and a batch of throwaway API-key
identities generated just for this run - real virtual users, not one key
reused past its own rate limit - fires concurrent requests at it, tears the
server down, and reports throughput and latency percentiles.

With the default mock LLM backend, this measures the *system's own*
overhead - routing through the graph, guardrails, SQLite I/O for the audit
log and checkpointer - since a mock LLM call is near-instant; it is not a
measurement of Claude API latency. Pass --backend claude (with
ANTHROPIC_API_KEY set) to include real model latency in the numbers.

Run: python scripts/load_test.py --concurrency 20 --requests 300
"""
import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from copilot.governance.identity import generate_api_key  # noqa: E402

QUERIES = [
    "What is the timely filing deadline for Medicare claims?",
    "What is the denial rate for Aetna?",
    "Why was claim CLM-000039 denied?",
    "How many claims were denied by UnitedHealthcare?",
    "What does denial code CO-50 mean?",
]


def seed_throwaway_identities(path: Path, n: int) -> list[str]:
    if path.exists():
        path.unlink()
    return [generate_api_key(f"loadtest-user-{i}", "operator", path=str(path)) for i in range(n)]


def wait_for_health(base_url: str, timeout_s: float = 60.0) -> None:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            response = httpx.get(f"{base_url}/health", timeout=2.0)
            if response.status_code == 200:
                return
        except httpx.HTTPError as exc:
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"server did not become healthy within {timeout_s}s: {last_error}")


async def fire(client: httpx.AsyncClient, api_key: str, query: str) -> tuple[float, int | None]:
    start = time.perf_counter()
    try:
        response = await client.post("/chat", json={"query": query}, headers={"X-API-Key": api_key})
        status = response.status_code
    except httpx.HTTPError:
        status = None
    return (time.perf_counter() - start) * 1000, status


async def worker(client: httpx.AsyncClient, jobs: list[tuple[str, str]], results: list) -> None:
    for api_key, query in jobs:
        results.append(await fire(client, api_key, query))


async def run_load(base_url: str, keys: list[str], total_requests: int, concurrency: int):
    jobs = [(keys[i % len(keys)], QUERIES[i % len(QUERIES)]) for i in range(total_requests)]
    buckets = [jobs[i::concurrency] for i in range(concurrency)]
    results: list[tuple[float, int | None]] = []

    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        start = time.perf_counter()
        await asyncio.gather(*(worker(client, bucket, results) for bucket in buckets))
        wall_time = time.perf_counter() - start

    return results, wall_time


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    floor_i, ceil_i = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[floor_i] + (ordered[ceil_i] - ordered[floor_i]) * (k - floor_i)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--port", type=int, default=8199)
    parser.add_argument("--backend", default="mock", choices=["mock", "claude"])
    args = parser.parse_args()

    run_dir = REPO_ROOT / "data" / "loadtest_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    keys_path = run_dir / "api_keys.json"

    n_identities = max(args.concurrency, 10)
    raw_keys = seed_throwaway_identities(keys_path, n_identities)

    env = os.environ.copy()
    env["COPILOT_LLM_BACKEND"] = args.backend
    env["COPILOT_API_KEYS_FILE"] = str(keys_path)
    env["COPILOT_AUDIT_DB"] = str(run_dir / "audit.db")
    env["COPILOT_CHECKPOINT_DB"] = str(run_dir / "checkpoints.db")
    env["COPILOT_TRACE_LOG"] = str(run_dir / "traces.jsonl")

    base_url = f"http://127.0.0.1:{args.port}"
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.server:app", "--port", str(args.port)],
        cwd=str(REPO_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        wait_for_health(base_url)
        print(f"Server healthy at {base_url} (backend={args.backend}). "
              f"Running {args.requests} requests at concurrency={args.concurrency} "
              f"across {n_identities} distinct identities...\n")
        results, wall_time = asyncio.run(run_load(base_url, raw_keys, args.requests, args.concurrency))
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    latencies = [r[0] for r in results]
    status_counts: dict[str, int] = {}
    for _, status in results:
        key = str(status) if status is not None else "connection_error"
        status_counts[key] = status_counts.get(key, 0) + 1

    summary = {
        "backend": args.backend,
        "total_requests": len(results),
        "concurrency": args.concurrency,
        "distinct_identities": n_identities,
        "wall_time_s": round(wall_time, 3),
        "throughput_rps": round(len(results) / wall_time, 2) if wall_time else 0.0,
        "status_counts": status_counts,
        "latency_ms": {
            "p50": round(percentile(latencies, 0.50), 2),
            "p95": round(percentile(latencies, 0.95), 2),
            "p99": round(percentile(latencies, 0.99), 2),
            "max": round(max(latencies), 2) if latencies else 0.0,
        },
    }

    print(json.dumps(summary, indent=2))
    report_path = REPO_ROOT / "data" / "load_test_report.json"
    report_path.write_text(json.dumps(summary, indent=2))
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
