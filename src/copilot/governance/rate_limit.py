"""A minimal fixed-window rate limiter, keyed by authenticated identity.

Deliberately simple (in-memory, single-process) rather than reaching for a
dependency - at this project's scale that's the right tradeoff, and the
interface (`check(key) -> bool`) is small enough to swap for a Redis-backed
limiter behind a load balancer without touching call sites.
"""
import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, max_requests: int = 30, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> bool:
        """Returns True and records a hit if `key` is under its limit for the
        current window, False (and does not record a hit) if it's over."""
        now = time.time()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > self.window_seconds:
                hits.popleft()
            if len(hits) >= self.max_requests:
                return False
            hits.append(now)
            return True
