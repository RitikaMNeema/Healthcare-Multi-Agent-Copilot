"""Retry-with-backoff plus model fallback for every LLM call in the system.

`call_with_fallback` tries each model in order; within a model it retries a
few times with exponential backoff before giving up on that model and moving
to the next one. If every model fails, the caller gets `AllModelsFailedError`
and can decide whether to fall back to a canned safe response.
"""
import logging
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class AllModelsFailedError(Exception):
    pass


def call_with_fallback(
    attempt: Callable[[str], T],
    *,
    models: list[str],
    max_retries_per_model: int = 2,
    base_delay: float = 0.5,
) -> tuple[T, str]:
    """`attempt(model_id)` should perform one LLM call and return its result.

    Returns (result, model_id_that_succeeded). Raises AllModelsFailedError if
    every model in `models` exhausted its retries.
    """
    last_exc: Exception | None = None
    for model in models:
        for retry in range(max_retries_per_model):
            try:
                return attempt(model), model
            except Exception as exc:  # noqa: BLE001 - intentionally broad: any model/tool failure triggers fallback
                last_exc = exc
                logger.warning("model %s attempt %d/%d failed: %s", model, retry + 1, max_retries_per_model, exc)
                if retry < max_retries_per_model - 1:
                    time.sleep(base_delay * (2**retry))
        logger.warning("model %s exhausted retries, falling back to next model", model)

    raise AllModelsFailedError(f"all models {models} failed; last error: {last_exc}") from last_exc
