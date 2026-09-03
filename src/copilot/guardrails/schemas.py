from typing import Literal

from pydantic import BaseModel, Field


class PlannerOutput(BaseModel):
    task_type: Literal["research", "tool_task", "general"]
    steps: list[str] = Field(min_length=1, max_length=6)
    needs_retrieval: bool


class GuardrailVerdict(BaseModel):
    risk: Literal["low", "medium", "high"]
    issues: list[str]
    requires_approval: bool
    rationale: str
