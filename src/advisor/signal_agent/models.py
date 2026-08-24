"""Result models for the autonomous Signal agent."""

from __future__ import annotations

from advisor.scalping.models import ScalpSignal
from pydantic import BaseModel, Field


class SignalAgentVerdict(BaseModel):
    """The LLM's final message — rationale only, never the numeric signals
    themselves. Those come from the tool-call side-channel (see agent.py);
    trusting the model to transcribe risk-relevant numbers back out of its
    own context is exactly the kind of failure mode this project has
    elsewhere had to guard against."""

    rationale: str = ""
    focus_symbols: list[str] = Field(default_factory=list)
    notes: str = ""


class SignalAgentResult(BaseModel):
    objective: str
    rationale: str = ""
    focus_symbols: list[str] = Field(default_factory=list)
    notes: str = ""
    signals: list[ScalpSignal] = Field(default_factory=list)
    swing_signals: list[dict] = Field(default_factory=list)
    run_id: str | None = None
    budget_exhausted: bool = False
