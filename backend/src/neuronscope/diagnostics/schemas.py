"""Serializable evidence emitted by diagnostic rules."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class DiagnosticType(StrEnum):
    VANISHING_GRADIENTS = "vanishing_gradients"
    EXPLODING_GRADIENTS = "exploding_gradients"
    DEAD_RELU = "dead_relu"


class DiagnosticSeverity(StrEnum):
    WARNING = "warning"
    CRITICAL = "critical"


class DiagnosticEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    layer_name: str
    epochs: tuple[int, ...]
    metric: str
    observed_values: tuple[float, ...]
    threshold: float = Field(ge=0.0)


class DiagnosticResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: DiagnosticType
    severity: DiagnosticSeverity
    evidence: tuple[DiagnosticEvidence, ...]
    explanation: str
    possible_actions: tuple[str, ...]
