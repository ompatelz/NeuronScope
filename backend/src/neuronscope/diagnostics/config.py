"""Centralized, validated thresholds for deterministic diagnostic rules."""

from pydantic import BaseModel, ConfigDict, Field


class DiagnosticsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    consecutive_epochs: int = Field(default=3, ge=2, le=20)
    vanishing_gradient_norm: float = Field(default=1e-6, gt=0.0)
    exploding_gradient_norm: float = Field(default=100.0, gt=0.0)
    dead_relu_zero_percentage: float = Field(default=95.0, ge=0.0, le=100.0)
