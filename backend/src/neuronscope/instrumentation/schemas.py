"""Serializable scalar summaries captured while a model trains."""

from pydantic import BaseModel, ConfigDict, Field


class ActivationStatistics(BaseModel):
    """Finite-value activation statistics for one hidden layer."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    mean: float | None
    std: float | None = Field(ge=0.0)
    minimum: float | None
    maximum: float | None
    zero_percentage: float = Field(ge=0.0, le=100.0)
    nonfinite_count: int = Field(ge=0)


class GradientStatistics(BaseModel):
    """Statistics across all available parameter gradients for one layer."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    norm: float | None = Field(ge=0.0)
    mean: float | None
    std: float | None = Field(ge=0.0)
    nonfinite_count: int = Field(ge=0)


class LayerInstrumentation(BaseModel):
    """One epoch's scalar debugger data for a learned affine layer."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    layer_name: str
    activation: ActivationStatistics | None
    weight_norm: float | None = Field(ge=0.0)
    gradients: GradientStatistics


class EpochInstrumentation(BaseModel):
    """Layer summaries captured from one training forward/backward pass."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    epoch: int = Field(ge=1)
    layers: tuple[LayerInstrumentation, ...]
