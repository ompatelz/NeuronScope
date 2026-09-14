"""Validated configuration and structured results for model training."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from neuronscope.instrumentation.schemas import EpochInstrumentation


class OptimizerName(StrEnum):
    """Optimizers supported by the first synchronous training engine."""

    SGD = "sgd"
    ADAM = "adam"


class TrainingConfig(BaseModel):
    """Bounded controls for a deterministic full-batch training run."""

    model_config = ConfigDict(frozen=True)

    optimizer: OptimizerName = OptimizerName.ADAM
    learning_rate: float = Field(default=0.01, gt=0.0, le=1.0)
    epochs: int = Field(default=200, ge=1, le=5_000)
    instrumentation: bool = False


class EpochMetrics(BaseModel):
    """Observable model quality after one parameter update."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    epoch: int = Field(ge=1)
    loss: float = Field(ge=0.0)
    accuracy: float = Field(ge=0.0, le=1.0)


class TrainingResult(BaseModel):
    """Serializable history and final quality for one completed run."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    config: TrainingConfig
    history: tuple[EpochMetrics, ...]
    instrumentation: tuple[EpochInstrumentation, ...] = ()
    final_loss: float = Field(ge=0.0)
    final_accuracy: float = Field(ge=0.0, le=1.0)
