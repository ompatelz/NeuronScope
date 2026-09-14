"""Validated configuration for NeuronScope multilayer perceptrons."""

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

LayerWidth = Annotated[int, Field(ge=1, le=256)]


class ActivationName(StrEnum):
    """Hidden-layer activation functions supported by the workbench."""

    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"


class InitializationName(StrEnum):
    """Weight initialization strategies supported by the model builder."""

    DEFAULT = "default"
    XAVIER = "xavier"
    HE = "he"


class MLPConfig(BaseModel):
    """A bounded, serializable description of a feed-forward MLP."""

    model_config = ConfigDict(frozen=True)

    input_size: LayerWidth = 2
    hidden_layers: tuple[LayerWidth, ...] = Field(default=(8, 8), min_length=1, max_length=8)
    output_size: LayerWidth = 1
    activation: ActivationName = ActivationName.RELU
    initialization: InitializationName = InitializationName.DEFAULT
    seed: int = Field(default=0, ge=0, le=2**63 - 1)
