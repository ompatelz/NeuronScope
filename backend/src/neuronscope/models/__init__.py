"""Public model-building interface."""

from neuronscope.models.mlp import (
    ConfigurableMLP,
    DenseLayerMetadata,
    MLPArchitecture,
    build_mlp,
)
from neuronscope.models.schemas import ActivationName, InitializationName, MLPConfig

__all__ = [
    "ActivationName",
    "ConfigurableMLP",
    "DenseLayerMetadata",
    "InitializationName",
    "MLPArchitecture",
    "MLPConfig",
    "build_mlp",
]
