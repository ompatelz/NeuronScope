"""Configurable and inspectable PyTorch multilayer perceptron."""

from dataclasses import dataclass
from typing import Literal, cast

import torch
from torch import Tensor, nn

from neuronscope.models.schemas import ActivationName, InitializationName, MLPConfig


@dataclass(frozen=True)
class DenseLayerMetadata:
    """Stable metadata for one learned affine layer."""

    name: str
    input_size: int
    output_size: int
    activation: ActivationName | None
    parameter_count: int


@dataclass(frozen=True)
class MLPArchitecture:
    """A visualization-friendly description of an MLP."""

    input_size: int
    hidden_layers: tuple[int, ...]
    output_size: int
    layers: tuple[DenseLayerMetadata, ...]
    total_parameters: int


def _new_activation(name: ActivationName) -> nn.Module:
    match name:
        case ActivationName.RELU:
            return nn.ReLU()
        case ActivationName.SIGMOID:
            return nn.Sigmoid()
        case ActivationName.TANH:
            return nn.Tanh()


def _initialize_linear(
    layer: nn.Linear,
    initialization: InitializationName,
    activation: ActivationName | None,
) -> None:
    if initialization is InitializationName.DEFAULT:
        return

    if initialization is InitializationName.XAVIER:
        gain_name: Literal["relu", "sigmoid", "tanh", "linear"] = (
            activation.value if activation is not None else "linear"
        )
        nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain(gain_name))
    elif initialization is InitializationName.HE:
        nonlinearity: Literal["relu", "linear"] = (
            "relu" if activation is ActivationName.RELU else "linear"
        )
        nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)

    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class ConfigurableMLP(nn.Module):
    """A small binary or multiclass-ready MLP without training behavior."""

    config: MLPConfig
    hidden_linears: nn.ModuleList
    hidden_activations: nn.ModuleList
    output_layer: nn.Linear

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        self.config = config

        widths = (config.input_size, *config.hidden_layers, config.output_size)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.hidden_linears = nn.ModuleList(
                nn.Linear(widths[index], widths[index + 1])
                for index in range(len(config.hidden_layers))
            )
            self.hidden_activations = nn.ModuleList(
                _new_activation(config.activation) for _ in config.hidden_layers
            )
            self.output_layer = nn.Linear(widths[-2], widths[-1])

            for layer in self.hidden_linears:
                if isinstance(layer, nn.Linear):
                    _initialize_linear(layer, config.initialization, config.activation)
            _initialize_linear(self.output_layer, config.initialization, None)

    def forward(self, inputs: Tensor) -> Tensor:
        """Return raw output logits; probability conversion belongs to consumers."""
        values = inputs
        for linear, activation in zip(self.hidden_linears, self.hidden_activations, strict=True):
            values = cast(Tensor, activation(linear(values)))
        return cast(Tensor, self.output_layer(values))

    @property
    def architecture(self) -> MLPArchitecture:
        """Describe learned layers without exposing PyTorch module internals."""
        hidden_metadata = tuple(
            DenseLayerMetadata(
                name=f"hidden_{index}",
                input_size=layer.in_features,
                output_size=layer.out_features,
                activation=self.config.activation,
                parameter_count=sum(parameter.numel() for parameter in layer.parameters()),
            )
            for index, layer in enumerate(self.hidden_linears)
            if isinstance(layer, nn.Linear)
        )
        output_metadata = DenseLayerMetadata(
            name="output",
            input_size=self.output_layer.in_features,
            output_size=self.output_layer.out_features,
            activation=None,
            parameter_count=sum(parameter.numel() for parameter in self.output_layer.parameters()),
        )
        layers = (*hidden_metadata, output_metadata)
        return MLPArchitecture(
            input_size=self.config.input_size,
            hidden_layers=self.config.hidden_layers,
            output_size=self.config.output_size,
            layers=layers,
            total_parameters=sum(layer.parameter_count for layer in layers),
        )


def build_mlp(config: MLPConfig) -> ConfigurableMLP:
    """Build a deterministically initialized model from validated configuration."""
    return ConfigurableMLP(config)
