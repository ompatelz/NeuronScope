"""Tests for validated and deterministic MLP construction."""

import pytest
import torch
from pydantic import ValidationError
from torch import nn

from neuronscope.models import (
    ActivationName,
    InitializationName,
    MLPConfig,
    build_mlp,
)


def test_mlp_constructs_requested_architecture_and_output_shape() -> None:
    config = MLPConfig(input_size=2, hidden_layers=(4, 3), output_size=1, seed=17)
    model = build_mlp(config)

    output = model(torch.zeros((5, 2)))

    assert output.shape == (5, 1)
    assert model.architecture.input_size == 2
    assert model.architecture.hidden_layers == (4, 3)
    assert [layer.name for layer in model.architecture.layers] == [
        "hidden_0",
        "hidden_1",
        "output",
    ]
    assert model.architecture.total_parameters == 31


@pytest.mark.parametrize(
    ("activation", "expected_type"),
    [
        (ActivationName.RELU, nn.ReLU),
        (ActivationName.SIGMOID, nn.Sigmoid),
        (ActivationName.TANH, nn.Tanh),
    ],
)
def test_mlp_uses_configured_hidden_activation(
    activation: ActivationName, expected_type: type[nn.Module]
) -> None:
    model = build_mlp(MLPConfig(hidden_layers=(3, 3), activation=activation))

    assert all(isinstance(module, expected_type) for module in model.hidden_activations)
    assert model.architecture.layers[-1].activation is None


def test_same_seed_produces_same_parameters() -> None:
    config = MLPConfig(hidden_layers=(6, 4), initialization=InitializationName.XAVIER, seed=123)

    first = build_mlp(config)
    second = build_mlp(config)

    assert all(
        torch.equal(first_value, second.state_dict()[name])
        for name, first_value in first.state_dict().items()
    )


def test_different_seeds_produce_different_parameters() -> None:
    first = build_mlp(MLPConfig(hidden_layers=(4,), seed=1))
    second = build_mlp(MLPConfig(hidden_layers=(4,), seed=2))

    assert any(
        not torch.equal(first_value, second.state_dict()[name])
        for name, first_value in first.state_dict().items()
    )


@pytest.mark.parametrize("initialization", [InitializationName.XAVIER, InitializationName.HE])
def test_explicit_initialization_zeros_biases(initialization: InitializationName) -> None:
    model = build_mlp(MLPConfig(hidden_layers=(5,), initialization=initialization))

    assert all(
        layer.bias is not None and torch.count_nonzero(layer.bias).item() == 0
        for layer in (*model.hidden_linears, model.output_layer)
    )


@pytest.mark.parametrize(
    "values",
    [
        {"input_size": 0},
        {"hidden_layers": ()},
        {"hidden_layers": (257,)},
        {"hidden_layers": (1,) * 9},
        {"output_size": 0},
        {"seed": -1},
        {"activation": "softmax"},
        {"initialization": "normal"},
    ],
)
def test_invalid_model_configuration_is_rejected(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        MLPConfig.model_validate(values)
