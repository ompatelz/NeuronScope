"""Tests for exact, input-specific model forward traces."""

import pytest
import torch

from neuronscope.datasets import DatasetConfig, DatasetPoint, DatasetResult
from neuronscope.experiments import compute_forward_trace
from neuronscope.models import MLPConfig, build_mlp


def test_forward_trace_contains_real_layer_values_and_restores_mode() -> None:
    model = build_mlp(MLPConfig(hidden_layers=(2,)))
    dataset = DatasetResult(
        config=DatasetConfig(samples=40),
        points=[DatasetPoint(x=1.0, y=-2.0, label=1)],
    )
    with torch.no_grad():
        model.hidden_linears[0].weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        model.hidden_linears[0].bias.zero_()
        model.output_layer.weight.copy_(torch.tensor([[2.0, -3.0]]))
        model.output_layer.bias.copy_(torch.tensor([-0.5]))
    model.train()

    trace = compute_forward_trace(model, dataset, 0)

    assert model.training is True
    assert trace.input_values == (1.0, -2.0)
    assert trace.expected_label == 1
    assert trace.layers[0].pre_activations == (1.0, -2.0)
    assert trace.layers[0].activations == (1.0, 0.0)
    assert trace.output_logit == pytest.approx(1.5)
    assert trace.predicted_probability == pytest.approx(torch.sigmoid(torch.tensor(1.5)).item())
    assert trace.predicted_label == 1


def test_forward_trace_payload_never_contains_model_parameters() -> None:
    model = build_mlp(MLPConfig(hidden_layers=(4,), seed=3))
    dataset = DatasetResult(
        config=DatasetConfig(samples=40),
        points=[DatasetPoint(x=0.2, y=0.4, label=0)],
    )

    payload = compute_forward_trace(model, dataset, 0).model_dump()

    assert "weight" not in str(payload)
    assert "bias" not in str(payload)
    assert len(payload["layers"][0]["activations"]) == 4
