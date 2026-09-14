"""Tests for bounded, removable model instrumentation."""

import math

import pytest
import torch

from neuronscope.datasets import DatasetConfig, DatasetKind, generate_dataset
from neuronscope.instrumentation import (
    TrainingInstrumentationCollector,
    summarize_activations,
)
from neuronscope.models import MLPConfig, build_mlp
from neuronscope.training import TrainingConfig, train_model


def test_activation_statistics_include_zeros_and_nonfinite_values() -> None:
    values = torch.tensor([0.0, 1.0, -1.0, float("nan"), float("inf")])

    summary = summarize_activations(values)

    assert summary.mean == pytest.approx(0.0)
    assert summary.std == pytest.approx(math.sqrt(2.0 / 3.0))
    assert summary.minimum == -1.0
    assert summary.maximum == 1.0
    assert summary.zero_percentage == 20.0
    assert summary.nonfinite_count == 2


def test_collector_captures_stable_layer_names_and_removes_hooks() -> None:
    model = build_mlp(MLPConfig(hidden_layers=(4, 3), seed=2))
    collector = TrainingInstrumentationCollector(model)
    collector.attach()
    collector.begin_training_pass()

    model(torch.ones((4, 2))).sum().backward()
    summary = collector.capture_epoch(1)

    assert [layer.layer_name for layer in summary.layers] == [
        "hidden_0",
        "hidden_1",
        "output",
    ]
    assert all(layer.weight_norm is not None for layer in summary.layers)
    assert all(layer.gradients.norm is not None for layer in summary.layers)
    assert summary.layers[0].activation is not None
    assert summary.layers[-1].activation is None

    collector.remove()
    assert not collector.attached
    assert all(len(module._forward_hooks) == 0 for module in model.hidden_activations)


def test_repeated_instrumented_training_does_not_accumulate_hooks() -> None:
    dataset = generate_dataset(DatasetConfig(kind=DatasetKind.XOR, samples=40, seed=3))
    model = build_mlp(MLPConfig(hidden_layers=(4,), seed=3))
    config = TrainingConfig(epochs=2, instrumentation=True)

    first = train_model(model, dataset, config)
    second = train_model(model, dataset, config)

    assert len(first.instrumentation) == 2
    assert len(second.instrumentation) == 2
    assert all(len(module._forward_hooks) == 0 for module in model.hidden_activations)


def test_training_instrumentation_is_optional_and_matches_epochs() -> None:
    dataset = generate_dataset(DatasetConfig(samples=40, seed=8))

    plain = train_model(
        build_mlp(MLPConfig(hidden_layers=(4,), seed=8)),
        dataset,
        TrainingConfig(epochs=2),
    )
    instrumented = train_model(
        build_mlp(MLPConfig(hidden_layers=(4,), seed=8)),
        dataset,
        TrainingConfig(epochs=3, instrumentation=True),
    )

    assert plain.instrumentation == ()
    assert [summary.epoch for summary in instrumented.instrumentation] == [1, 2, 3]
    assert all(summary.layers[0].activation is not None for summary in instrumented.instrumentation)
