"""Tests for deterministic full-batch binary-classification training."""

import pytest
from pydantic import ValidationError

from neuronscope.datasets import DatasetConfig, DatasetKind, generate_dataset
from neuronscope.models import ActivationName, InitializationName, MLPConfig, build_mlp
from neuronscope.training import OptimizerName, TrainingConfig, train_model


@pytest.mark.parametrize("optimizer", list(OptimizerName))
def test_training_returns_bounded_metrics_for_each_epoch(optimizer: OptimizerName) -> None:
    dataset = generate_dataset(DatasetConfig(kind=DatasetKind.XOR, samples=80, seed=4))
    model = build_mlp(MLPConfig(hidden_layers=(8,), seed=5))
    config = TrainingConfig(optimizer=optimizer, learning_rate=0.02, epochs=4)

    result = train_model(model, dataset, config)

    assert result.config == config
    assert [metric.epoch for metric in result.history] == [1, 2, 3, 4]
    assert all(metric.loss >= 0 for metric in result.history)
    assert all(0 <= metric.accuracy <= 1 for metric in result.history)
    assert result.final_loss == result.history[-1].loss
    assert result.final_accuracy == result.history[-1].accuracy


def test_adam_training_learns_two_moons_without_asserting_an_exact_history() -> None:
    dataset = generate_dataset(
        DatasetConfig(kind=DatasetKind.TWO_MOONS, samples=200, noise=0.08, seed=7)
    )
    model = build_mlp(
        MLPConfig(
            hidden_layers=(16, 16),
            activation=ActivationName.RELU,
            initialization=InitializationName.HE,
            seed=7,
        )
    )

    result = train_model(
        model,
        dataset,
        TrainingConfig(optimizer=OptimizerName.ADAM, learning_rate=0.02, epochs=150),
    )

    assert result.final_loss < result.history[0].loss * 0.5
    assert result.final_accuracy >= 0.9


def test_equivalent_runs_are_deterministic() -> None:
    dataset = generate_dataset(DatasetConfig(kind=DatasetKind.CIRCLES, samples=80, seed=12))
    model_config = MLPConfig(hidden_layers=(6,), seed=12)
    training_config = TrainingConfig(learning_rate=0.01, epochs=5)

    first = train_model(build_mlp(model_config), dataset, training_config)
    second = train_model(build_mlp(model_config), dataset, training_config)

    assert first == second


@pytest.mark.parametrize(
    "values",
    [
        {"optimizer": "rmsprop"},
        {"learning_rate": 0},
        {"learning_rate": 1.01},
        {"epochs": 0},
        {"epochs": 5_001},
    ],
)
def test_invalid_training_configuration_is_rejected(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(values)


@pytest.mark.parametrize(
    "model_config",
    [MLPConfig(input_size=3), MLPConfig(output_size=2)],
)
def test_training_rejects_models_incompatible_with_generated_datasets(
    model_config: MLPConfig,
) -> None:
    dataset = generate_dataset(DatasetConfig(samples=40))

    with pytest.raises(ValueError, match="input_size=2 and output_size=1"):
        train_model(build_mlp(model_config), dataset, TrainingConfig(epochs=1))
