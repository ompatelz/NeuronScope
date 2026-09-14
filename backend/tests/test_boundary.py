"""Tests for real final-model decision-boundary inference."""

import pytest
from pydantic import ValidationError

from neuronscope.datasets import DatasetConfig, DatasetKind, generate_dataset
from neuronscope.experiments import (
    DecisionBoundaryConfig,
    ExperimentRequest,
    compute_decision_boundary,
    run_experiment,
)
from neuronscope.models import MLPConfig, build_mlp
from neuronscope.training import TrainingConfig, train_model


def test_boundary_grid_has_bounded_shape_bounds_and_probabilities() -> None:
    dataset = generate_dataset(DatasetConfig(kind=DatasetKind.XOR, samples=40, seed=4))
    model = build_mlp(MLPConfig(hidden_layers=(4,), seed=4))
    train_model(model, dataset, TrainingConfig(epochs=2))

    boundary = compute_decision_boundary(model, dataset, resolution=24)

    assert boundary.resolution == 24
    assert len(boundary.x_coordinates) == 24
    assert len(boundary.y_coordinates) == 24
    assert len(boundary.probabilities) == 24 * 24
    assert boundary.x_coordinates[0] < min(point.x for point in dataset.points)
    assert boundary.x_coordinates[-1] > max(point.x for point in dataset.points)
    assert boundary.y_coordinates[0] < min(point.y for point in dataset.points)
    assert boundary.y_coordinates[-1] > max(point.y for point in dataset.points)
    assert all(0.0 <= probability <= 1.0 for probability in boundary.probabilities)


def test_experiment_boundary_is_deterministic_for_equal_seeded_requests() -> None:
    request = ExperimentRequest(
        dataset=DatasetConfig(kind=DatasetKind.CIRCLES, samples=40, seed=9),
        model=MLPConfig(hidden_layers=(4,), seed=9),
        training=TrainingConfig(epochs=2),
        boundary=DecisionBoundaryConfig(resolution=24),
    )

    assert run_experiment(request).boundary == run_experiment(request).boundary


@pytest.mark.parametrize("resolution", [23, 81])
def test_boundary_resolution_is_bounded(resolution: int) -> None:
    with pytest.raises(ValidationError):
        DecisionBoundaryConfig(resolution=resolution)
