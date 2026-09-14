"""Decision-boundary inference over a bounded two-dimensional grid."""

import torch

from neuronscope.datasets import DatasetResult
from neuronscope.experiments.schemas import DecisionBoundaryResult
from neuronscope.models import ConfigurableMLP


def compute_decision_boundary(
    model: ConfigurableMLP,
    dataset: DatasetResult,
    resolution: int,
) -> DecisionBoundaryResult:
    """Evaluate the trained model on a row-major grid covering the dataset."""

    if not dataset.points:
        raise ValueError("A decision boundary requires at least one dataset point.")

    x_values = [point.x for point in dataset.points]
    y_values = [point.y for point in dataset.points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_margin = max((x_max - x_min) * 0.08, 0.1)
    y_margin = max((y_max - y_min) * 0.08, 0.1)

    x_coordinates = torch.linspace(x_min - x_margin, x_max + x_margin, resolution)
    y_coordinates = torch.linspace(y_min - y_margin, y_max + y_margin, resolution)
    grid_y, grid_x = torch.meshgrid(y_coordinates, x_coordinates, indexing="ij")
    points = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits = model(points)
            probabilities = torch.sigmoid(logits).reshape(-1)
    finally:
        model.train(was_training)

    if not torch.isfinite(probabilities).all():
        raise FloatingPointError("Decision-boundary inference produced non-finite probabilities.")

    return DecisionBoundaryResult(
        resolution=resolution,
        x_coordinates=tuple(float(value) for value in x_coordinates.tolist()),
        y_coordinates=tuple(float(value) for value in y_coordinates.tolist()),
        probabilities=tuple(float(value) for value in probabilities.tolist()),
    )
