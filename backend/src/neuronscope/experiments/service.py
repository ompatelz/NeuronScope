"""Application service composing dataset, model, and training domains."""

from neuronscope.datasets import generate_dataset
from neuronscope.experiments.boundary import compute_decision_boundary
from neuronscope.experiments.schemas import ExperimentRequest, ExperimentResponse
from neuronscope.models import build_mlp
from neuronscope.training import train_model


def run_experiment(request: ExperimentRequest) -> ExperimentResponse:
    """Run one bounded experiment synchronously without depending on HTTP."""

    dataset = generate_dataset(request.dataset)
    model = build_mlp(request.model)
    training = train_model(model, dataset, request.training)
    boundary = compute_decision_boundary(model, dataset, request.boundary.resolution)
    return ExperimentResponse(
        dataset=dataset,
        architecture=model.architecture,
        training=training,
        boundary=boundary,
    )
