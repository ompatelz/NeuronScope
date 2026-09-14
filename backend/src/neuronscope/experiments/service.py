"""Application service composing dataset, model, and training domains."""

from neuronscope.datasets import generate_dataset
from neuronscope.diagnostics import evaluate_diagnostics
from neuronscope.experiments.boundary import compute_decision_boundary
from neuronscope.experiments.schemas import (
    ExperimentRequest,
    ExperimentResponse,
    PlaybackResult,
    PlaybackSnapshot,
)
from neuronscope.models import build_mlp
from neuronscope.training import select_snapshot_epochs, train_model_with_snapshots


def run_experiment(request: ExperimentRequest) -> ExperimentResponse:
    """Run one bounded experiment synchronously without depending on HTTP."""

    dataset = generate_dataset(request.dataset)
    model = build_mlp(request.model)
    snapshot_epochs = select_snapshot_epochs(
        request.training.epochs, request.playback.max_snapshots
    )
    run = train_model_with_snapshots(model, dataset, request.training, snapshot_epochs)
    training = run.result
    final_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    metrics = {item.epoch: item for item in training.history}
    instrumentation = {item.epoch: item for item in training.instrumentation}
    snapshots: list[PlaybackSnapshot] = []
    first_boundary = None
    try:
        for epoch in snapshot_epochs:
            model.load_state_dict(run.states[epoch])
            observed = compute_decision_boundary(model, dataset, request.boundary.resolution)
            first_boundary = first_boundary or observed
            snapshots.append(
                PlaybackSnapshot(
                    epoch=epoch,
                    metrics=metrics[epoch],
                    instrumentation=instrumentation.get(epoch),
                    probabilities=observed.probabilities,
                )
            )
    finally:
        model.load_state_dict(final_state)
    boundary = compute_decision_boundary(model, dataset, request.boundary.resolution)
    assert first_boundary is not None
    playback = PlaybackResult(
        resolution=first_boundary.resolution,
        x_coordinates=first_boundary.x_coordinates,
        y_coordinates=first_boundary.y_coordinates,
        snapshots=tuple(snapshots),
    )
    diagnostics = evaluate_diagnostics(
        training.instrumentation, request.diagnostics, request.model.activation
    )
    return ExperimentResponse(
        dataset=dataset,
        architecture=model.architecture,
        training=training,
        boundary=boundary,
        diagnostics=diagnostics,
        playback=playback,
    )
