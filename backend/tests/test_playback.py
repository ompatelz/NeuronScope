"""Tests for bounded, real training playback snapshots."""

from neuronscope.datasets import DatasetConfig, DatasetKind
from neuronscope.experiments import ExperimentRequest, PlaybackConfig, run_experiment
from neuronscope.models import MLPConfig
from neuronscope.training import TrainingConfig, select_snapshot_epochs


def test_snapshot_epoch_policy_is_capped_and_keeps_first_and_last() -> None:
    selected = select_snapshot_epochs(epochs=100, max_snapshots=12)

    assert len(selected) == 12
    assert selected[0] == 1
    assert selected[-1] == 100
    assert tuple(sorted(set(selected))) == selected
    assert select_snapshot_epochs(epochs=3, max_snapshots=12) == (1, 2, 3)


def test_playback_uses_one_fixed_grid_and_real_changing_predictions() -> None:
    response = run_experiment(
        ExperimentRequest(
            dataset=DatasetConfig(kind=DatasetKind.TWO_MOONS, samples=80, noise=0.08, seed=7),
            model=MLPConfig(hidden_layers=(8,), seed=7),
            training=TrainingConfig(epochs=20, learning_rate=0.03, instrumentation=True),
            playback=PlaybackConfig(max_snapshots=5),
        )
    )

    assert [snapshot.epoch for snapshot in response.playback.snapshots][0] == 1
    assert response.playback.snapshots[-1].epoch == 20
    assert len(response.playback.snapshots) == 5
    expected_size = response.playback.resolution**2
    assert all(
        len(snapshot.probabilities) == expected_size for snapshot in response.playback.snapshots
    )
    assert response.playback.x_coordinates == response.boundary.x_coordinates
    assert response.playback.y_coordinates == response.boundary.y_coordinates
    assert response.playback.snapshots[-1].probabilities == response.boundary.probabilities
    assert (
        response.playback.snapshots[0].probabilities
        != response.playback.snapshots[-1].probabilities
    )
    assert all(snapshot.metrics.epoch == snapshot.epoch for snapshot in response.playback.snapshots)
    assert all(snapshot.instrumentation is not None for snapshot in response.playback.snapshots)
    traces = [snapshot.forward_pass for snapshot in response.playback.snapshots]
    assert all(trace.sample_index == 0 for trace in traces)
    assert all(trace.input_values == traces[0].input_values for trace in traces)
    assert all(len(trace.layers[0].activations) == 8 for trace in traces)
    assert traces[0].layers != traces[-1].layers


def test_playback_response_does_not_serialize_model_weights() -> None:
    response = run_experiment(
        ExperimentRequest(
            dataset=DatasetConfig(samples=40, seed=3),
            model=MLPConfig(hidden_layers=(4,), seed=3),
            training=TrainingConfig(epochs=2),
        )
    )

    payload = response.playback.model_dump()
    assert "state_dict" not in str(payload)
    assert "weight" not in str(payload)
