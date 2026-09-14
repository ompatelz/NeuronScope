"""API and service tests for complete synchronous experiments."""

from fastapi.testclient import TestClient

from neuronscope.datasets import DatasetConfig, DatasetKind
from neuronscope.experiments import ExperimentRequest, run_experiment
from neuronscope.main import create_app
from neuronscope.models import MLPConfig
from neuronscope.training import TrainingConfig


def test_service_composes_a_complete_experiment_without_http() -> None:
    response = run_experiment(
        ExperimentRequest(
            dataset=DatasetConfig(kind=DatasetKind.XOR, samples=40, seed=3),
            model=MLPConfig(hidden_layers=(4,), seed=3),
            training=TrainingConfig(epochs=2),
        )
    )

    assert len(response.dataset.points) == 40
    assert response.architecture.hidden_layers == (4,)
    assert len(response.training.history) == 2
    assert response.boundary.resolution == 48
    assert len(response.boundary.probabilities) == 48 * 48
    assert response.diagnostics == ()
    assert response.playback.snapshots[0].epoch == 1
    assert response.playback.snapshots[-1].epoch == 2


def test_experiment_endpoint_serializes_metrics_and_instrumentation() -> None:
    response = TestClient(create_app()).post(
        "/api/v1/experiments",
        json={
            "dataset": {"kind": "two_moons", "samples": 40, "noise": 0.08, "seed": 5},
            "model": {
                "input_size": 2,
                "hidden_layers": [4],
                "output_size": 1,
                "activation": "relu",
                "initialization": "he",
                "seed": 5,
            },
            "training": {
                "optimizer": "adam",
                "learning_rate": 0.02,
                "epochs": 2,
                "instrumentation": True,
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["dataset"]["points"]) == 40
    assert body["architecture"]["hidden_layers"] == [4]
    assert body["architecture"]["layers"][0]["name"] == "hidden_0"
    assert len(body["training"]["history"]) == 2
    assert set(body["training"]["history"][0]) == {"epoch", "loss", "accuracy"}
    assert len(body["training"]["instrumentation"]) == 2
    layer = body["training"]["instrumentation"][0]["layers"][0]
    assert set(layer) == {"layer_name", "activation", "weight_norm", "gradients"}
    assert layer["activation"] is not None
    assert layer["gradients"]["norm"] is not None
    assert body["boundary"]["resolution"] == 48
    assert len(body["boundary"]["probabilities"]) == 48 * 48
    assert isinstance(body["diagnostics"], list)
    assert body["playback"]["snapshots"][-1]["epoch"] == 2


def test_experiment_endpoint_rejects_invalid_nested_configurations() -> None:
    client = TestClient(create_app())
    valid = {
        "dataset": {"kind": "xor", "samples": 40},
        "model": {"input_size": 2, "hidden_layers": [4], "output_size": 1},
        "training": {"epochs": 2},
    }

    invalid_dataset = client.post(
        "/api/v1/experiments",
        json={**valid, "dataset": {"kind": "xor", "samples": 3}},
    )
    invalid_model = client.post(
        "/api/v1/experiments",
        json={**valid, "model": {"input_size": 3, "hidden_layers": [4], "output_size": 1}},
    )
    invalid_training = client.post(
        "/api/v1/experiments",
        json={**valid, "training": {"epochs": 0}},
    )

    assert invalid_dataset.status_code == 422
    assert invalid_model.status_code == 422
    assert invalid_training.status_code == 422
