"""Synchronous experiment API."""

from fastapi import APIRouter

from neuronscope.experiments import ExperimentRequest, ExperimentResponse, run_experiment

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.post("", response_model=ExperimentResponse)
def create_experiment(request: ExperimentRequest) -> ExperimentResponse:
    """Generate data, build a model, and execute one bounded training run."""

    return run_experiment(request)
