"""Synchronous experiment API."""

from threading import Lock

from fastapi import APIRouter, HTTPException, status

from neuronscope.experiments import ExperimentRequest, ExperimentResponse, run_experiment

router = APIRouter(prefix="/experiments", tags=["experiments"])
_experiment_slot = Lock()


@router.post("", response_model=ExperimentResponse)
def create_experiment(request: ExperimentRequest) -> ExperimentResponse:
    """Generate data, build a model, and execute one bounded training run."""

    if not _experiment_slot.acquire(blocking=False):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Another experiment is already running. Retry after it completes.",
            headers={"Retry-After": "1"},
        )
    try:
        return run_experiment(request)
    finally:
        _experiment_slot.release()
