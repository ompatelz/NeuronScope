from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["system"])


class HealthResponse(BaseModel):
    status: Literal["ok"]


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Report that the API process is ready to receive requests."""
    return HealthResponse(status="ok")
