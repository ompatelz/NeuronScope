from fastapi.testclient import TestClient

from neuronscope.core.config import Settings
from neuronscope.main import create_app


def test_health_endpoint_reports_ready() -> None:
    app = create_app(Settings())

    response = TestClient(app).get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
