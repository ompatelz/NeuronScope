from pathlib import Path

from fastapi.testclient import TestClient

from neuronscope.core.config import Settings
from neuronscope.main import create_app


def test_health_endpoint_reports_ready() -> None:
    app = create_app(Settings())

    response = TestClient(app).get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_built_frontend_can_be_served_from_the_api(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    (tmp_path / "index.html").write_text("<h1>NeuronScope</h1>", encoding="utf-8")
    (assets / "app.js").write_text("console.log('ready')", encoding="utf-8")

    client = TestClient(create_app(Settings(frontend_dist=tmp_path)))

    assert client.get("/").text == "<h1>NeuronScope</h1>"
    assert client.get("/assets/app.js").text == "console.log('ready')"
    assert client.get("/api/v1/health").json() == {"status": "ok"}
    assert client.get("/").headers["cache-control"] == "no-cache"
    assert "default-src 'self'" in client.get("/").headers["content-security-policy"]
    assert client.get("/assets/app.js").headers["cache-control"].endswith("immutable")
