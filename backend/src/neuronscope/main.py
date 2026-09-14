from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

from neuronscope.api.routes.datasets import router as datasets_router
from neuronscope.api.routes.experiments import router as experiments_router
from neuronscope.api.routes.health import router as health_router
from neuronscope.core.config import Settings, get_settings
from neuronscope.core.http import SecurityHeadersMiddleware


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the HTTP application without initializing training resources."""
    resolved_settings = settings or get_settings()
    app = FastAPI(title=resolved_settings.app_name, version="0.1.0")
    app.add_middleware(GZipMiddleware, minimum_size=512)
    app.add_middleware(SecurityHeadersMiddleware)
    app.include_router(health_router, prefix=resolved_settings.api_v1_prefix)
    app.include_router(datasets_router, prefix=resolved_settings.api_v1_prefix)
    app.include_router(experiments_router, prefix=resolved_settings.api_v1_prefix)
    if resolved_settings.frontend_dist is not None:
        app.mount(
            "/",
            StaticFiles(directory=resolved_settings.frontend_dist, html=True),
            name="frontend",
        )
    return app


app = create_app()
