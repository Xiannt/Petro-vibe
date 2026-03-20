from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router
from app.core.settings import Settings, get_settings
from app.orchestrator.query_service import QueryService
from app.utils.logging import configure_logging


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory used by uvicorn and tests."""

    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings.log_level)
    application = FastAPI(
        title=resolved_settings.app_name,
        version=resolved_settings.app_version,
    )
    application.state.settings = resolved_settings
    application.state.query_service = QueryService(resolved_settings)
    application.include_router(router)
    return application


app = create_app()
