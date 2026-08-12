"""
main.py
-------
FastAPI application exposing the Phase 1 fraud detection model.

Endpoints
    GET  /            service description
    GET  /health      service status + whether the model is really loaded
    GET  /model-info  metadata about the served pipeline
    POST /predict     score one transaction
    GET  /docs        interactive Swagger UI (provided by FastAPI)

The model is loaded once during application startup via the lifespan hook.
If loading fails the app still starts, but /health answers 503 and /predict
answers 503 as well - we would rather return an explicit error than serve
predictions from a model that is not there.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from app import __version__
from app.model_service import ModelNotLoadedError, model_service
from app.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    RootResponse,
    TransactionRequest,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fraud-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup and keep it for the process lifetime."""
    logger.info("Loading model from %s ...", model_service.model_path)
    if model_service.load():
        logger.info(
            "Model loaded (%d features, sha256=%s).",
            len(model_service.feature_order),
            (model_service.model_sha256 or "")[:12],
        )
    else:
        logger.error("Model failed to load: %s", model_service.load_error)
    yield
    logger.info("Shutting down fraud detection API.")


app = FastAPI(
    title="Credit Card Fraud Detection API",
    description=(
        "Serves the Random Forest pipeline trained in Phase 1 of the MAI201 "
        "MLOps project on the Kaggle/ULB credit card fraud dataset. Send one "
        "transaction (Time, V1-V28, Amount) to /predict and the API returns "
        "the predicted class and the fraud probability.\n\n"
        "Academic demonstration - not a production financial control."
    ),
    version=__version__,
    lifespan=lifespan,
    contact={"name": "MAI201 Group - Abdulraouf Zabalawi, Mohamed Roble, Someyah Balashi"},
)


@app.get("/", response_model=RootResponse, tags=["service"])
def root() -> RootResponse:
    """Basic information about the API."""
    return RootResponse(
        service="Credit Card Fraud Detection API",
        version=__version__,
        description=(
            "MAI201 MLOps Phase 2 - Random Forest fraud classifier served with "
            "FastAPI, containerised with Docker and deployed on Render."
        ),
        docs_url="/docs",
        health_url="/health",
        predict_url="/predict",
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["service"],
    responses={503: {"description": "Model is not loaded; service cannot score."}},
)
def health() -> JSONResponse:
    """Report service health. Only 'healthy' when the model is really loaded."""
    loaded = model_service.is_loaded
    payload = HealthResponse(
        status="healthy" if loaded else "unhealthy",
        model_loaded=loaded,
        model_path=str(model_service.model_path),
        version=__version__,
        detail=None if loaded else model_service.load_error,
    )
    code = status.HTTP_200_OK if loaded else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=code, content=payload.model_dump())


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=["model"],
    responses={503: {"description": "Model is not loaded."}},
)
def model_info() -> ModelInfoResponse:
    """Non-sensitive metadata about the model currently being served."""
    try:
        details = model_service.info()
    except ModelNotLoadedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    return ModelInfoResponse(model_version=__version__, **details)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["model"],
    responses={
        422: {"description": "Request body failed validation."},
        503: {"description": "Model is not loaded."},
    },
)
def predict(transaction: TransactionRequest) -> PredictionResponse:
    """Score a single transaction and return the class and fraud probability."""
    try:
        result = model_service.predict_one(transaction.model_dump())
    except ModelNotLoadedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except Exception as exc:  # noqa: BLE001 - never leak a stack trace to callers
        logger.exception("Unexpected prediction failure")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {type(exc).__name__}",
        ) from exc

    return PredictionResponse(model_version=__version__, **result)
