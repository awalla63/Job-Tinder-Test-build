"""Job Tinder API.

The embedding model and the LightGBM booster are loaded once in the lifespan context
and held on app.state. `@app.on_event("startup")` is deprecated and does not compose
with a lifespan, so everything goes through the context manager.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.api.config import get_settings
from services.api.database import close_pool, init_pool
from services.api.routers import matching, onboarding
from services.api.scoring import Scorer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("jobtinder")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    log.info("Opening connection pool")
    init_pool()

    log.info("Loading encoder (first run downloads the model)")
    from ml.encoder import get_encoder

    app.state.encoder = get_encoder()

    scorer = Scorer()
    scorer.load()
    app.state.scorer = scorer
    if not scorer.ready:
        log.warning("Serving without a fit model: ranking falls back to cosine similarity.")

    log.info("Ready. ANN candidate pool = %d", settings.ann_candidates)
    yield

    close_pool()


app = FastAPI(title="Job Tinder API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(onboarding.router)
app.include_router(matching.router)


@app.get("/health")
def health() -> dict:
    scorer = getattr(app.state, "scorer", None)
    return {
        "status": "ok",
        "model_loaded": bool(scorer and scorer.ready),
        "features": (scorer.feature_names if scorer and scorer.ready else []),
    }
