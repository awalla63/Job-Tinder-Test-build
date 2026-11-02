"""Modal deployment for the Job Tinder API.

Deploy:   modal deploy modal_app.py
Dev:      modal serve modal_app.py   (hot-reloading temporary URL)

The FastAPI app is served unchanged: Modal runs its lifespan once per container, so the
encoder, the LightGBM booster and the Postgres pool load at container start exactly as
they do under uvicorn locally.

Secrets come from a Modal Secret named `Supabase_Secret_Job_Tinder`, created once with:

    modal secret create Supabase_Secret_Job_Tinder \
        DATABASE_URL=... SUPABASE_URL=... SUPABASE_PROJECT_REF=... \
        SUPABASE_SERVICE_ROLE_KEY=... CORS_ORIGINS=https://<app>.vercel.app,http://localhost:5173

DATABASE_URL must be the session pooler URL, same as locally. Values in the secret
override the non-secret defaults set on the image below, so CORS_ORIGINS goes there once
the Vercel URL is known -- no redeploy of this file needed, just a secret update.
"""
from __future__ import annotations

import modal

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

image = (
    # 3.12 for the same reason render.yaml pinned it: torch/onnxruntime/lightgbm wheel
    # availability lags on brand-new Python releases.
    modal.Image.debian_slim(python_version="3.12")
    # requirements.txt carries the CPU-only torch index, so this doesn't pull CUDA.
    .pip_install_from_requirements("requirements.txt")
    # Bake the encoder (including its ONNX export) into the image. Otherwise every cold
    # container downloads it from Hugging Face before it can answer a request.
    .run_commands(
        "python -c \"from sentence_transformers import SentenceTransformer; "
        f"SentenceTransformer('{EMBEDDING_MODEL}', backend='onnx')\""
    )
    .env(
        {
            "EMBEDDING_MODEL": EMBEDDING_MODEL,
            "FIT_MODEL_PATH": "models/fit_model.txt",
            "ANN_CANDIDATES": "200",
            "CORS_ORIGINS": "http://localhost:5173",
        }
    )
    # Local code goes last so code-only changes don't rebuild the dependency layers.
    # Both packages land under /root, and config.py resolves REPO_ROOT as two levels up
    # from services/api/, so /root/models is where it looks for the fit model.
    .add_local_python_source("services", "ml")
    .add_local_dir("models", remote_path="/root/models")
)

app = modal.App("job-tinder-api", image=image)


@app.function(
    secrets=[modal.Secret.from_name("Supabase_Secret_Job_Tinder")],
    # Encoder + ONNX runtime + booster sit around 1 GB resident; 2 GB leaves headroom
    # for resume parsing and request bursts.
    memory=2048,
    cpu=1.0,
    # Each container opens its own pool (max 8 connections, see database.py). Two
    # containers keeps the worst case at 16, within what the Supabase session pooler
    # allows on the free tier alongside the occasional ingest/embed job.
    max_containers=2,
    # Stay warm for 5 minutes after the last request so a user clicking through the app
    # pays the model-load cold start once, not on every page.
    scaledown_window=300,
)
# FastAPI runs sync endpoints in a threadpool, so one container can overlap requests
# that are waiting on Postgres.
@modal.concurrent(max_inputs=16)
@modal.asgi_app()
def api():
    from services.api.main import app as fastapi_app

    return fastapi_app
