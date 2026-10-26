"""Runtime settings, loaded from the repo-root .env."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")


class Settings:
    def __init__(self) -> None:
        self.supabase_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
        self.project_ref = os.environ.get("SUPABASE_PROJECT_REF", "")
        self.service_role_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
        self.database_url = os.environ.get("DATABASE_URL", "")
        self.fit_model_path = REPO_ROOT / os.environ.get("FIT_MODEL_PATH", "models/fit_model.txt")
        self.fit_artifacts_path = self.fit_model_path.parent / "fit_artifacts.joblib"
        self.cors_origins = [
            o.strip() for o in os.environ.get(
                "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
            ).split(",") if o.strip()
        ]
        # Candidates pulled from the vector index before LightGBM reranks them. 200 is
        # well inside the "cheap" range (1-5 ms to score) and gives the reranker enough
        # room to meaningfully reorder.
        self.ann_candidates = int(os.environ.get("ANN_CANDIDATES", "200"))
        self.dev_mode = os.environ.get("DEV_MODE", "").lower() in {"1", "true", "yes"}

    @property
    def jwks_url(self) -> str:
        base = self.supabase_url or f"https://{self.project_ref}.supabase.co"
        return f"{base}/auth/v1/.well-known/jwks.json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
