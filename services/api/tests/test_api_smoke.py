"""HTTP-layer smoke tests.

Runs the real ASGI app with the database pool and encoder stubbed out, so the routing,
dependency wiring and auth rejection paths are exercised without needing a live Supabase
project. Anything that actually touches Postgres belongs in an integration test.
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from services.api import database, main
from services.api.auth import CurrentUser, get_current_user


class StubEncoder:
    def encode_resumes(self, texts, batch_size=32):
        return np.tile(np.eye(1, 384, dtype=np.float32), (len(texts), 1))

    def encode_jobs(self, texts, batch_size=32):
        return np.tile(np.eye(1, 384, dtype=np.float32), (len(texts), 1))


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(database, "init_pool", lambda: None)
    monkeypatch.setattr(database, "close_pool", lambda: None)
    monkeypatch.setattr(main, "init_pool", lambda: None)
    monkeypatch.setattr(main, "close_pool", lambda: None)
    monkeypatch.setattr("ml.encoder.get_encoder", lambda: StubEncoder())
    with TestClient(main.app) as test_client:
        yield test_client


def test_health_reports_model_state(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    # models/ is populated by ml/train.py; either state is valid, but the shape is fixed.
    assert isinstance(body["model_loaded"], bool)
    assert isinstance(body["features"], list)


@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/me"),
        ("get", "/deck/jobs"),
        ("get", "/deck/candidates?job_id=abc"),
        ("get", "/matches"),
        ("get", "/employer/jobs"),
        ("post", "/swipe"),
        ("post", "/survey"),
        ("post", "/auth/complete-profile"),
    ],
)
def test_every_data_route_requires_a_token(client, method, path):
    """No endpoint that touches user data may be reachable anonymously."""
    response = getattr(client, method)(path, **({"json": {}} if method == "post" else {}))
    assert response.status_code in (401, 403), f"{method.upper()} {path} was not protected"


def test_malformed_bearer_token_is_rejected(client):
    response = client.get("/me", headers={"Authorization": "Bearer not-a-real-jwt"})
    assert response.status_code == 401


def test_complete_profile_rejects_an_unknown_role(client, monkeypatch):
    monkeypatch.setitem(
        main.app.dependency_overrides, get_current_user,
        lambda: CurrentUser("user-1", "a@b.c", {}),
    )
    response = client.post(
        "/auth/complete-profile", json={"role": "recruiter", "display_name": "X"}
    )
    assert response.status_code == 422
    main.app.dependency_overrides.clear()


def test_employer_signup_requires_a_company_name(client, monkeypatch):
    monkeypatch.setitem(
        main.app.dependency_overrides, get_current_user,
        lambda: CurrentUser("user-1", "a@b.c", {}),
    )
    response = client.post(
        "/auth/complete-profile",
        json={"role": "employer", "display_name": "X", "company_name": "  "},
    )
    assert response.status_code == 400
    main.app.dependency_overrides.clear()


def test_deck_limit_is_bounded(client, monkeypatch):
    """limit is capped so one request cannot ask the scorer for the whole corpus."""
    monkeypatch.setitem(
        main.app.dependency_overrides, get_current_user,
        lambda: CurrentUser("user-1", "a@b.c", {}),
    )
    response = client.get("/deck/jobs?limit=5000")
    assert response.status_code == 422
    main.app.dependency_overrides.clear()
