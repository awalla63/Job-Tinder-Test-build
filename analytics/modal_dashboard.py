"""Modal deployment for the Streamlit data dashboard.

Deploy:   modal deploy analytics/modal_dashboard.py
Dev:      modal serve analytics/modal_dashboard.py

Runs `streamlit run` as a web server inside the container. It reuses the API's secret
(Supabase_Secret_Job_Tinder) for DATABASE_URL; the dashboard sets its own session
read-only, and only ever runs aggregate queries over the `analytics` schema.
"""
from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import modal

LOCAL_SCRIPT = Path(__file__).parent / "dashboard.py"
REMOTE_SCRIPT = "/root/dashboard.py"
PORT = 8000

image = (
    modal.Image.debian_slim(python_version="3.12")
    # Pinned to the versions the dashboard was tested against locally.
    .pip_install("streamlit==1.45.1", "altair==5.5.0", "pandas==2.2.3",
                 "psycopg[binary]==3.3.6")
    .add_local_file(LOCAL_SCRIPT, REMOTE_SCRIPT)
)

app = modal.App("job-tinder-dashboard", image=image)


@app.function(
    secrets=[modal.Secret.from_name("Supabase_Secret_Job_Tinder")],
    # One container is plenty for a course dashboard, and keeps it to one DB connection.
    max_containers=1,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=50)
@modal.web_server(PORT, startup_timeout=120)
def dashboard():
    cmd = (
        f"streamlit run {shlex.quote(REMOTE_SCRIPT)} --server.port {PORT} "
        "--server.address 0.0.0.0 --server.headless true "
        # Modal's proxy terminates the connection, so Streamlit's own origin checks
        # would reject the websocket.
        "--server.enableCORS false --server.enableXsrfProtection false "
        "--browser.gatherUsageStats false"
    )
    subprocess.Popen(cmd, shell=True)
