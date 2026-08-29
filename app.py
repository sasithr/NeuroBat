import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask

from database import init_db

from routes.analysis import analysis_bp
from routes.auth import auth_bp
from routes.pages import pages_bp
from routes.player import player_bp
from services.biomechanics import ENGINE_VERSION

# ============================================================
# ENVIRONMENT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

print(
    "Biomechanics Engine:",
    ENGINE_VERSION,
)
# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

app.secret_key = os.getenv(
    "NEUROBAT_SECRET_KEY",
    "neurobat-dev-secret-key",
)

# Maximum upload size: 200 MB
app.config["MAX_CONTENT_LENGTH"] = (
    200 * 1024 * 1024
)


# ============================================================
# PATHS
# ============================================================

FRONTEND_DIR = BASE_DIR / "frontend"

UPLOADS_DIR = (
    BASE_DIR / ".." / "uploads"
).resolve()

VIDEOS_DIR = (
    BASE_DIR / ".." / "videos"
).resolve()

OUTPUTS_DIR = BASE_DIR / "outputs"


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

for directory in (
    FRONTEND_DIR,
    UPLOADS_DIR,
    VIDEOS_DIR,
    OUTPUTS_DIR,
):

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# MAKE PATHS AVAILABLE TO BLUEPRINTS
# ============================================================

app.config.update({

    "FRONTEND_DIR":
        str(FRONTEND_DIR),

    "UPLOADS_DIR":
        str(UPLOADS_DIR),

    "VIDEOS_DIR":
        str(VIDEOS_DIR),

    "OUTPUTS_DIR":
        str(OUTPUTS_DIR),

})


# ============================================================
# CORS
# ============================================================

@app.after_request
def add_cors_headers(response):
    """
    Add basic CORS headers.

    NeuroBat currently serves the frontend through Flask,
    but these headers remain useful for development/API access.
    """

    response.headers[
        "Access-Control-Allow-Origin"
    ] = "*"

    response.headers[
        "Access-Control-Allow-Methods"
    ] = (
        "GET,POST,PUT,PATCH,DELETE,OPTIONS"
    )

    response.headers[
        "Access-Control-Allow-Headers"
    ] = (
        "Content-Type, Authorization"
    )

    return response


# ============================================================
# REGISTER BLUEPRINTS
# ============================================================

app.register_blueprint(
    auth_bp
)

app.register_blueprint(
    player_bp
)

app.register_blueprint(
    analysis_bp
)

app.register_blueprint(
    pages_bp
)


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

try:

    init_db()

    print(
        "[NeuroBat] PostgreSQL database "
        "initialized successfully."
    )

except Exception as error:

    print(
        "[NeuroBat] Database initialization failed:"
    )

    print(
        error
    )

    raise


# ============================================================
# START DEVELOPMENT SERVER
# ============================================================

if __name__ == "__main__":

    print()

    print(
        "=============================================="
    )

    print(
        " NeuroBat — Biomechanics Analysis Engine V2.1"
    )

    print(
        "=============================================="
    )

    print(
        "Frontend:",
        FRONTEND_DIR,
    )

    print(
        "Uploads:",
        UPLOADS_DIR,
    )

    print(
        "Videos:",
        VIDEOS_DIR,
    )

    print(
        "Outputs:",
        OUTPUTS_DIR,
    )

    print(
        "Database: PostgreSQL / neurobat"
    )

    print(
        "Server: http://127.0.0.1:5000"
    )

    print(
        "=============================================="
    )

    print()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False,
    )