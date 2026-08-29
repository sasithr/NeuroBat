"""
Public/private HTML page routes for NeuroBat.
"""

import os

from flask import (
    Blueprint,
    current_app,
    jsonify,
    send_from_directory,
)

from database import get_logged_in_user


pages_bp = Blueprint(
    "pages",
    __name__,
)


def serve_frontend_file(filename):

    frontend_dir = current_app.config[
        "FRONTEND_DIR"
    ]

    file_path = os.path.join(
        frontend_dir,
        filename,
    )

    if os.path.isfile(file_path):

        return send_from_directory(
            frontend_dir,
            filename,
        )

    return send_from_directory(
        frontend_dir,
        "landingpage.html",
    )


@pages_bp.route("/")
def home():
    return serve_frontend_file(
        "landingpage.html"
    )


@pages_bp.route("/health")
def health():
    return jsonify({
        "status": "running",
        "project": "NeuroBat",
        "version": "2.1",
        "analysis_engine": "Biomechanics V2.1",
    })


@pages_bp.route("/pricing")
def pricing_page():
    return serve_frontend_file(
        "pricing.html"
    )


@pages_bp.route("/contact")
def contact_page():
    return serve_frontend_file(
        "contact.html"
    )


@pages_bp.route("/shots")
def shots_page():
    return serve_frontend_file(
        "shots.html"
    )


@pages_bp.route("/features")
def features_page():

    frontend_dir = current_app.config[
        "FRONTEND_DIR"
    ]

    feature_file = os.path.join(
        frontend_dir,
        "feature.html",
    )

    if os.path.isfile(feature_file):

        return send_from_directory(
            frontend_dir,
            "feature.html",
        )

    return serve_frontend_file(
        "features.html"
    )


@pages_bp.route("/analytics")
def analytics_page():
    return serve_frontend_file(
        "analytics.html"
    )


@pages_bp.route("/analyze-page")
def analyze_page():
    return serve_frontend_file(
        "analyze.html"
    )


@pages_bp.route("/dashboard")
def dashboard_page():

    if not get_logged_in_user():

        return jsonify({
            "error": "Unauthorized."
        }), 401

    return serve_frontend_file(
        "dashboard.html"
    )


@pages_bp.route("/history")
def history_page():

    if not get_logged_in_user():

        return jsonify({
            "error":
                "You must be logged in to view analysis history."
        }), 401

    return serve_frontend_file(
        "history.html"
    )


@pages_bp.route("/progress")
def progress_page():

    if not get_logged_in_user():

        return jsonify({
            "error":
                "You must be logged in to view progress."
        }), 401

    return serve_frontend_file(
        "progress.html"
    )


@pages_bp.route("/comparison")
def comparison_page():

    if not get_logged_in_user():

        return jsonify({
            "error":
                "You must be logged in to compare sessions."
        }), 401

    return serve_frontend_file(
        "comparison.html"
    )


@pages_bp.route("/analysis/<int:analysis_id>")
def analysis_detail_page(analysis_id):

    if not get_logged_in_user():

        return jsonify({
            "error":
                "You must be logged in to view this analysis."
        }), 401

    return serve_frontend_file(
        "analysis-detail.html"
    )


# Keep this last inside this blueprint.
@pages_bp.route("/<path:filename>")
def static_files(filename):

    frontend_dir = current_app.config[
        "FRONTEND_DIR"
    ]

    requested_path = os.path.join(
        frontend_dir,
        filename,
    )

    if os.path.isfile(
        requested_path
    ):

        return send_from_directory(
            frontend_dir,
            filename,
        )

    return send_from_directory(
        frontend_dir,
        "landingpage.html",
    )