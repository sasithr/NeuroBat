"""
Player profile routes for NeuroBat.
"""

from flask import (
    Blueprint,
    jsonify,
    request,
)

from database import (
    get_logged_in_user,
    save_player_profile,
)

from routes.pages import serve_frontend_file


player_bp = Blueprint(
    "player",
    __name__,
)


@player_bp.route(
    "/player-info",
    methods=["GET", "POST"],
)
def player_info_page():
    user = get_logged_in_user()

    if not user:
        return jsonify({
            "error": "Unauthorized."
        }), 401

    if request.method == "GET":
        return serve_frontend_file(
            "player_info.html"
        )

    profile = {
        "batting_hand":
            request.form.get(
                "batting_hand",
                "",
            ),

        "player_role":
            request.form.get(
                "player_role",
                "",
            ),

        "experience_level":
            request.form.get(
                "experience_level",
                "",
            ),

        "age_group":
            request.form.get(
                "age_group",
                "",
            ),

        "team_name":
            request.form.get(
                "team_name",
                "",
            ),
    }

    save_player_profile(
        user["email"],
        profile,
    )

    return jsonify({
        "redirect": "/analyze-page"
    })