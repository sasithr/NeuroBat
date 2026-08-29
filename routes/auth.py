"""
Authentication routes for NeuroBat.

Handles:

- Login
- Registration
- Logout
- Current signed-in user (/api/me)

User accounts are stored permanently in PostgreSQL.
"""

from flask import (
    Blueprint,
    jsonify,
    redirect,
    request,
    session,
)

from werkzeug.security import (
    check_password_hash,
    generate_password_hash,
)

from database import (
    create_user,
    get_logged_in_user,
    get_player_profile,
    get_user_by_email,
    update_last_login,
)

from routes.pages import serve_frontend_file


# ============================================================
# AUTHENTICATION BLUEPRINT
# ============================================================

auth_bp = Blueprint(
    "auth",
    __name__,
)


# ============================================================
# CURRENT LOGGED-IN USER
# ============================================================

@auth_bp.route(
    "/api/me",
    methods=["GET"],
)
def api_me():
    """
    Return information about the currently logged-in user.

    Used by the frontend to determine whether a user
    has an active authenticated session.
    """

    user = get_logged_in_user()

    if not user:
        return jsonify(
            {
                "error": "Unauthorized."
            }
        ), 401

    profile = get_player_profile(
        user["email"]
    )

    return jsonify(
        {
            "user": {
                "id": user["id"],
                "full_name": user["full_name"],
                "email": user["email"],
            },
            "player_profile": profile,
        }
    ), 200


# ============================================================
# LOGIN
# ============================================================

@auth_bp.route(
    "/login",
    methods=["GET", "POST"],
)
def login_page():
    """
    GET:
        Display the NeuroBat login page.

    POST:
        Authenticate the user against PostgreSQL.
    """

    # --------------------------------------------------------
    # SHOW LOGIN PAGE
    # --------------------------------------------------------

    if request.method == "GET":

        if get_logged_in_user():
            return redirect("/")

        return serve_frontend_file(
            "login.html"
        )

    # --------------------------------------------------------
    # READ LOGIN FORM
    # --------------------------------------------------------

    email = request.form.get(
        "email",
        "",
    ).strip().lower()

    password = request.form.get(
        "password",
        "",
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not email or not password:

        return jsonify(
            {
                "error":
                    "Email and password are required."
            }
        ), 400

    # --------------------------------------------------------
    # FIND USER IN POSTGRESQL
    # --------------------------------------------------------

    user = get_user_by_email(
        email
    )

    # --------------------------------------------------------
    # VERIFY PASSWORD
    # --------------------------------------------------------

    if (
        not user
        or not check_password_hash(
            user["password_hash"],
            password,
        )
    ):

        return jsonify(
            {
                "error":
                    "Invalid email or password."
            }
        ), 401

    # --------------------------------------------------------
    # CREATE FLASK SESSION
    # --------------------------------------------------------

    session.clear()

    session["user_email"] = user["email"]

    session["user_full_name"] = user[
        "full_name"
    ]

    # --------------------------------------------------------
    # UPDATE LAST LOGIN
    # --------------------------------------------------------

    update_last_login(
        user["email"]
    )

    # --------------------------------------------------------
    # LOGIN SUCCESS
    # --------------------------------------------------------

    return jsonify(
        {
            "message": "Login successful.",
            "redirect": "/",
        }
    ), 200


# ============================================================
# REGISTRATION
# ============================================================

@auth_bp.route(
    "/register",
    methods=["GET", "POST"],
)
def register_page():
    """
    GET:
        Display the NeuroBat registration page.

    POST:
        Create a permanent user account in PostgreSQL.
    """

    # --------------------------------------------------------
    # SHOW REGISTRATION PAGE
    # --------------------------------------------------------

    if request.method == "GET":

        if get_logged_in_user():
            return redirect("/")

        return serve_frontend_file(
            "register.html"
        )

    # --------------------------------------------------------
    # READ REGISTRATION FORM
    # --------------------------------------------------------

    full_name = request.form.get(
        "full_name",
        "",
    ).strip()

    email = request.form.get(
        "email",
        "",
    ).strip().lower()

    password = request.form.get(
        "password",
        "",
    )

    confirm_password = request.form.get(
        "confirm_password",
        "",
    )

    # --------------------------------------------------------
    # REQUIRED FIELD VALIDATION
    # --------------------------------------------------------

    if (
        not full_name
        or not email
        or not password
        or not confirm_password
    ):

        return jsonify(
            {
                "error":
                    "All fields are required."
            }
        ), 400

    # --------------------------------------------------------
    # PASSWORD MATCH
    # --------------------------------------------------------

    if password != confirm_password:

        return jsonify(
            {
                "error":
                    "Passwords do not match."
            }
        ), 400

    # --------------------------------------------------------
    # PASSWORD LENGTH
    # --------------------------------------------------------

    if len(password) < 8:

        return jsonify(
            {
                "error":
                    "Password must be at least 8 characters."
            }
        ), 400

    # --------------------------------------------------------
    # CHECK EXISTING ACCOUNT
    # --------------------------------------------------------

    existing_user = get_user_by_email(
        email
    )

    if existing_user:

        return jsonify(
            {
                "error":
                    "An account with this email already exists."
            }
        ), 409

    # --------------------------------------------------------
    # HASH PASSWORD
    # --------------------------------------------------------

    password_hash = generate_password_hash(
        password
    )

    # --------------------------------------------------------
    # CREATE USER IN POSTGRESQL
    # --------------------------------------------------------

    user = create_user(
        full_name=full_name,
        email=email,
        password_hash=password_hash,
    )

    # create_user() can return None if another request creates
    # the same email between our check and INSERT.
    if not user:

        return jsonify(
            {
                "error":
                    "An account with this email already exists."
            }
        ), 409

    # --------------------------------------------------------
    # CREATE LOGIN SESSION
    # --------------------------------------------------------

    session.clear()

    session["user_email"] = user["email"]

    session["user_full_name"] = user[
        "full_name"
    ]

    # --------------------------------------------------------
    # REGISTRATION SUCCESS
    # --------------------------------------------------------

    return jsonify(
        {
            "message":
                "Account created successfully.",

            "redirect":
                "/player-info",
        }
    ), 201


# ============================================================
# LOGOUT
# ============================================================

@auth_bp.route(
    "/logout",
    methods=["GET", "POST"],
)
def logout_page():
    """
    End the current NeuroBat session.
    """

    session.clear()

    return redirect("/")
