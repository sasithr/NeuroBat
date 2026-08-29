"""
NeuroBat PostgreSQL data-access layer.

This module handles:

- PostgreSQL connection
- Database table initialization
- User registration
- User lookup
- Logged-in user lookup
- Player profile storage and updates

Route files should not contain PostgreSQL-specific logic.
They should call the functions defined in this module.
"""

import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from flask import session
from psycopg.rows import dict_row


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

# Get the folder containing this database.py file
BASE_DIR = Path(__file__).resolve().parent

# Explicitly load backend/.env
load_dotenv(dotenv_path=BASE_DIR / ".env")

# Read PostgreSQL connection string from .env
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is missing. "
        "Create backend/.env and add the PostgreSQL connection string."
    )


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():
    """
    Create and return a PostgreSQL database connection.

    dict_row makes query results behave like Python dictionaries.

    Example:
        user["email"]
        user["full_name"]
    """

    return psycopg.connect(
        DATABASE_URL,
        row_factory=dict_row,
        connect_timeout=5
    )


# ============================================================
# INITIALIZE DATABASE TABLES
# ============================================================

def init_db():
    """
    Create the NeuroBat database tables if they do not already exist.

    Current tables:
    - users
    - player_profiles
    - analysis_sessions
    """

    with get_connection() as conn:

        with conn.cursor() as cursor:

            # ==================================================
            # USERS TABLE
            # ==================================================

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS users (

                    id BIGINT GENERATED ALWAYS
                        AS IDENTITY PRIMARY KEY,

                    full_name VARCHAR(100)
                        NOT NULL,

                    email VARCHAR(120)
                        UNIQUE
                        NOT NULL,

                    password_hash TEXT
                        NOT NULL,

                    created_at TIMESTAMPTZ
                        NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,

                    last_login TIMESTAMPTZ

                );
                """
            )

            # ==================================================
            # PLAYER PROFILES TABLE
            # ==================================================

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS player_profiles (

                    id BIGINT GENERATED ALWAYS
                        AS IDENTITY PRIMARY KEY,

                    user_id BIGINT
                        UNIQUE
                        NOT NULL,

                    batting_hand VARCHAR(30),

                    player_role VARCHAR(50),

                    experience_level VARCHAR(50),

                    age_group VARCHAR(50),

                    team_name VARCHAR(120),

                    created_at TIMESTAMPTZ
                        NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,

                    updated_at TIMESTAMPTZ
                        NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,

                    CONSTRAINT fk_player_user
                        FOREIGN KEY (user_id)
                        REFERENCES users(id)
                        ON DELETE CASCADE

                );
                """
            )


            # ------------------------------------------------
            # ANALYSIS SESSIONS
            # ------------------------------------------------

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_sessions (

                    id BIGINT GENERATED ALWAYS
                       AS IDENTITY PRIMARY KEY,

                    user_id BIGINT
                        NOT NULL,

                    original_video_name VARCHAR(255),

                    processed_video_filename VARCHAR(255),

                    processed_video_url VARCHAR(500),

                    processed_video_codec VARCHAR(50),

                    analysis_engine_version VARCHAR(100),

                    frames_processed INTEGER,

                    poses_detected INTEGER,

                    pose_detection_rate DOUBLE PRECISION,

                    valid_biomechanics_frames INTEGER,

                    valid_biomechanics_frame_rate DOUBLE PRECISION,

                    average_landmark_visibility DOUBLE PRECISION,

                    left_knee_angle DOUBLE PRECISION,

                    right_knee_angle DOUBLE PRECISION,

                    average_knee_angle DOUBLE PRECISION,

                    stance_width DOUBLE PRECISION,

                    normalized_stance_width DOUBLE PRECISION,

                    median_shoulder_width_px DOUBLE PRECISION,

                    median_ankle_separation_px DOUBLE PRECISION,

                    stance_category VARCHAR(50),

                    shoulder_alignment DOUBLE PRECISION,

                    shoulder_tilt_degrees DOUBLE PRECISION,

                    hip_tilt_degrees DOUBLE PRECISION,

                    head_stability DOUBLE PRECISION,

                    head_displacement DOUBLE PRECISION,

                    left_foot_movement DOUBLE PRECISION,

                    right_foot_movement DOUBLE PRECISION,

                    moving_foot_proxy VARCHAR(30),

                    front_foot_movement_proxy DOUBLE PRECISION,

                    stance_rating VARCHAR(50),

                    prototype_score INTEGER,

                    prototype_score_percent INTEGER,

                    created_at TIMESTAMPTZ
                        NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,

                    CONSTRAINT fk_analysis_user

                        FOREIGN KEY (user_id)

                        REFERENCES users(id)

                        ON DELETE CASCADE

                );
                """
            )


            # ------------------------------------------------
            # ANALYSIS INDEX
            # ------------------------------------------------

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_analysis_sessions_user_id

                ON analysis_sessions(user_id);
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_analysis_sessions_created_at

                ON analysis_sessions(created_at);
                """
            )
# ============================================================
# GET USER BY EMAIL
# ============================================================
def get_user_by_email(email):
    """
    Find a NeuroBat user using their email address.

    Returns:
        dict containing user information

    Returns None if the user does not exist.
    """

    if not email:
        return None

    email_key = str(email).strip().lower()

    if not email_key:
        return None

    with get_connection() as conn:

        with conn.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    id,
                    full_name,
                    email,
                    password_hash,
                    created_at,
                    last_login

                FROM users

                WHERE email = %s;
                """,
                (
                    email_key,
                )
            )

            user = cursor.fetchone()

    return user


# ============================================================
# CREATE USER
# ============================================================

def create_user(
    full_name,
    email,
    password_hash
):
    """
    Create a new NeuroBat user.

    Returns:
        Newly created user dictionary.

    Returns:
        None if an account with the same email already exists.
    """

    full_name = str(full_name).strip()
    email_key = str(email).strip().lower()

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not full_name:
        raise ValueError(
            "Full name is required."
        )

    if not email_key:
        raise ValueError(
            "Email is required."
        )

    if not password_hash:
        raise ValueError(
            "Password hash is required."
        )

    # --------------------------------------------------------
    # CREATE USER
    # --------------------------------------------------------

    with get_connection() as conn:

        with conn.cursor() as cursor:

            cursor.execute(
                """
                INSERT INTO users (
                    full_name,
                    email,
                    password_hash
                )

                VALUES (
                    %s,
                    %s,
                    %s
                )

                ON CONFLICT (email)
                DO NOTHING

                RETURNING
                    id,
                    full_name,
                    email,
                    password_hash,
                    created_at,
                    last_login;
                """,
                (
                    full_name,
                    email_key,
                    password_hash,
                )
            )

            user = cursor.fetchone()

    # None means email already exists
    return user


# ============================================================
# UPDATE LAST LOGIN
# ============================================================

def update_last_login(email):
    """
    Update last_login when the user successfully signs in.
    """

    if not email:
        return None

    email_key = str(email).strip().lower()

    with get_connection() as conn:

        with conn.cursor() as cursor:

            cursor.execute(
                """
                UPDATE users

                SET last_login = CURRENT_TIMESTAMP

                WHERE email = %s

                RETURNING
                    id,
                    full_name,
                    email,
                    created_at,
                    last_login;
                """,
                (
                    email_key,
                )
            )

            user = cursor.fetchone()

    return user


# ============================================================
# GET LOGGED-IN USER
# ============================================================

def get_logged_in_user():
    """
    Get the currently logged-in NeuroBat user
    using the Flask session.
    """

    email = session.get(
        "user_email"
    )

    if not email:
        return None

    return get_user_by_email(
        email
    )


# ============================================================
# GET PLAYER PROFILE
# ============================================================

def get_player_profile(email):
    """
    Retrieve the player profile belonging to a user.

    Returns an empty dictionary if no profile exists yet.
    """

    if not email:
        return {}

    email_key = str(email).strip().lower()

    if not email_key:
        return {}

    with get_connection() as conn:

        with conn.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    p.id,
                    p.user_id,
                    p.batting_hand,
                    p.player_role,
                    p.experience_level,
                    p.age_group,
                    p.team_name,
                    p.created_at,
                    p.updated_at

                FROM player_profiles p

                JOIN users u
                    ON u.id = p.user_id

                WHERE u.email = %s;
                """,
                (
                    email_key,
                )
            )

            profile = cursor.fetchone()

    if not profile:
        return {}

    return profile


# ============================================================
# SAVE / UPDATE PLAYER PROFILE
# ============================================================

def save_player_profile(
    email,
    profile
):
    """
    Create or update a player's NeuroBat profile.

    One player profile is allowed per user.

    If a profile already exists, PostgreSQL updates it.
    """

    if not email:
        raise ValueError(
            "Email is required."
        )

    if profile is None:
        profile = {}

    email_key = str(email).strip().lower()

    # --------------------------------------------------------
    # FIND USER
    # --------------------------------------------------------

    user = get_user_by_email(
        email_key
    )

    if not user:
        raise ValueError(
            "User does not exist."
        )

    # --------------------------------------------------------
    # CREATE OR UPDATE PLAYER PROFILE
    # --------------------------------------------------------

    with get_connection() as conn:

        with conn.cursor() as cursor:

            cursor.execute(
                """
                INSERT INTO player_profiles (

                    user_id,
                    batting_hand,
                    player_role,
                    experience_level,
                    age_group,
                    team_name

                )

                VALUES (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s
                )

                ON CONFLICT (user_id)

                DO UPDATE SET

                    batting_hand =
                        EXCLUDED.batting_hand,

                    player_role =
                        EXCLUDED.player_role,

                    experience_level =
                        EXCLUDED.experience_level,

                    age_group =
                        EXCLUDED.age_group,

                    team_name =
                        EXCLUDED.team_name,

                    updated_at =
                        CURRENT_TIMESTAMP

                RETURNING
                    id,
                    user_id,
                    batting_hand,
                    player_role,
                    experience_level,
                    age_group,
                    team_name,
                    created_at,
                    updated_at;
                """,
                (
                    user["id"],

                    profile.get(
                        "batting_hand",
                        ""
                    ),

                    profile.get(
                        "player_role",
                        ""
                    ),

                    profile.get(
                        "experience_level",
                        ""
                    ),

                    profile.get(
                        "age_group",
                        ""
                    ),

                    profile.get(
                        "team_name",
                        ""
                    ),
                )
            )

            saved_profile = cursor.fetchone()

    return saved_profile
# ============================================================
# SAVE ANALYSIS SESSION
# ============================================================

def save_analysis_session(
    email,
    result
):

    user = get_user_by_email(
        email
    )


    if not user:

        raise ValueError(
            "User does not exist."
        )


    with get_connection() as conn:

        with conn.cursor() as cursor:

            cursor.execute(
                """
                INSERT INTO analysis_sessions (

                    user_id,

                    original_video_name,

                    processed_video_filename,

                    processed_video_url,

                    processed_video_codec,

                    analysis_engine_version,

                    frames_processed,

                    poses_detected,

                    pose_detection_rate,

                    valid_biomechanics_frames,

                    valid_biomechanics_frame_rate,

                    average_landmark_visibility,

                    left_knee_angle,

                    right_knee_angle,

                    average_knee_angle,

                    stance_width,

                    normalized_stance_width,

                    median_shoulder_width_px,

                    median_ankle_separation_px,

                    stance_category,

                    shoulder_alignment,

                    shoulder_tilt_degrees,

                    hip_tilt_degrees,

                    head_stability,

                    head_displacement,

                    left_foot_movement,

                    right_foot_movement,

                    moving_foot_proxy,

                    front_foot_movement_proxy,

                    stance_rating,

                    prototype_score,

                    prototype_score_percent

                )

                VALUES (

                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s

                )

                RETURNING

                    id,
                    user_id,
                    original_video_name,
                    processed_video_filename,
                    prototype_score_percent,
                    stance_rating,
                    created_at;
                """,

                (
                    user["id"],

                    result.get(
                        "original_video_name"
                    ),

                    result.get(
                        "processed_video_filename"
                    ),

                    result.get(
                        "processed_video_url"
                    ),

                    result.get(
                        "processed_video_codec"
                    ),

                    result.get(
                        "analysis_engine_version"
                    ),

                    result.get(
                        "frames_processed"
                    ),

                    result.get(
                        "poses_detected"
                    ),

                    result.get(
                        "pose_detection_rate"
                    ),

                    result.get(
                        "valid_biomechanics_frames"
                    ),

                    result.get(
                        "valid_biomechanics_frame_rate"
                    ),

                    result.get(
                        "average_landmark_visibility"
                    ),

                    result.get(
                        "left_knee_angle"
                    ),

                    result.get(
                        "right_knee_angle"
                    ),

                    result.get(
                        "average_knee_angle"
                    ),

                    result.get(
                        "stance_width"
                    ),

                    result.get(
                        "normalized_stance_width"
                    ),

                    result.get(
                        "median_shoulder_width_px"
                    ),

                    result.get(
                        "median_ankle_separation_px"
                    ),

                    result.get(
                        "stance_category"
                    ),

                    result.get(
                        "shoulder_alignment"
                    ),

                    result.get(
                        "shoulder_tilt_degrees"
                    ),

                    result.get(
                        "hip_tilt_degrees"
                    ),

                    result.get(
                        "head_stability"
                    ),

                    result.get(
                        "head_displacement"
                    ),

                    result.get(
                        "left_foot_movement"
                    ),

                    result.get(
                        "right_foot_movement"
                    ),

                    result.get(
                        "moving_foot_proxy"
                    ),

                    result.get(
                        "front_foot_movement_proxy"
                    ),

                    result.get(
                        "stance_rating"
                    ),

                    result.get(
                        "prototype_score"
                    ),

                    result.get(
                        "prototype_score_percent"
                    ),
                )
            )


            saved_analysis = (
                cursor.fetchone()
            )


    return saved_analysis

# ============================================================
# GET ANALYSIS HISTORY
# ============================================================

def get_analysis_history(
    email,
    limit=50,
):
    """
    Return previous NeuroBat analysis sessions
    belonging to one user.

    Newest analyses are returned first.

    The query joins users directly by email so it does not
    depend on another lookup function. Datetime values are
    converted to ISO strings so the result is safe for JSON.
    """

    if not email:
        return []

    email_key = str(email).strip().lower()

    if not email_key:
        return []

    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 50

    # Prevent accidental/unbounded history queries.
    limit = max(1, min(limit, 100))

    with get_connection() as conn:

        with conn.cursor() as cursor:

            cursor.execute(
                """
                SELECT

                    a.id,
                    a.user_id,

                    a.original_video_name,

                    a.processed_video_filename,
                    a.processed_video_url,
                    a.processed_video_codec,

                    a.analysis_engine_version,

                    a.frames_processed,
                    a.poses_detected,
                    a.pose_detection_rate,

                    a.valid_biomechanics_frames,
                    a.valid_biomechanics_frame_rate,
                    a.average_landmark_visibility,

                    a.left_knee_angle,
                    a.right_knee_angle,
                    a.average_knee_angle,

                    a.stance_width,
                    a.normalized_stance_width,

                    a.median_shoulder_width_px,
                    a.median_ankle_separation_px,

                    a.stance_category,

                    a.shoulder_alignment,
                    a.shoulder_tilt_degrees,
                    a.hip_tilt_degrees,

                    a.head_stability,
                    a.head_displacement,

                    a.left_foot_movement,
                    a.right_foot_movement,
                    a.moving_foot_proxy,
                    a.front_foot_movement_proxy,

                    a.stance_rating,

                    a.prototype_score,
                    a.prototype_score_percent,

                    a.created_at

                FROM analysis_sessions AS a

                INNER JOIN users AS u
                    ON u.id = a.user_id

                WHERE u.email = %s

                ORDER BY a.created_at DESC, a.id DESC

                LIMIT %s;
                """,
                (
                    email_key,
                    limit,
                )
            )

            rows = cursor.fetchall()

    analyses = []

    for row in rows:

        item = dict(row)

        created_at = item.get(
            "created_at"
        )

        if created_at is not None:
            item["created_at"] = (
                created_at.isoformat()
            )

        analyses.append(
            item
        )

    return analyses
# ============================================================
# GET PLAYER PROGRESS DATA
# ============================================================

def get_progress_sessions(
    email,
    limit=100,
):
    """
    Return historical NeuroBat analysis data
    required for player progress tracking.

    Oldest session is returned first so the
    frontend can plot chronological charts.
    """

    if not email:
        return []

    email_key = (
        str(email)
        .strip()
        .lower()
    )

    if not email_key:
        return []


    try:

        limit = int(
            limit
        )

    except (
        TypeError,
        ValueError,
    ):

        limit = 100


    limit = max(
        1,
        min(
            limit,
            200,
        )
    )


    with get_connection() as conn:

        with conn.cursor() as cursor:

            cursor.execute(
                """
                SELECT

                    a.id,

                    a.original_video_name,

                    a.prototype_score,
                    a.prototype_score_percent,

                    a.average_knee_angle,

                    a.head_stability,
                    a.head_displacement,

                    a.stance_width,
                    a.normalized_stance_width,
                    a.stance_category,
                    a.stance_rating,

                    a.shoulder_alignment,
                    a.shoulder_tilt_degrees,
                    a.hip_tilt_degrees,

                    a.left_foot_movement,
                    a.right_foot_movement,
                    a.front_foot_movement_proxy,

                    a.pose_detection_rate,

                    a.valid_biomechanics_frames,
                    a.valid_biomechanics_frame_rate,

                    a.average_landmark_visibility,

                    a.created_at

                FROM analysis_sessions AS a

                INNER JOIN users AS u
                    ON u.id = a.user_id

                WHERE u.email = %s

                ORDER BY
                    a.created_at ASC,
                    a.id ASC

                LIMIT %s;
                """,
                (
                    email_key,
                    limit,
                )
            )


            rows = cursor.fetchall()


    sessions = []


    for row in rows:

        item = dict(
            row
        )


        created_at = item.get(
            "created_at"
        )


        if created_at is not None:

            item[
                "created_at"
            ] = created_at.isoformat()


        sessions.append(
            item
        )


    return sessions


# ============================================================
# GET ONE ANALYSIS BY ID
# ============================================================

def get_analysis_by_id(
    email,
    analysis_id,
):
    """
    Return one saved NeuroBat analysis session.

    The analysis is returned only if it belongs
    to the supplied user email.
    """

    if not email:
        return None

    email_key = (
        str(email)
        .strip()
        .lower()
    )

    if not email_key:
        return None


    try:

        analysis_id = int(
            analysis_id
        )

    except (
        TypeError,
        ValueError,
    ):

        return None


    if analysis_id <= 0:
        return None


    with get_connection() as conn:

        with conn.cursor() as cursor:

            cursor.execute(
                """
                SELECT

                    a.id,

                    a.original_video_name,

                    a.processed_video_filename,
                    a.processed_video_url,
                    a.processed_video_codec,

                    a.analysis_engine_version,

                    a.frames_processed,
                    a.poses_detected,
                    a.pose_detection_rate,

                    a.valid_biomechanics_frames,
                    a.valid_biomechanics_frame_rate,
                    a.average_landmark_visibility,

                    a.left_knee_angle,
                    a.right_knee_angle,
                    a.average_knee_angle,

                    a.stance_width,
                    a.normalized_stance_width,

                    a.median_shoulder_width_px,
                    a.median_ankle_separation_px,

                    a.stance_category,

                    a.shoulder_alignment,
                    a.shoulder_tilt_degrees,
                    a.hip_tilt_degrees,

                    a.head_stability,
                    a.head_displacement,

                    a.left_foot_movement,
                    a.right_foot_movement,
                    a.moving_foot_proxy,
                    a.front_foot_movement_proxy,

                    a.stance_rating,

                    a.prototype_score,
                    a.prototype_score_percent,

                    a.created_at

                FROM analysis_sessions AS a

                INNER JOIN users AS u
                    ON u.id = a.user_id

                WHERE
                    a.id = %s
                    AND
                    u.email = %s

                LIMIT 1;
                """,
                (
                    analysis_id,
                    email_key,
                )
            )


            analysis = cursor.fetchone()


    if not analysis:
        return None


    analysis = dict(
        analysis
    )


    created_at = analysis.get(
        "created_at"
    )


    if created_at is not None:

        analysis[
            "created_at"
        ] = created_at.isoformat()


    return analysis



# ============================================================
# DATABASE CONNECTION TEST
# ============================================================

def test_database_connection():
    """
    Test whether NeuroBat can connect to PostgreSQL.

    Returns information about the connected database.
    """

    with get_connection() as conn:

        with conn.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    current_database() AS database,
                    current_user AS user;
                """
            )

            result = cursor.fetchone()

    return result


# ============================================================
# RUN DATABASE.PY DIRECTLY
# ============================================================

if __name__ == "__main__":

    try:

        print("Testing NeuroBat PostgreSQL connection...")

        connection_info = test_database_connection()

        print("PostgreSQL connection successful.")
        print(
            "Database:",
            connection_info["database"]
        )
        print(
            "User:",
            connection_info["user"]
        )

        print("\nInitializing NeuroBat database tables...")

        init_db()

        print("Database initialization successful.")
        print("users table ready.")
        print("player_profiles table ready.")
        print("analysis_sessions table ready.")

    except Exception as error:

        print("\nPostgreSQL connection failed.")
        print("Error:", error)