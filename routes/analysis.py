"""
Video-analysis request routes for NeuroBat.

Responsibilities:
- Serve example/source videos
- Serve processed analysis videos
- Accept video uploads
- Run the NeuroBat biomechanics engine
- Save completed analysis sessions to PostgreSQL
- Return analysis history for the logged-in user
"""

from flask import (
    Blueprint,
    current_app,
    jsonify,
    request,
    send_from_directory,
)

from database import (
    get_analysis_by_id,
    get_analysis_history,
    get_logged_in_user,
    get_player_profile,
    get_progress_sessions,
    save_analysis_session,
)

from services.biomechanics import (
    ENGINE_VERSION,
    MAX_ANALYSIS_FRAMES,
    process_video,
)

from services.video_service import (
    allowed_video,
    remove_file_safely,
    save_temporary_upload,
)

from services.feature_dataset import (
    append_feature_row,
    compute_file_sha256,
)

from services.ml_predictor import (
    predict_shot_type,
)


# ============================================================
# BLUEPRINT
# ============================================================

analysis_bp = Blueprint(
    "analysis",
    __name__,
)


# ============================================================
# FRAME INSPECTOR RESPONSE HELPER
# ============================================================

def prepare_frame_inspector_response(result):
    """
    Add compact frontend metadata for the NeuroBat Frame Inspector.

    The biomechanics engine already returns the JSON-safe ``frame_analysis``
    list. This helper does not duplicate those records; it adds only the
    information the frontend needs to initialize the viewer and jump between
    detected batting phases.
    """

    frame_analysis = result.get("frame_analysis")

    if not isinstance(frame_analysis, list):
        frame_analysis = []
        result["frame_analysis"] = frame_analysis

    frame_meta = result.get("frame_analysis_meta")

    if not isinstance(frame_meta, dict):
        frame_meta = {
            "total_frames": len(frame_analysis),
            "fps": result.get(
                "analysis_scope",
                {},
            ).get("fps"),
            "frame_step_seconds": None,
            "source": "processed_video_timeline",
        }
        result["frame_analysis_meta"] = frame_meta

    phase_detection = result.get("batting_phase_detection") or {}
    phases = phase_detection.get("phases") or {}

    phase_jump_frames = {}

    for phase_name in (
        "setup",
        "backlift",
        "downswing",
        "impact",
        "follow_through",
    ):
        phase_data = phases.get(phase_name)

        if not isinstance(phase_data, dict):
            continue

        if phase_name == "impact":
            target_frame = phase_detection.get("impact_frame")

            if target_frame is None:
                target_frame = phase_data.get("start_frame")
        else:
            target_frame = phase_data.get("start_frame")

        if target_frame is not None:
            try:
                phase_jump_frames[phase_name] = int(target_frame)
            except (TypeError, ValueError):
                pass

    result["frame_inspector"] = {
        "available": bool(frame_analysis),
        "engine_version": result.get(
            "analysis_engine_version",
            ENGINE_VERSION,
        ),
        "total_frames": len(frame_analysis),
        "fps": frame_meta.get("fps"),
        "processed_video_url": result.get("processed_video_url"),
        "phase_jump_frames": phase_jump_frames,
        "estimated_impact_frame": phase_detection.get("impact_frame"),
        "estimated_impact_is_proxy": bool(
            phase_detection.get("impact_is_proxy")
        ),
    }

    return result


# ============================================================
# VIDEO FILE ROUTE
# ============================================================

@analysis_bp.route(
    "/videos/<path:filename>"
)
def video_files(filename):

    return send_from_directory(
        current_app.config["VIDEOS_DIR"],
        filename,
    )


# ============================================================
# PROCESSED OUTPUT FILE ROUTE
# ============================================================

@analysis_bp.route(
    "/outputs/<path:filename>"
)
def output_files(filename):

    return send_from_directory(
        current_app.config["OUTPUTS_DIR"],
        filename,
    )


# ============================================================
# ANALYSIS HISTORY API
# ============================================================

@analysis_bp.route(
    "/api/analysis-history",
    methods=["GET"],
)
def analysis_history():

    try:

        # ----------------------------------------------------
        # REQUIRE LOGGED-IN USER
        # ----------------------------------------------------

        user = get_logged_in_user()

        if not user:

            return jsonify({
                "error":
                    "You must be logged in to view analysis history."
            }), 401


        # ----------------------------------------------------
        # LOAD ANALYSIS HISTORY
        # ----------------------------------------------------

        analyses = get_analysis_history(
            user["email"],
            limit=50,
        )


        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        return jsonify({
            "count": len(analyses),
            "analyses": analyses,
        }), 200


    except Exception as error:

        print(
            "[NeuroBat] Analysis history error:",
            str(error),
        )

        return jsonify({
            "error": str(error)
        }), 500

# ============================================================
# PLAYER PROGRESS API
# ============================================================

@analysis_bp.route(
    "/api/progress",
    methods=["GET"],
)
def player_progress():

    try:

        # ----------------------------------------------------
        # REQUIRE LOGGED-IN USER
        # ----------------------------------------------------

        user = get_logged_in_user()


        if not user:

            return jsonify({
                "error":
                    "You must be logged in to view progress."
            }), 401


        # ----------------------------------------------------
        # LOAD HISTORICAL SESSIONS
        # ----------------------------------------------------

        sessions = get_progress_sessions(
            user["email"],
            limit=100,
        )


        # ----------------------------------------------------
        # NO ANALYSES YET
        # ----------------------------------------------------

        if not sessions:

            return jsonify({

                "total_sessions": 0,

                "latest_score": None,

                "previous_score": None,

                "score_change": None,

                "average_score": None,

                "best_score": None,

                "sessions": [],

            }), 200


        # ----------------------------------------------------
        # SCORE VALUES
        # ----------------------------------------------------

        scores = []


        for session_item in sessions:

            score = session_item.get(
                "prototype_score_percent"
            )


            if score is not None:

                try:

                    scores.append(
                        float(score)
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    pass


        # ----------------------------------------------------
        # SUMMARY VALUES
        # ----------------------------------------------------

        latest_score = (
            scores[-1]
            if scores
            else None
        )


        previous_score = (
            scores[-2]
            if len(scores) >= 2
            else None
        )


        score_change = None


        if (
            latest_score is not None
            and
            previous_score is not None
        ):

            score_change = round(
                latest_score -
                previous_score,
                2,
            )


        average_score = (
            round(
                sum(scores) /
                len(scores),
                2,
            )
            if scores
            else None
        )


        best_score = (
            max(scores)
            if scores
            else None
        )


        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        return jsonify({

            "total_sessions":
                len(sessions),

            "latest_score":
                latest_score,

            "previous_score":
                previous_score,

            "score_change":
                score_change,

            "average_score":
                average_score,

            "best_score":
                best_score,

            "sessions":
                sessions,

        }), 200


    except Exception as error:

        print(
            "[NeuroBat] Progress API error:",
            str(error),
        )


        return jsonify({
            "error":
                str(error)
        }), 500
# ============================================================
# SINGLE ANALYSIS API
# ============================================================

@analysis_bp.route(
    "/api/analysis/<int:analysis_id>",
    methods=["GET"],
)
def get_saved_analysis(
    analysis_id,
):

    try:

        # ----------------------------------------------------
        # REQUIRE LOGGED-IN USER
        # ----------------------------------------------------

        user = get_logged_in_user()


        if not user:

            return jsonify({
                "error":
                    "You must be logged in to view this analysis."
            }), 401


        # ----------------------------------------------------
        # LOAD ANALYSIS
        # ----------------------------------------------------

        analysis = get_analysis_by_id(
            user["email"],
            analysis_id,
        )


        # ----------------------------------------------------
        # NOT FOUND / NOT OWNED BY USER
        # ----------------------------------------------------

        if not analysis:

            return jsonify({
                "error":
                    "Analysis session not found."
            }), 404


        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        return jsonify({
            "analysis":
                analysis
        }), 200


    except Exception as error:

        print(
            "[NeuroBat] Analysis detail API error:",
            str(error),
        )


        return jsonify({
            "error":
                str(error)
        }), 500




    
# ============================================================
# ANALYZE VIDEO
# ============================================================

@analysis_bp.route(
    "/analyze",
    methods=["POST"],
)
def analyze():

    # ========================================================
    # REQUIRE LOGGED-IN USER
    # ========================================================

    user = get_logged_in_user()

    if not user:

        return jsonify({
            "error":
                "You must be logged in to analyze a video."
        }), 401


    # ========================================================
    # CHECK VIDEO EXISTS
    # ========================================================

    if "video" not in request.files:

        return jsonify({
            "error":
                "No video uploaded."
        }), 400


    uploaded_file = request.files[
        "video"
    ]


    # ========================================================
    # CHECK FILE SELECTED
    # ========================================================

    if (
        not uploaded_file
        or uploaded_file.filename == ""
    ):

        return jsonify({
            "error":
                "No video file selected."
        }), 400


    # ========================================================
    # CHECK VIDEO FORMAT
    # ========================================================

    if not allowed_video(
        uploaded_file.filename
    ):

        return jsonify({
            "error":
                "Unsupported video format. "
                "Use MP4, MOV, AVI, WEBM or MKV."
        }), 400


    temporary_path = None
    source_video_sha256 = None


    try:

        # ====================================================
        # SAVE TEMPORARY UPLOAD
        # ====================================================

        (
            original_filename,
            temporary_path,

        ) = save_temporary_upload(

            uploaded_file,

            current_app.config[
                "UPLOADS_DIR"
            ],
        )


        # ====================================================
        # SOURCE VIDEO FINGERPRINT
        # ====================================================

        source_video_sha256 = compute_file_sha256(
            temporary_path
        )


        # ====================================================
        # LOAD PLAYER CONTEXT
        # ====================================================

        player_profile = get_player_profile(
            user["email"]
        )

        batting_hand = (
            player_profile.get("batting_hand")
            if player_profile
            else None
        )

        age_group = (
            player_profile.get("age_group")
            if player_profile
            else None
        )


        # ====================================================
        # RUN BIOMECHANICS ENGINE
        # ====================================================

        result = process_video(
            temporary_path,
            max_frames=MAX_ANALYSIS_FRAMES,
            batting_hand=batting_hand,
            age_group=age_group,
        )


        # ====================================================
        # BIOMECHANICS ERROR
        # ====================================================

        if "error" in result:

            return jsonify(
                result
            ), 422


        # ====================================================
        # ADD ORIGINAL VIDEO NAME
        # ====================================================

        result[
            "original_video_name"
        ] = original_filename


        # ====================================================
        # PREPARE FRAME INSPECTOR RESPONSE
        # ====================================================

        result = prepare_frame_inspector_response(
            result
        )


        # ====================================================
        # RUN XGBOOST SHOT-TYPE PROOF-OF-CONCEPT
        # ====================================================

        try:

            result["ml_prediction"] = predict_shot_type(
                result
            )

        except Exception as ml_error:

            print(
                "[NeuroBat] XGBoost prediction error:",
                str(ml_error),
            )

            result["ml_prediction"] = {
                "status": "prediction_error",
                "available": False,
                "reason": str(ml_error),
                "model_type": "XGBoost",
                "target": "shot_type",
                "scope": "academic_proof_of_concept",
            }


        # ====================================================
        # SAVE ANALYSIS TO POSTGRESQL
        # ====================================================

        saved_analysis = save_analysis_session(
            user["email"],
            result,
        )


        # ====================================================
        # ADD DATABASE SESSION ID TO RESPONSE
        # ====================================================

        if saved_analysis:

            result[
                "analysis_session_id"
            ] = saved_analysis["id"]

        else:

            result[
                "analysis_session_id"
            ] = None


        # ====================================================
        # EXPORT ML FEATURE DATASET ROW
        # ====================================================

        try:

            dataset_export = append_feature_row(
                result,
                analysis_session_id=result.get(
                    "analysis_session_id"
                ),
                video_sha256=source_video_sha256,
                skip_duplicate_video=True,
            )

        except Exception as dataset_error:

            print(
                "[NeuroBat] Feature dataset export error:",
                str(dataset_error),
            )

            dataset_export = {
                "status": "export_error",
                "error": str(dataset_error),
            }


        result[
            "feature_dataset_export"
        ] = dataset_export


        # ====================================================
        # SUCCESS
        # ====================================================

        print(
            "[NeuroBat] Analysis completed and saved:",
            result.get(
                "analysis_session_id"
            ),
        )

        return jsonify(
            result
        ), 200


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as error:

        print(
            "[NeuroBat] Analysis error:",
            str(error),
        )

        return jsonify({
            "error": str(error)
        }), 500


    # ========================================================
    # DELETE TEMPORARY ORIGINAL UPLOAD
    # ========================================================

    finally:

        remove_file_safely(
            temporary_path
        )