"""
NeuroBat ML Feature Dataset Service
===================================

Converts one completed NeuroBat V3.x analysis result into a flat,
machine-learning-ready CSV row.

Important design rules
----------------------
- One unique source batting clip = one dataset row.
- The same source video is detected using SHA-256 and is not appended twice
  by default.
- Phase-specific biomechanics are preferred over whole-clip averages.
- Expert/technique labels remain blank until manually validated.
- Pose-derived proxies are stored as measurements, not treated as ground-truth
  coaching labels.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BACKEND_DIR / "datasets"
DEFAULT_DATASET_PATH = DATASETS_DIR / "neurobat_features.csv"


PHASE_ORDER = (
    "setup",
    "backlift",
    "downswing",
    "impact",
    "follow_through",
)


PHASE_FEATURES = (
    "left_knee_angle_2d_degrees",
    "right_knee_angle_2d_degrees",
    "average_knee_angle_2d_degrees",
    "left_elbow_angle_2d_degrees",
    "right_elbow_angle_2d_degrees",
    "trunk_lean_2d_degrees",
    "shoulder_tilt_degrees",
    "hip_tilt_degrees",
    "shoulder_hip_separation_3d_proxy_degrees",
    "normalized_stance_width",
    "body_scaled_stance_width",
    "head_movement_body_lengths",
    "left_foot_movement_body_lengths",
    "right_foot_movement_body_lengths",
    "body_center_movement_body_lengths",
    "mean_wrist_speed_body_lengths_per_second",
    "peak_wrist_speed_body_lengths_per_second",
    "average_landmark_visibility",
    "mean_metric_coverage_percent",
)


BASE_COLUMNS = [
    "analysis_session_id",
    "source_video_sha256",
    "original_video_name",
    "analysis_engine_version",

    "batting_hand",
    "age_group",
    "front_foot_side",
    "back_foot_side",

    "shot_type_label",
    "expert_technique_label",
    "weakness_label",
    "expert_score",
    "expert_notes",

    "fps",
    "frames_processed",
    "poses_detected",
    "pose_detection_rate",
    "metric_usable_frames",
    "metric_usable_frame_rate",
    "complete_core_biomechanics_frames",
    "complete_core_biomechanics_frame_rate",
    "average_landmark_visibility",

    "phase_detection_status",
    "phase_detection_confidence",
    "phase_detection_confidence_label",
    "estimated_impact_frame",
    "estimated_impact_time_seconds",
    "impact_is_proxy",

    "stride_displacement_body_lengths",
    "body_center_setup_to_impact_displacement_body_lengths",
    "body_center_forward_transfer_body_lengths",
    "balance_sway_body_lengths",
    "balance_stability_proxy",
    "estimated_weight_transfer_proxy_body_lengths",

    "left_knee_angle",
    "right_knee_angle",
    "average_knee_angle",
    "left_elbow_angle",
    "right_elbow_angle",
    "trunk_lean_degrees",
    "shoulder_hip_separation_proxy_degrees",
    "normalized_stance_width",
]


def _phase_columns():
    columns = []

    for phase_name in PHASE_ORDER:
        columns.append(
            f"{phase_name}_duration_seconds"
        )

        for feature_name in PHASE_FEATURES:
            columns.append(
                f"{phase_name}_{feature_name}"
            )

    return columns


FEATURE_COLUMNS = (
    BASE_COLUMNS
    + _phase_columns()
)


def compute_file_sha256(path, chunk_size=1024 * 1024):
    """
    Return SHA-256 for a local video file.
    """
    digest = hashlib.sha256()

    with open(path, "rb") as source:
        while True:
            chunk = source.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def _get(mapping, key, default=None):
    if isinstance(mapping, dict):
        return mapping.get(key, default)

    return default


def _clean_value(value):
    if value is None:
        return ""

    if isinstance(value, bool):
        return int(value)

    return value


def build_feature_row(
    result,
    analysis_session_id=None,
    video_sha256=None,
):
    """
    Flatten one NeuroBat analysis response into one ML feature row.
    """
    analysis_scope = (
        _get(result, "analysis_scope", {})
        or {}
    )

    player_context = (
        _get(result, "player_context", {})
        or {}
    )

    phase_detection = (
        _get(result, "batting_phase_detection", {})
        or {}
    )

    phases = (
        _get(phase_detection, "phases", {})
        or {}
    )

    phase_metrics = (
        _get(result, "phase_metrics", {})
        or {}
    )

    v3_metrics = (
        _get(result, "v3_metrics", {})
        or {}
    )

    row = {
        "analysis_session_id":
            analysis_session_id,

        "source_video_sha256":
            video_sha256,

        "original_video_name":
            _get(result, "original_video_name"),

        "analysis_engine_version":
            _get(result, "analysis_engine_version"),

        "batting_hand":
            _get(player_context, "batting_hand"),

        "age_group":
            _get(player_context, "age_group"),

        "front_foot_side":
            _get(player_context, "front_foot_side"),

        "back_foot_side":
            _get(player_context, "back_foot_side"),

        "shot_type_label":
            "",

        "expert_technique_label":
            "",

        "weakness_label":
            "",

        "expert_score":
            "",

        "expert_notes":
            "",

        "fps":
            _get(analysis_scope, "fps"),

        "frames_processed":
            _get(result, "frames_processed"),

        "poses_detected":
            _get(result, "poses_detected"),

        "pose_detection_rate":
            _get(result, "pose_detection_rate"),

        "metric_usable_frames":
            _get(result, "metric_usable_frames"),

        "metric_usable_frame_rate":
            _get(result, "metric_usable_frame_rate"),

        "complete_core_biomechanics_frames":
            _get(result, "valid_biomechanics_frames"),

        "complete_core_biomechanics_frame_rate":
            _get(result, "valid_biomechanics_frame_rate"),

        "average_landmark_visibility":
            _get(result, "average_landmark_visibility"),

        "phase_detection_status":
            _get(phase_detection, "status"),

        "phase_detection_confidence":
            _get(phase_detection, "confidence"),

        "phase_detection_confidence_label":
            _get(phase_detection, "confidence_label"),

        "estimated_impact_frame":
            _get(phase_detection, "impact_frame"),

        "estimated_impact_time_seconds":
            _get(phase_detection, "impact_time_seconds"),

        "impact_is_proxy":
            _get(phase_detection, "impact_is_proxy"),

        "stride_displacement_body_lengths":
            _get(
                result,
                "stride_displacement_body_lengths",
            ),

        "body_center_setup_to_impact_displacement_body_lengths":
            _get(
                v3_metrics,
                "body_center_setup_to_impact_displacement_body_lengths",
            ),

        "body_center_forward_transfer_body_lengths":
            _get(
                v3_metrics,
                "body_center_forward_transfer_body_lengths",
            ),

        "balance_sway_body_lengths":
            _get(
                result,
                "balance_sway_body_lengths",
            ),

        "balance_stability_proxy":
            _get(
                result,
                "balance_stability_proxy",
            ),

        "estimated_weight_transfer_proxy_body_lengths":
            _get(
                result,
                "estimated_weight_transfer_proxy_body_lengths",
            ),

        "left_knee_angle":
            _get(result, "left_knee_angle"),

        "right_knee_angle":
            _get(result, "right_knee_angle"),

        "average_knee_angle":
            _get(result, "average_knee_angle"),

        "left_elbow_angle":
            _get(result, "left_elbow_angle"),

        "right_elbow_angle":
            _get(result, "right_elbow_angle"),

        "trunk_lean_degrees":
            _get(result, "trunk_lean_degrees"),

        "shoulder_hip_separation_proxy_degrees":
            _get(
                result,
                "shoulder_hip_separation_proxy_degrees",
            ),

        "normalized_stance_width":
            _get(result, "normalized_stance_width"),
    }

    for phase_name in PHASE_ORDER:
        phase_data = (
            _get(phases, phase_name, {})
            or {}
        )

        metrics = (
            _get(phase_metrics, phase_name, {})
            or {}
        )

        row[
            f"{phase_name}_duration_seconds"
        ] = _get(
            phase_data,
            "duration_seconds",
        )

        for feature_name in PHASE_FEATURES:
            row[
                f"{phase_name}_{feature_name}"
            ] = _get(
                metrics,
                feature_name,
            )

    return {
        column:
            _clean_value(
                row.get(column)
            )
        for column in FEATURE_COLUMNS
    }


def _existing_video_hashes(csv_path):
    path = Path(csv_path)

    if not path.exists():
        return set()

    hashes = set()

    try:
        with path.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as handle:

            reader = csv.DictReader(handle)

            for row in reader:
                value = (
                    row.get(
                        "source_video_sha256",
                        "",
                    )
                    or ""
                ).strip()

                if value:
                    hashes.add(value)

    except (OSError, csv.Error):
        return set()

    return hashes


def append_feature_row(
    result,
    analysis_session_id=None,
    video_sha256=None,
    csv_path=None,
    skip_duplicate_video=True,
):
    """
    Append one analysis to the NeuroBat ML feature CSV.
    """
    path = Path(
        csv_path
        or DEFAULT_DATASET_PATH
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        skip_duplicate_video
        and video_sha256
        and video_sha256
        in _existing_video_hashes(path)
    ):
        return {
            "status": "duplicate_skipped",
            "dataset_path": str(path),
            "source_video_sha256": video_sha256,
        }

    row = build_feature_row(
        result,
        analysis_session_id=analysis_session_id,
        video_sha256=video_sha256,
    )

    file_exists = path.exists()

    with path.open(
        "a",
        newline="",
        encoding="utf-8-sig",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=FEATURE_COLUMNS,
            extrasaction="ignore",
        )

        if (
            not file_exists
            or path.stat().st_size == 0
        ):
            writer.writeheader()

        writer.writerow(row)

    return {
        "status": "added",
        "dataset_path": str(path),
        "source_video_sha256": video_sha256,
        "feature_count": len(FEATURE_COLUMNS),
    }