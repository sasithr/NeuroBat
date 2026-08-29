"""
NeuroBat Biomechanics Engine V3.2.1
=================================

Metric-specific landmark quality, pose-based batting phase detection, and
extended cricket batting biomechanics.

Scientific boundaries
---------------------
- MediaPipe tracks the player, not the bat or ball.
- The "impact" event is an estimated wrist-kinematics proxy, not verified
  bat-ball contact.
- Shoulder/hip axial orientation and shoulder-hip separation are monocular
  MediaPipe world-landmark proxies, not laboratory-grade 3D measurements.
- Trunk lean is measured in the 2D image plane.
- Balance and weight-transfer outputs are video-derived movement proxies, not
  force-plate or centre-of-pressure measurements.
"""

from __future__ import annotations

import math
import os
import uuid
from pathlib import Path

from services.phase_detection import (
    detect_batting_phases,
    phase_name_for_array_index,
    public_phase_detection,
)


# =============================================================================
# PATHS / ENGINE SETTINGS
# =============================================================================

BACKEND_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = str(BACKEND_DIR / "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

ENGINE_VERSION = "NeuroBat-Biomechanics-V3.2.1-MissingValueFix"

MAX_ANALYSIS_FRAMES = 360

MIN_LANDMARK_VISIBILITY = 0.40
BODY_REFERENCE_MIN_VISIBILITY = 0.25
PHASE_MIN_WRIST_VISIBILITY = 0.25
PHASE_SINGLE_WRIST_HIGH_VISIBILITY = 0.60

# Legacy prototype thresholds kept only so existing dashboards/history do not
# break. They are not validated universal cricket coaching norms.
STANCE_WIDTH_THRESHOLDS = {
    "Very Narrow": 0.20,
    "Narrow": 0.30,
    "Balanced": 0.45,
    "Wide": 0.55,
}


# =============================================================================
# GEOMETRY / QUALITY HELPERS
# =============================================================================


def calculate_angle(a, b, c):
    """Calculate 2D internal angle ABC in degrees."""
    radians = (
        math.atan2(c[1] - b[1], c[0] - b[0])
        - math.atan2(a[1] - b[1], a[0] - b[0])
    )
    angle = abs(math.degrees(radians))
    if angle > 180:
        angle = 360 - angle
    return angle


def calculate_angle_3d(a, b, c):
    """Calculate 3D internal angle ABC in degrees using a vector dot product."""
    ba = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
    bc = (c[0] - b[0], c[1] - b[1], c[2] - b[2])

    norm_ba = math.sqrt(sum(value * value for value in ba))
    norm_bc = math.sqrt(sum(value * value for value in bc))

    if norm_ba <= 1e-9 or norm_bc <= 1e-9:
        return None

    cosine = sum(ba[i] * bc[i] for i in range(3)) / (norm_ba * norm_bc)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def calculate_line_angle(a, b):
    """Absolute 2D tilt relative to horizontal: 0° horizontal, 90° vertical."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    angle = abs(math.degrees(math.atan2(dy, dx)))
    if angle > 90:
        angle = 180 - angle
    return angle


def calculate_trunk_lean(shoulder_center, hip_center):
    """
    Absolute trunk lean relative to vertical in the image plane.

    0° = upright/vertical. This is not full 3D trunk orientation.
    """
    dx = shoulder_center[0] - hip_center[0]
    dy = shoulder_center[1] - hip_center[1]

    if abs(dx) + abs(dy) <= 1e-9:
        return None

    angle_from_horizontal = abs(math.degrees(math.atan2(dy, dx)))
    if angle_from_horizontal > 90:
        angle_from_horizontal = 180 - angle_from_horizontal

    return abs(90.0 - angle_from_horizontal)


def calculate_axial_orientation_proxy(a, b):
    """
    Orientation of a left-right body segment in MediaPipe world x-z space.

    This is a monocular 3D pose proxy, not laboratory-grade torso rotation.
    """
    dx = b[0] - a[0]
    dz = b[2] - a[2]
    if abs(dx) + abs(dz) <= 1e-9:
        return None
    return math.degrees(math.atan2(dz, dx))


def angular_difference_degrees(angle_a, angle_b):
    """Smallest absolute angular difference between two orientations."""
    if angle_a is None or angle_b is None:
        return None
    difference = (float(angle_a) - float(angle_b) + 180.0) % 360.0 - 180.0
    return abs(difference)


def euclidean_distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def midpoint(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def landmark_to_pixel(landmark, width, height):
    return (float(landmark.x) * width, float(landmark.y) * height)


def landmark_to_world_tuple(landmark):
    return (float(landmark.x), float(landmark.y), float(landmark.z))


def relative_point(point, origin, scale):
    if scale <= 1e-9:
        return None
    return ((point[0] - origin[0]) / scale, (point[1] - origin[1]) / scale)


def point_in_body_units(point, scale):
    """Image-origin point expressed in current body-scale units."""
    if point is None or scale <= 1e-9:
        return None
    return (point[0] / scale, point[1] / scale)


def landmark_visible(landmark, threshold=MIN_LANDMARK_VISIBILITY):
    if landmark is None:
        return False
    try:
        return float(landmark.visibility) >= float(threshold)
    except (TypeError, ValueError, AttributeError):
        return False


def landmarks_visible(landmarks, threshold=MIN_LANDMARK_VISIBILITY):
    return bool(landmarks) and all(
        landmark_visible(landmark, threshold) for landmark in landmarks
    )


def coverage_summary(valid_count, total_count):
    total_count = max(0, int(total_count))
    valid_count = max(0, int(valid_count))
    percent = (valid_count / total_count * 100.0) if total_count else 0.0
    return {
        "valid_frames": valid_count,
        "total_frames": total_count,
        "coverage_percent": round(percent, 1),
    }


# =============================================================================
# NUMERIC HELPERS
# =============================================================================


def safe_mean(values, np):
    clean = [
        float(value)
        for value in values
        if value is not None and np.isfinite(value)
    ]
    if not clean:
        return 0.0
    return float(np.mean(clean))


def safe_std(values, np):
    clean = [
        float(value)
        for value in values
        if value is not None and np.isfinite(value)
    ]
    if not clean:
        return 0.0
    return float(np.std(clean))


def safe_median(values, np):
    clean = [
        float(value)
        for value in values
        if value is not None and np.isfinite(value)
    ]
    if not clean:
        return 0.0
    return float(np.median(clean))


def safe_round(value, places=3):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return round(value, places)


def circular_mean_degrees(values, np):
    clean = [
        float(value)
        for value in values
        if value is not None and np.isfinite(value)
    ]
    if not clean:
        return None

    radians = np.radians(clean)
    mean_sin = float(np.mean(np.sin(radians)))
    mean_cos = float(np.mean(np.cos(radians)))

    if abs(mean_sin) + abs(mean_cos) <= 1e-9:
        return None

    return float(np.degrees(np.arctan2(mean_sin, mean_cos)))


def displacement_from_baseline(points, np, baseline_count=5):
    """Maximum 2D displacement from the early median point."""
    clean = [point for point in points if point is not None]
    if not clean:
        return 0.0

    baseline_count = min(max(1, baseline_count), len(clean))
    baseline = (
        float(np.median([point[0] for point in clean[:baseline_count]])),
        float(np.median([point[1] for point in clean[:baseline_count]])),
    )

    distances = [euclidean_distance(baseline, point) for point in clean]
    return float(max(distances)) if distances else 0.0


def rms_displacement_from_median(points, np):
    """RMS 2D sway around the median point."""
    clean = [point for point in points if point is not None]
    if not clean:
        return None

    center = (
        float(np.median([point[0] for point in clean])),
        float(np.median([point[1] for point in clean])),
    )
    squared = [euclidean_distance(center, point) ** 2 for point in clean]
    return float(math.sqrt(float(np.mean(squared)))) if squared else None


def median_point(points, np):
    clean = [point for point in points if point is not None]
    if not clean:
        return None
    return (
        float(np.median([point[0] for point in clean])),
        float(np.median([point[1] for point in clean])),
    )


def project_displacement(start_point, end_point, axis_start, axis_end):
    """
    Project start->end displacement onto axis_start->axis_end.

    Returns signed displacement in the same units as the points.
    """
    if None in (start_point, end_point, axis_start, axis_end):
        return None

    axis = (
        axis_end[0] - axis_start[0],
        axis_end[1] - axis_start[1],
    )
    norm = math.sqrt(axis[0] ** 2 + axis[1] ** 2)
    if norm <= 1e-9:
        return None

    unit = (axis[0] / norm, axis[1] / norm)
    displacement = (
        end_point[0] - start_point[0],
        end_point[1] - start_point[1],
    )
    return displacement[0] * unit[0] + displacement[1] * unit[1]


# =============================================================================
# PLAYER CONTEXT
# =============================================================================


def normalize_batting_hand(value):
    if value is None:
        return None
    text = str(value).strip().lower()

    if text in {"right", "right-handed", "right handed", "r", "rh"}:
        return "Right"
    if text in {"left", "left-handed", "left handed", "l", "lh"}:
        return "Left"
    return None


def front_back_sides(batting_hand):
    hand = normalize_batting_hand(batting_hand)
    if hand == "Right":
        return "Left", "Right"
    if hand == "Left":
        return "Right", "Left"
    return None, None


# =============================================================================
# VIDEO OUTPUT
# =============================================================================


def create_video_writer(cv2, width, height, fps):
    """Create a browser-compatible annotated output writer."""
    candidates = [("webm", "VP80"), ("mp4", "mp4v")]

    for extension, codec in candidates:
        filename = f"processed_{uuid.uuid4().hex[:10]}.{extension}"
        path = os.path.join(OUTPUTS_DIR, filename)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

        if writer.isOpened():
            return writer, filename, path, codec

        writer.release()
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    raise RuntimeError("OpenCV could not create a processed output video.")


def remove_output_safely(path):
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


# =============================================================================
# PHASE / CROSS-PHASE HELPERS
# =============================================================================


def _phase_values(records, start, end, key):
    return [
        record.get(key)
        for record in records[start : end + 1]
        if record.get(key) is not None
    ]


def _phase_median_point(records, phase_range, key, np):
    if not phase_range:
        return None
    start, end = phase_range
    return median_point(_phase_values(records, start, end, key), np)


def build_cross_phase_movement_metrics(frame_records, detection, batting_hand, np):
    """
    Build V3.2 movement proxies across setup -> impact.

    All results are video-derived proxies and depend on a static camera and a
    visible setup/impact region.
    """
    result = {
        "stride_displacement_body_lengths": None,
        "body_center_setup_to_impact_displacement_body_lengths": None,
        "body_center_forward_transfer_body_lengths": None,
        "balance_sway_body_lengths": None,
        "balance_stability_proxy": None,
        "estimated_weight_transfer_proxy_body_lengths": None,
        "front_foot_side": None,
        "back_foot_side": None,
    }

    if not detection or detection.get("status") != "detected":
        return result

    ranges = detection.get("_ranges", {})
    setup_range = ranges.get("setup")
    impact_range = ranges.get("impact")
    selected_start = ranges.get("setup", ranges.get("backlift", (0, 0)))[0]
    selected_end = ranges.get("follow_through", ranges.get("impact", (0, 0)))[1]

    selected_body_centers = _phase_values(
        frame_records,
        selected_start,
        selected_end,
        "body_center_rel",
    )

    sway = rms_displacement_from_median(selected_body_centers, np)
    result["balance_sway_body_lengths"] = safe_round(sway, 3)
    if sway is not None:
        # Dimensionless stability proxy. Higher = less sway. Not a validated score.
        result["balance_stability_proxy"] = safe_round(1.0 / (1.0 + sway), 3)

    if not setup_range or not impact_range:
        return result

    setup_body = _phase_median_point(frame_records, setup_range, "body_center_rel", np)
    impact_body = _phase_median_point(frame_records, impact_range, "body_center_rel", np)

    if setup_body is not None and impact_body is not None:
        result["body_center_setup_to_impact_displacement_body_lengths"] = safe_round(
            euclidean_distance(setup_body, impact_body),
            3,
        )

    front_side, back_side = front_back_sides(batting_hand)
    result["front_foot_side"] = front_side
    result["back_foot_side"] = back_side

    if not front_side:
        return result

    front_key = "left_ankle_rel" if front_side == "Left" else "right_ankle_rel"
    back_key = "right_ankle_rel" if front_side == "Left" else "left_ankle_rel"

    setup_front = _phase_median_point(frame_records, setup_range, front_key, np)
    impact_front = _phase_median_point(frame_records, impact_range, front_key, np)
    setup_back = _phase_median_point(frame_records, setup_range, back_key, np)

    if setup_front is not None and impact_front is not None:
        result["stride_displacement_body_lengths"] = safe_round(
            euclidean_distance(setup_front, impact_front),
            3,
        )

    if (
        setup_body is not None
        and impact_body is not None
        and setup_back is not None
        and setup_front is not None
    ):
        forward_transfer = project_displacement(
            setup_body,
            impact_body,
            setup_back,
            setup_front,
        )
        result["body_center_forward_transfer_body_lengths"] = safe_round(
            forward_transfer,
            3,
        )
        # Same projection exposed using coaching-domain wording. It is movement,
        # not force distribution or actual body weight.
        result["estimated_weight_transfer_proxy_body_lengths"] = safe_round(
            forward_transfer,
            3,
        )

    return result


def _phase_metric_summary(records, start, end, wrist_speed, batting_hand, np):
    """Metric-specific phase summary used by V3.1+ engines."""
    subset = records[start : end + 1]
    total_frames = len(subset)

    def values(key):
        return [record.get(key) for record in subset if record.get(key) is not None]

    def mean_value(key, places):
        vals = values(key)
        return safe_round(safe_mean(vals, np), places) if vals else None

    def median_value(key, places):
        vals = values(key)
        return safe_round(safe_median(vals, np), places) if vals else None

    def circular_value(key, places):
        vals = values(key)
        return safe_round(circular_mean_degrees(vals, np), places) if vals else None

    coverage_keys = {
        "left_knee": "left_knee_angle",
        "right_knee": "right_knee_angle",
        "average_knee": "average_knee_angle",
        "left_elbow": "left_elbow_angle",
        "right_elbow": "right_elbow_angle",
        "shoulder_alignment": "shoulder_tilt",
        "hip_alignment": "hip_tilt",
        "trunk": "trunk_lean_degrees",
        "shoulder_hip_separation": "shoulder_hip_separation_3d_proxy_degrees",
        "stance": "normalized_stance_width_frame",
        "head_control": "nose_rel",
        "left_foot_movement": "left_ankle_rel",
        "right_foot_movement": "right_ankle_rel",
        "body_center": "body_center_rel",
    }

    metric_coverage = {}
    for label, key in coverage_keys.items():
        count = sum(record.get(key) is not None for record in subset)
        metric_coverage[label] = coverage_summary(count, total_frames)

    primary_keys = list(coverage_keys.values())
    usable_frames = sum(
        any(record.get(key) is not None for key in primary_keys)
        for record in subset
    )
    complete_core_frames = sum(
        bool(record.get("biomechanics_valid")) for record in subset
    )

    coverage_percentages = [
        item["coverage_percent"] for item in metric_coverage.values()
    ]
    mean_coverage = (
        float(np.mean(coverage_percentages)) if coverage_percentages else 0.0
    )

    if mean_coverage >= 80.0:
        quality_label = "High"
    elif mean_coverage >= 50.0:
        quality_label = "Moderate"
    elif usable_frames > 0:
        quality_label = "Low"
    else:
        quality_label = "Unavailable"

    summary = {
        "valid_biomechanics_frames": int(usable_frames),
        "complete_core_biomechanics_frames": int(complete_core_frames),
        "total_phase_frames": total_frames,
        "phase_biomechanics_quality": quality_label,
        "mean_metric_coverage_percent": round(mean_coverage, 1),
        "metric_coverage": metric_coverage,
        "left_knee_angle_2d_degrees": mean_value("left_knee_angle", 1),
        "right_knee_angle_2d_degrees": mean_value("right_knee_angle", 1),
        "average_knee_angle_2d_degrees": mean_value("average_knee_angle", 1),
        "left_knee_angle_3d_proxy_degrees": mean_value("left_knee_angle_3d", 1),
        "right_knee_angle_3d_proxy_degrees": mean_value("right_knee_angle_3d", 1),
        "left_elbow_angle_2d_degrees": mean_value("left_elbow_angle", 1),
        "right_elbow_angle_2d_degrees": mean_value("right_elbow_angle", 1),
        "normalized_stance_width": median_value("normalized_stance_width_frame", 3),
        "body_scaled_stance_width": median_value("body_scaled_stance_width", 3),
        "shoulder_tilt_degrees": mean_value("shoulder_tilt", 2),
        "hip_tilt_degrees": mean_value("hip_tilt", 2),
        "trunk_lean_2d_degrees": mean_value("trunk_lean_degrees", 1),
        "shoulder_axial_orientation_3d_proxy_degrees": circular_value(
            "shoulder_axial_orientation_3d", 2
        ),
        "hip_axial_orientation_3d_proxy_degrees": circular_value(
            "hip_axial_orientation_3d", 2
        ),
        "shoulder_hip_separation_3d_proxy_degrees": mean_value(
            "shoulder_hip_separation_3d_proxy_degrees", 1
        ),
        "average_landmark_visibility": mean_value("average_visibility", 3),
    }

    nose_values = values("nose_rel")
    left_foot_values = values("left_ankle_rel")
    right_foot_values = values("right_ankle_rel")
    body_center_values = values("body_center_rel")

    summary["head_movement_body_lengths"] = (
        safe_round(displacement_from_baseline(nose_values, np), 3)
        if nose_values
        else None
    )
    summary["left_foot_movement_body_lengths"] = (
        safe_round(displacement_from_baseline(left_foot_values, np), 3)
        if left_foot_values
        else None
    )
    summary["right_foot_movement_body_lengths"] = (
        safe_round(displacement_from_baseline(right_foot_values, np), 3)
        if right_foot_values
        else None
    )
    summary["body_center_movement_body_lengths"] = (
        safe_round(displacement_from_baseline(body_center_values, np), 3)
        if body_center_values
        else None
    )

    speed_slice = wrist_speed[start : end + 1] if wrist_speed is not None else []
    if len(speed_slice):
        summary["mean_wrist_speed_body_lengths_per_second"] = safe_round(
            float(np.mean(speed_slice)), 3
        )
        summary["peak_wrist_speed_body_lengths_per_second"] = safe_round(
            float(np.max(speed_slice)), 3
        )
    else:
        summary["mean_wrist_speed_body_lengths_per_second"] = None
        summary["peak_wrist_speed_body_lengths_per_second"] = None

    front_side, back_side = front_back_sides(batting_hand)
    if front_side:
        front_key = (
            "left_foot_movement_body_lengths"
            if front_side == "Left"
            else "right_foot_movement_body_lengths"
        )
        back_key = (
            "right_foot_movement_body_lengths"
            if front_side == "Left"
            else "left_foot_movement_body_lengths"
        )
        summary["front_foot_side"] = front_side
        summary["back_foot_side"] = back_side
        summary["front_foot_movement_body_lengths"] = summary.get(front_key)
        summary["back_foot_movement_body_lengths"] = summary.get(back_key)
    else:
        summary["front_foot_side"] = None
        summary["back_foot_side"] = None

    return summary


def build_phase_metrics(frame_records, detection, batting_hand, np):
    if not detection or detection.get("status") != "detected":
        return {}

    ranges = detection.get("_ranges", {})
    wrist_speed = detection.get("_wrist_speed")
    phase_metrics = {}

    for phase_name, (start, end) in ranges.items():
        phase_metrics[phase_name] = _phase_metric_summary(
            frame_records,
            start,
            end,
            wrist_speed,
            batting_hand,
            np,
        )

    return phase_metrics


def build_frame_analysis(frame_records, detection):
    """
    Build JSON-safe per-frame biomechanics data for the NeuroBat
    Frame-by-Frame Analysis / Biomechanics Frame Inspector.

    Raw MediaPipe objects are intentionally excluded so this structure can be
    returned directly through Flask jsonify().
    """
    frame_analysis = []

    impact_array_index = None
    selected_start = None
    selected_end = None

    if detection and detection.get("status") == "detected":
        impact_array_index = detection.get("_impact_array_index")
        ranges = detection.get("_ranges", {})

        if ranges:
            selected_start = ranges.get(
                "setup",
                ranges.get("backlift", (None, None)),
            )[0]

            selected_end = ranges.get(
                "follow_through",
                ranges.get("impact", (None, None)),
            )[1]

    def rounded_point(point, places=4):
        if point is None:
            return None
        try:
            return [
                round(float(point[0]), places),
                round(float(point[1]), places),
            ]
        except (TypeError, ValueError, IndexError):
            return None

    for array_index, record in enumerate(frame_records):
        phase_name = phase_name_for_array_index(
            array_index,
            detection,
        )

        in_selected_stroke = (
            selected_start is not None
            and selected_end is not None
            and selected_start <= array_index <= selected_end
        )

        frame_analysis.append({
            "array_index": array_index,
            "frame_index": int(record.get("frame_index", array_index + 1)),
            "time_seconds": safe_round(record.get("time_seconds"), 4),

            "phase": phase_name,
            "in_selected_stroke": bool(in_selected_stroke),
            "is_estimated_impact_frame": bool(
                impact_array_index is not None
                and array_index == impact_array_index
            ),

            "pose_detected": bool(record.get("pose_detected")),
            "metric_usable": bool(record.get("metric_usable")),
            "complete_core_biomechanics": bool(
                record.get("biomechanics_valid")
            ),
            "average_landmark_visibility": safe_round(
                record.get("average_visibility"),
                3,
            ),
            "metric_validity": dict(
                record.get("metric_validity") or {}
            ),

            "left_knee_angle_2d_degrees": safe_round(
                record.get("left_knee_angle"),
                1,
            ),
            "right_knee_angle_2d_degrees": safe_round(
                record.get("right_knee_angle"),
                1,
            ),
            "average_knee_angle_2d_degrees": safe_round(
                record.get("average_knee_angle"),
                1,
            ),

            "left_elbow_angle_2d_degrees": safe_round(
                record.get("left_elbow_angle"),
                1,
            ),
            "right_elbow_angle_2d_degrees": safe_round(
                record.get("right_elbow_angle"),
                1,
            ),

            "shoulder_tilt_degrees": safe_round(
                record.get("shoulder_tilt"),
                2,
            ),
            "hip_tilt_degrees": safe_round(
                record.get("hip_tilt"),
                2,
            ),
            "trunk_lean_2d_degrees": safe_round(
                record.get("trunk_lean_degrees"),
                1,
            ),

            "left_knee_angle_3d_proxy_degrees": safe_round(
                record.get("left_knee_angle_3d"),
                1,
            ),
            "right_knee_angle_3d_proxy_degrees": safe_round(
                record.get("right_knee_angle_3d"),
                1,
            ),
            "shoulder_axial_orientation_3d_proxy_degrees": safe_round(
                record.get("shoulder_axial_orientation_3d"),
                2,
            ),
            "hip_axial_orientation_3d_proxy_degrees": safe_round(
                record.get("hip_axial_orientation_3d"),
                2,
            ),
            "shoulder_hip_separation_3d_proxy_degrees": safe_round(
                record.get(
                    "shoulder_hip_separation_3d_proxy_degrees"
                ),
                1,
            ),

            "normalized_stance_width": safe_round(
                record.get("normalized_stance_width_frame"),
                3,
            ),
            "body_scaled_stance_width": safe_round(
                record.get("body_scaled_stance_width"),
                3,
            ),

            "head_position_body_relative": rounded_point(
                record.get("nose_rel")
            ),
            "left_foot_position_body_relative": rounded_point(
                record.get("left_ankle_rel")
            ),
            "right_foot_position_body_relative": rounded_point(
                record.get("right_ankle_rel")
            ),
            "body_center_position_body_units": rounded_point(
                record.get("body_center_rel")
            ),
        })

    return frame_analysis


# =============================================================================
# BACKWARD-COMPATIBLE SUMMARY HELPERS
# =============================================================================


def legacy_stance_category(avg_stance_width):
    if avg_stance_width <= STANCE_WIDTH_THRESHOLDS["Very Narrow"]:
        return "Very Narrow"
    if avg_stance_width <= STANCE_WIDTH_THRESHOLDS["Narrow"]:
        return "Narrow"
    if avg_stance_width <= STANCE_WIDTH_THRESHOLDS["Balanced"]:
        return "Balanced"
    if avg_stance_width <= STANCE_WIDTH_THRESHOLDS["Wide"]:
        return "Wide"
    return "Very Wide"


def legacy_prototype_score(
    avg_stance_width,
    avg_shoulder_diff,
    head_stability,
    average_knee_angle,
):
    """
    Preserve the old prototype score so existing UI/history does not break.

    These thresholds are not validated universal cricket technique standards.
    """
    score = 0

    if 0.20 <= avg_stance_width <= 0.45:
        score += 1
    if avg_shoulder_diff < 0.03:
        score += 1
    if head_stability < 0.05:
        score += 1
    if (
        average_knee_angle is not None
        and 130 <= average_knee_angle <= 170
    ):
        score += 1

    if score == 4:
        rating = "Excellent"
    elif score == 3:
        rating = "Good"
    elif score == 2:
        rating = "Average"
    else:
        rating = "Needs Improvement"

    return score, rating


# =============================================================================
# ANNOTATED VIDEO
# =============================================================================


def write_annotated_video(
    path,
    frame_records,
    phase_detection,
    cv2,
    mp_pose,
    mp_drawing,
    width,
    height,
    fps,
):
    writer = None
    output_path = None

    try:
        writer, output_filename, output_path, output_codec = create_video_writer(
            cv2, width, height, fps
        )

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError("Unable to reopen uploaded video for annotation.")

        try:
            for array_index, record in enumerate(frame_records):
                success, frame = cap.read()
                if not success:
                    break

                pose_landmarks = record.get("pose_landmarks")
                if pose_landmarks is not None:
                    mp_drawing.draw_landmarks(
                        frame,
                        pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(
                            color=(255, 80, 80), thickness=3, circle_radius=4
                        ),
                        connection_drawing_spec=mp_drawing.DrawingSpec(
                            color=(80, 200, 255), thickness=3, circle_radius=2
                        ),
                    )

                phase_name = phase_name_for_array_index(array_index, phase_detection)
                phase_label = (
                    phase_name.replace("_", " ").title()
                    if phase_name
                    else "Outside Selected Stroke"
                )

                cv2.rectangle(frame, (0, 0), (width, 82), (5, 15, 35), -1)
                cv2.putText(
                    frame,
                    "NeuroBat Biomechanics Engine V3.2",
                    (18, 33),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.76,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                confidence_label = phase_detection.get("confidence_label", "Low")
                cv2.putText(
                    frame,
                    f"Phase: {phase_label} | Detection Confidence: {confidence_label}",
                    (18, 64),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (96, 165, 250),
                    2,
                    cv2.LINE_AA,
                )

                y = 112
                overlay_items = [
                    ("Left Knee", record.get("left_knee_angle")),
                    ("Right Knee", record.get("right_knee_angle")),
                    ("Left Elbow", record.get("left_elbow_angle")),
                    ("Right Elbow", record.get("right_elbow_angle")),
                    ("Trunk Lean", record.get("trunk_lean_degrees")),
                ]

                for label, value in overlay_items:
                    if value is None:
                        continue
                    cv2.putText(
                        frame,
                        f"{label}: {value:.1f} deg",
                        (25, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.56,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    y += 25

                stance_value = record.get("normalized_stance_width_frame")
                if stance_value is not None:
                    cv2.putText(
                        frame,
                        f"Stance Ratio: {stance_value:.2f}",
                        (25, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.56,
                        (96, 165, 250),
                        2,
                        cv2.LINE_AA,
                    )
                    y += 25

                if phase_name == "impact":
                    cv2.putText(
                        frame,
                        "ESTIMATED IMPACT WINDOW (POSE PROXY)",
                        (25, y + 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.60,
                        (0, 165, 255),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.putText(
                    frame,
                    f"Frame: {record['frame_index']} | Time: {record['time_seconds']:.2f}s",
                    (25, height - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.56,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                writer.write(frame)
        finally:
            cap.release()

        writer.release()
        writer = None
        return output_filename, output_path, output_codec

    except Exception:
        if writer is not None:
            writer.release()
        remove_output_safely(output_path)
        raise


# =============================================================================
# MAIN ENGINE
# =============================================================================


def process_video(
    path,
    max_frames=MAX_ANALYSIS_FRAMES,
    batting_hand=None,
    player_height_cm=None,
    player_weight_kg=None,
    age_group=None,
):
    """Run NeuroBat Biomechanics Engine V3.2.1 on one uploaded batting video."""

    try:
        import cv2
        import mediapipe as mp
        import numpy as np
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load NeuroBat video-analysis dependencies: {exc}"
        ) from exc

    batting_hand = normalize_batting_hand(batting_hand)

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.60,
        min_tracking_confidence=0.60,
    )

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        pose.close()
        raise RuntimeError("Unable to open uploaded video.")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if width <= 0 or height <= 0:
        cap.release()
        pose.close()
        raise RuntimeError("Invalid video dimensions.")

    try:
        max_frames = int(max_frames)
    except (TypeError, ValueError):
        max_frames = MAX_ANALYSIS_FRAMES
    max_frames = max(1, max_frames)

    frame_records = []
    frames_processed = 0
    poses_detected = 0
    valid_biomechanics_frames = 0
    metric_usable_frames = 0

    raw_stance_widths = []
    shoulder_diffs = []
    head_x_positions = []

    left_knee_angles = []
    right_knee_angles = []
    left_elbow_angles = []
    right_elbow_angles = []
    trunk_lean_angles = []
    shoulder_hip_separation_proxies = []

    left_knee_angles_3d = []
    right_knee_angles_3d = []
    shoulder_tilts = []
    hip_tilts = []
    shoulder_axial_orientations_3d = []
    hip_axial_orientations_3d = []

    shoulder_widths_pixels = []
    ankle_separations_pixels = []
    normalized_stance_frame_values = []
    body_scaled_stance_values = []
    landmark_visibility_values = []

    nose_positions_pixels = []
    left_ankle_positions_pixels = []
    right_ankle_positions_pixels = []

    nose_relative_positions = []
    left_ankle_relative_positions = []
    right_ankle_relative_positions = []
    body_center_relative_positions = []

    try:
        while cap.isOpened() and frames_processed < max_frames:
            success, frame = cap.read()
            if not success:
                break

            frames_processed += 1
            frame_index = frames_processed
            time_seconds = (frame_index - 1) / fps

            record = {
                "frame_index": frame_index,
                "time_seconds": time_seconds,
                "pose_detected": False,
                "biomechanics_valid": False,
                "metric_usable": False,
                "phase_valid": False,
                "pose_landmarks": None,
                "average_visibility": 0.0,
                "metric_validity": {},
                "wrist_rel": None,
                "nose_rel": None,
                "left_ankle_rel": None,
                "right_ankle_rel": None,
                "body_center_rel": None,
                "left_knee_angle": None,
                "right_knee_angle": None,
                "average_knee_angle": None,
                "left_elbow_angle": None,
                "right_elbow_angle": None,
                "left_knee_angle_3d": None,
                "right_knee_angle_3d": None,
                "shoulder_tilt": None,
                "hip_tilt": None,
                "trunk_lean_degrees": None,
                "shoulder_axial_orientation_3d": None,
                "hip_axial_orientation_3d": None,
                "shoulder_hip_separation_3d_proxy_degrees": None,
                "normalized_stance_width_frame": None,
                "body_scaled_stance_width": None,
            }

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if not results.pose_landmarks:
                frame_records.append(record)
                continue

            poses_detected += 1
            record["pose_detected"] = True
            record["pose_landmarks"] = results.pose_landmarks

            landmarks = results.pose_landmarks.landmark

            nose = landmarks[mp_pose.PoseLandmark.NOSE]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
            right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

            core_landmarks = [
                nose,
                left_shoulder,
                right_shoulder,
                left_hip,
                right_hip,
                left_knee,
                right_knee,
                left_ankle,
                right_ankle,
            ]
            analysis_landmarks = core_landmarks + [
                left_elbow,
                right_elbow,
                left_wrist,
                right_wrist,
            ]

            visibility_values = [float(lm.visibility) for lm in analysis_landmarks]
            average_visibility = sum(visibility_values) / len(visibility_values)
            record["average_visibility"] = average_visibility
            landmark_visibility_values.append(average_visibility)

            # -----------------------------------------------------------------
            # Metric-specific landmark quality
            # -----------------------------------------------------------------

            left_knee_valid = landmarks_visible([left_hip, left_knee, left_ankle])
            right_knee_valid = landmarks_visible([right_hip, right_knee, right_ankle])
            left_elbow_valid = landmarks_visible(
                [left_shoulder, left_elbow, left_wrist]
            )
            right_elbow_valid = landmarks_visible(
                [right_shoulder, right_elbow, right_wrist]
            )
            shoulder_valid = landmarks_visible([left_shoulder, right_shoulder])
            hip_valid = landmarks_visible([left_hip, right_hip])
            trunk_valid = shoulder_valid and hip_valid
            stance_valid = landmarks_visible(
                [left_ankle, right_ankle, left_shoulder, right_shoulder]
            )
            body_reference_valid = landmarks_visible(
                [left_shoulder, right_shoulder, left_hip, right_hip],
                BODY_REFERENCE_MIN_VISIBILITY,
            )
            head_valid = landmark_visible(nose) and body_reference_valid
            left_foot_valid = landmark_visible(left_ankle) and body_reference_valid
            right_foot_valid = landmark_visible(right_ankle) and body_reference_valid
            complete_core_valid = landmarks_visible(core_landmarks)

            record["metric_validity"] = {
                "left_knee": left_knee_valid,
                "right_knee": right_knee_valid,
                "left_elbow": left_elbow_valid,
                "right_elbow": right_elbow_valid,
                "shoulder_alignment": shoulder_valid,
                "hip_alignment": hip_valid,
                "trunk": trunk_valid,
                "stance": stance_valid,
                "head_control": head_valid,
                "left_foot_movement": left_foot_valid,
                "right_foot_movement": right_foot_valid,
                "body_reference": body_reference_valid,
            }

            if complete_core_valid:
                valid_biomechanics_frames += 1
                record["biomechanics_valid"] = True

            if any(
                [
                    left_knee_valid,
                    right_knee_valid,
                    left_elbow_valid,
                    right_elbow_valid,
                    shoulder_valid,
                    hip_valid,
                    trunk_valid,
                    stance_valid,
                    head_valid,
                    left_foot_valid,
                    right_foot_valid,
                ]
            ):
                metric_usable_frames += 1
                record["metric_usable"] = True

            # -----------------------------------------------------------------
            # Pixel geometry
            # -----------------------------------------------------------------

            nose_px = landmark_to_pixel(nose, width, height)
            left_shoulder_px = landmark_to_pixel(left_shoulder, width, height)
            right_shoulder_px = landmark_to_pixel(right_shoulder, width, height)
            left_elbow_px = landmark_to_pixel(left_elbow, width, height)
            right_elbow_px = landmark_to_pixel(right_elbow, width, height)
            left_wrist_px = landmark_to_pixel(left_wrist, width, height)
            right_wrist_px = landmark_to_pixel(right_wrist, width, height)
            left_hip_px = landmark_to_pixel(left_hip, width, height)
            right_hip_px = landmark_to_pixel(right_hip, width, height)
            left_knee_px = landmark_to_pixel(left_knee, width, height)
            right_knee_px = landmark_to_pixel(right_knee, width, height)
            left_ankle_px = landmark_to_pixel(left_ankle, width, height)
            right_ankle_px = landmark_to_pixel(right_ankle, width, height)

            shoulder_center_px = midpoint(left_shoulder_px, right_shoulder_px)
            hip_center_px = midpoint(left_hip_px, right_hip_px)
            body_center_px = midpoint(shoulder_center_px, hip_center_px)

            shoulder_width_px = euclidean_distance(
                left_shoulder_px, right_shoulder_px
            )
            hip_width_px = euclidean_distance(left_hip_px, right_hip_px)
            torso_length_px = euclidean_distance(shoulder_center_px, hip_center_px)

            body_scale_px = max(
                torso_length_px,
                0.5 * (shoulder_width_px + hip_width_px),
                5.0,
            )

            # -----------------------------------------------------------------
            # Body-normalized movement features
            # -----------------------------------------------------------------

            if body_reference_valid:
                record["body_center_rel"] = point_in_body_units(
                    body_center_px, body_scale_px
                )
                if record["body_center_rel"] is not None:
                    body_center_relative_positions.append(record["body_center_rel"])

            if head_valid:
                record["nose_rel"] = relative_point(
                    nose_px, hip_center_px, body_scale_px
                )
                if record["nose_rel"] is not None:
                    nose_relative_positions.append(record["nose_rel"])
                    nose_positions_pixels.append(nose_px)
                    head_x_positions.append(float(nose.x))

            if left_foot_valid:
                record["left_ankle_rel"] = relative_point(
                    left_ankle_px, hip_center_px, body_scale_px
                )
                if record["left_ankle_rel"] is not None:
                    left_ankle_relative_positions.append(record["left_ankle_rel"])
                    left_ankle_positions_pixels.append(left_ankle_px)

            if right_foot_valid:
                record["right_ankle_rel"] = relative_point(
                    right_ankle_px, hip_center_px, body_scale_px
                )
                if record["right_ankle_rel"] is not None:
                    right_ankle_relative_positions.append(record["right_ankle_rel"])
                    right_ankle_positions_pixels.append(right_ankle_px)

            # -----------------------------------------------------------------
            # Wrist signal for phase detection
            # -----------------------------------------------------------------

            left_wrist_visibility = float(left_wrist.visibility)
            right_wrist_visibility = float(right_wrist.visibility)
            wrist_px = None

            if body_reference_valid:
                if (
                    left_wrist_visibility >= PHASE_MIN_WRIST_VISIBILITY
                    and right_wrist_visibility >= PHASE_MIN_WRIST_VISIBILITY
                ):
                    wrist_px = midpoint(left_wrist_px, right_wrist_px)
                elif left_wrist_visibility >= PHASE_SINGLE_WRIST_HIGH_VISIBILITY:
                    wrist_px = left_wrist_px
                elif right_wrist_visibility >= PHASE_SINGLE_WRIST_HIGH_VISIBILITY:
                    wrist_px = right_wrist_px

            if wrist_px is not None:
                record["wrist_rel"] = relative_point(
                    wrist_px, hip_center_px, body_scale_px
                )
                record["phase_valid"] = record["wrist_rel"] is not None

            # -----------------------------------------------------------------
            # 2D biomechanics
            # -----------------------------------------------------------------

            if left_knee_valid:
                value = calculate_angle(left_hip_px, left_knee_px, left_ankle_px)
                record["left_knee_angle"] = value
                left_knee_angles.append(value)

            if right_knee_valid:
                value = calculate_angle(right_hip_px, right_knee_px, right_ankle_px)
                record["right_knee_angle"] = value
                right_knee_angles.append(value)

            if (
                record["left_knee_angle"] is not None
                and record["right_knee_angle"] is not None
            ):
                record["average_knee_angle"] = (
                    record["left_knee_angle"] + record["right_knee_angle"]
                ) / 2.0

            if left_elbow_valid:
                value = calculate_angle(
                    left_shoulder_px, left_elbow_px, left_wrist_px
                )
                record["left_elbow_angle"] = value
                left_elbow_angles.append(value)

            if right_elbow_valid:
                value = calculate_angle(
                    right_shoulder_px, right_elbow_px, right_wrist_px
                )
                record["right_elbow_angle"] = value
                right_elbow_angles.append(value)

            if shoulder_valid:
                shoulder_tilt = calculate_line_angle(
                    left_shoulder_px, right_shoulder_px
                )
                record["shoulder_tilt"] = shoulder_tilt
                shoulder_tilts.append(shoulder_tilt)

                shoulder_difference = abs(
                    float(left_shoulder.y) - float(right_shoulder.y)
                )
                shoulder_diffs.append(shoulder_difference)

                if shoulder_width_px > 5:
                    shoulder_widths_pixels.append(shoulder_width_px)

            if hip_valid:
                hip_tilt = calculate_line_angle(left_hip_px, right_hip_px)
                record["hip_tilt"] = hip_tilt
                hip_tilts.append(hip_tilt)

            if trunk_valid:
                trunk_lean = calculate_trunk_lean(shoulder_center_px, hip_center_px)
                record["trunk_lean_degrees"] = trunk_lean
                if trunk_lean is not None:
                    trunk_lean_angles.append(trunk_lean)

            if stance_valid:
                raw_stance_width = abs(float(left_ankle.x) - float(right_ankle.x))
                ankle_separation_px = abs(left_ankle_px[0] - right_ankle_px[0])

                raw_stance_widths.append(raw_stance_width)
                if ankle_separation_px > 1:
                    ankle_separations_pixels.append(ankle_separation_px)

                if shoulder_width_px > 5:
                    frame_stance_ratio = ankle_separation_px / shoulder_width_px
                    record["normalized_stance_width_frame"] = frame_stance_ratio
                    normalized_stance_frame_values.append(frame_stance_ratio)

            if (
                landmark_visible(left_ankle)
                and landmark_visible(right_ankle)
                and body_reference_valid
            ):
                ankle_separation_px = abs(left_ankle_px[0] - right_ankle_px[0])
                body_scaled_stance = ankle_separation_px / body_scale_px
                record["body_scaled_stance_width"] = body_scaled_stance
                body_scaled_stance_values.append(body_scaled_stance)

            # -----------------------------------------------------------------
            # Optional MediaPipe world-landmark proxies
            # -----------------------------------------------------------------

            if results.pose_world_landmarks:
                world = results.pose_world_landmarks.landmark

                l_shoulder_w = landmark_to_world_tuple(
                    world[mp_pose.PoseLandmark.LEFT_SHOULDER]
                )
                r_shoulder_w = landmark_to_world_tuple(
                    world[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                )
                l_hip_w = landmark_to_world_tuple(
                    world[mp_pose.PoseLandmark.LEFT_HIP]
                )
                r_hip_w = landmark_to_world_tuple(
                    world[mp_pose.PoseLandmark.RIGHT_HIP]
                )
                l_knee_w = landmark_to_world_tuple(
                    world[mp_pose.PoseLandmark.LEFT_KNEE]
                )
                r_knee_w = landmark_to_world_tuple(
                    world[mp_pose.PoseLandmark.RIGHT_KNEE]
                )
                l_ankle_w = landmark_to_world_tuple(
                    world[mp_pose.PoseLandmark.LEFT_ANKLE]
                )
                r_ankle_w = landmark_to_world_tuple(
                    world[mp_pose.PoseLandmark.RIGHT_ANKLE]
                )

                if left_knee_valid:
                    left_knee_3d = calculate_angle_3d(
                        l_hip_w, l_knee_w, l_ankle_w
                    )
                    record["left_knee_angle_3d"] = left_knee_3d
                    if left_knee_3d is not None:
                        left_knee_angles_3d.append(left_knee_3d)

                if right_knee_valid:
                    right_knee_3d = calculate_angle_3d(
                        r_hip_w, r_knee_w, r_ankle_w
                    )
                    record["right_knee_angle_3d"] = right_knee_3d
                    if right_knee_3d is not None:
                        right_knee_angles_3d.append(right_knee_3d)

                shoulder_axial = None
                hip_axial = None

                if shoulder_valid:
                    shoulder_axial = calculate_axial_orientation_proxy(
                        l_shoulder_w, r_shoulder_w
                    )
                    record["shoulder_axial_orientation_3d"] = shoulder_axial
                    if shoulder_axial is not None:
                        shoulder_axial_orientations_3d.append(shoulder_axial)

                if hip_valid:
                    hip_axial = calculate_axial_orientation_proxy(l_hip_w, r_hip_w)
                    record["hip_axial_orientation_3d"] = hip_axial
                    if hip_axial is not None:
                        hip_axial_orientations_3d.append(hip_axial)

                separation = angular_difference_degrees(shoulder_axial, hip_axial)
                record["shoulder_hip_separation_3d_proxy_degrees"] = separation
                if separation is not None:
                    shoulder_hip_separation_proxies.append(separation)

            frame_records.append(record)

    finally:
        cap.release()
        pose.close()

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    if frames_processed == 0:
        return {"error": "No frames could be read from the video."}

    if poses_detected == 0:
        return {"error": "No human pose was detected in the video."}

    if metric_usable_frames == 0:
        return {
            "error": (
                "Pose landmarks were detected, but no primary biomechanics metric "
                "had sufficient landmark visibility for reliable measurement."
            )
        }

    # -------------------------------------------------------------------------
    # Phase detection / phase summaries
    # -------------------------------------------------------------------------

    phase_detection_internal = detect_batting_phases(frame_records, fps, np)
    phase_metrics = build_phase_metrics(
        frame_records,
        phase_detection_internal,
        batting_hand,
        np,
    )
    movement_metrics = build_cross_phase_movement_metrics(
        frame_records,
        phase_detection_internal,
        batting_hand,
        np,
    )

    # JSON-safe frame-by-frame data for the frontend Frame Inspector.
    frame_analysis = build_frame_analysis(
        frame_records,
        phase_detection_internal,
    )

    # -------------------------------------------------------------------------
    # Whole-clip summaries
    # -------------------------------------------------------------------------

    # Missing landmark data must remain unavailable (None), not become a
    # false 0.0-degree biomechanical measurement.
    avg_left_knee_angle = (
        safe_round(safe_mean(left_knee_angles, np), 1)
        if left_knee_angles
        else None
    )
    avg_right_knee_angle = (
        safe_round(safe_mean(right_knee_angles, np), 1)
        if right_knee_angles
        else None
    )

    # A bilateral average is reported only when both sides were actually
    # observed with sufficient landmark quality.
    if (
        avg_left_knee_angle is not None
        and avg_right_knee_angle is not None
    ):
        average_knee_angle = safe_round(
            (avg_left_knee_angle + avg_right_knee_angle) / 2.0,
            1,
        )
    else:
        average_knee_angle = None

    avg_left_elbow_angle = (
        safe_round(safe_mean(left_elbow_angles, np), 1)
        if left_elbow_angles
        else None
    )
    avg_right_elbow_angle = (
        safe_round(safe_mean(right_elbow_angles, np), 1)
        if right_elbow_angles
        else None
    )
    avg_trunk_lean = (
        safe_round(safe_mean(trunk_lean_angles, np), 1)
        if trunk_lean_angles
        else None
    )
    avg_shoulder_hip_separation = (
        safe_round(safe_mean(shoulder_hip_separation_proxies, np), 1)
        if shoulder_hip_separation_proxies
        else None
    )

    avg_stance_width = round(safe_mean(raw_stance_widths, np), 3)
    median_shoulder_width = safe_median(shoulder_widths_pixels, np)
    median_ankle_separation = safe_median(ankle_separations_pixels, np)

    avg_normalized_stance_width = (
        round(median_ankle_separation / median_shoulder_width, 3)
        if median_shoulder_width > 0
        else 0.0
    )

    avg_shoulder_diff = round(safe_mean(shoulder_diffs, np), 3)
    head_stability = round(safe_std(head_x_positions, np), 3)
    avg_shoulder_tilt = round(safe_mean(shoulder_tilts, np), 2)
    avg_hip_tilt = round(safe_mean(hip_tilts, np), 2)
    average_visibility = round(safe_mean(landmark_visibility_values, np), 3)

    mean_shoulder_width = safe_mean(shoulder_widths_pixels, np)

    def legacy_pixel_movement(points):
        if mean_shoulder_width <= 1 or not points:
            return 0.0

        baseline_count = min(5, len(points))
        baseline = (
            safe_mean([p[0] for p in points[:baseline_count]], np),
            safe_mean([p[1] for p in points[:baseline_count]], np),
        )
        distances = [
            euclidean_distance(baseline, point) / mean_shoulder_width
            for point in points
        ]
        return round(max(distances), 3) if distances else 0.0

    head_displacement = legacy_pixel_movement(nose_positions_pixels)
    left_foot_movement = legacy_pixel_movement(left_ankle_positions_pixels)
    right_foot_movement = legacy_pixel_movement(right_ankle_positions_pixels)

    if left_foot_movement > right_foot_movement:
        moving_foot = "Left"
        front_foot_movement = left_foot_movement
    elif right_foot_movement > left_foot_movement:
        moving_foot = "Right"
        front_foot_movement = right_foot_movement
    else:
        moving_foot = "Equal"
        front_foot_movement = left_foot_movement

    pose_detection_rate = round((poses_detected / frames_processed) * 100.0, 1)
    valid_frame_rate = round(
        (valid_biomechanics_frames / frames_processed) * 100.0, 1
    )
    metric_usable_frame_rate = round(
        (metric_usable_frames / frames_processed) * 100.0, 1
    )

    score, rating = legacy_prototype_score(
        avg_stance_width,
        avg_shoulder_diff,
        head_stability,
        average_knee_angle,
    )
    stance_category = legacy_stance_category(avg_stance_width)

    v3_metrics = {
        "head_displacement_body_relative": safe_round(
            displacement_from_baseline(nose_relative_positions, np), 3
        ),
        "left_foot_movement_body_relative": safe_round(
            displacement_from_baseline(left_ankle_relative_positions, np), 3
        ),
        "right_foot_movement_body_relative": safe_round(
            displacement_from_baseline(right_ankle_relative_positions, np), 3
        ),
        "body_center_movement_body_relative": safe_round(
            displacement_from_baseline(body_center_relative_positions, np), 3
        ),
        "median_frame_stance_width_ratio": safe_round(
            safe_median(normalized_stance_frame_values, np), 3
        ),
        "median_body_scaled_stance_width": safe_round(
            safe_median(body_scaled_stance_values, np), 3
        ),
        "left_knee_angle_3d_proxy_degrees": (
            safe_round(safe_mean(left_knee_angles_3d, np), 1)
            if left_knee_angles_3d
            else None
        ),
        "right_knee_angle_3d_proxy_degrees": (
            safe_round(safe_mean(right_knee_angles_3d, np), 1)
            if right_knee_angles_3d
            else None
        ),
        "left_elbow_angle_2d_degrees": avg_left_elbow_angle,
        "right_elbow_angle_2d_degrees": avg_right_elbow_angle,
        "trunk_lean_2d_degrees": avg_trunk_lean,
        "shoulder_axial_orientation_3d_proxy_degrees": safe_round(
            circular_mean_degrees(shoulder_axial_orientations_3d, np), 2
        ),
        "hip_axial_orientation_3d_proxy_degrees": safe_round(
            circular_mean_degrees(hip_axial_orientations_3d, np), 2
        ),
        "shoulder_hip_separation_3d_proxy_degrees": avg_shoulder_hip_separation,
        **movement_metrics,
    }

    # -------------------------------------------------------------------------
    # Annotated output
    # -------------------------------------------------------------------------

    output_filename, output_path, output_codec = write_annotated_video(
        path,
        frame_records,
        phase_detection_internal,
        cv2,
        mp_pose,
        mp_drawing,
        width,
        height,
        fps,
    )

    analysis_truncated = bool(
        total_source_frames > 0
        and frames_processed < total_source_frames
        and frames_processed >= max_frames
    )

    front_side, back_side = front_back_sides(batting_hand)

    # -------------------------------------------------------------------------
    # Final API response
    # -------------------------------------------------------------------------

    return {
        "frames_processed": frames_processed,
        "poses_detected": poses_detected,
        "pose_detection_rate": pose_detection_rate,
        "valid_biomechanics_frames": valid_biomechanics_frames,
        "valid_biomechanics_frame_rate": valid_frame_rate,
        "average_landmark_visibility": average_visibility,
        "metric_usable_frames": metric_usable_frames,
        "metric_usable_frame_rate": metric_usable_frame_rate,
        "biomechanics_quality_model": "metric_specific_landmark_visibility_v3_2",
        "left_knee_angle": avg_left_knee_angle,
        "right_knee_angle": avg_right_knee_angle,
        "average_knee_angle": average_knee_angle,
        "left_elbow_angle": avg_left_elbow_angle,
        "right_elbow_angle": avg_right_elbow_angle,
        "trunk_lean_degrees": avg_trunk_lean,
        "shoulder_hip_separation_proxy_degrees": avg_shoulder_hip_separation,
        "stance_width": avg_stance_width,
        "normalized_stance_width": avg_normalized_stance_width,
        "median_shoulder_width_px": round(median_shoulder_width, 2),
        "median_ankle_separation_px": round(median_ankle_separation, 2),
        "stance_category": stance_category,
        "shoulder_alignment": avg_shoulder_diff,
        "shoulder_tilt_degrees": avg_shoulder_tilt,
        "hip_tilt_degrees": avg_hip_tilt,
        "head_stability": head_stability,
        "head_displacement": head_displacement,
        "left_foot_movement": left_foot_movement,
        "right_foot_movement": right_foot_movement,
        "moving_foot_proxy": moving_foot,
        "front_foot_movement_proxy": front_foot_movement,
        "stride_displacement_body_lengths": movement_metrics.get(
            "stride_displacement_body_lengths"
        ),
        "balance_sway_body_lengths": movement_metrics.get(
            "balance_sway_body_lengths"
        ),
        "balance_stability_proxy": movement_metrics.get(
            "balance_stability_proxy"
        ),
        "estimated_weight_transfer_proxy_body_lengths": movement_metrics.get(
            "estimated_weight_transfer_proxy_body_lengths"
        ),
        "prototype_score": score,
        "prototype_score_percent": int(score / 4 * 100),
        "stance_rating": rating,
        "processed_video_filename": output_filename,
        "processed_video_url": "/outputs/" + output_filename,
        "processed_video_codec": output_codec,
        "analysis_engine_version": ENGINE_VERSION,
        "batting_phase_detection": public_phase_detection(
            phase_detection_internal
        ),
        "phase_metrics": phase_metrics,

        # Frame-by-frame analysis data used by the frontend inspector.
        "frame_analysis_meta": {
            "total_frames": len(frame_analysis),
            "fps": round(fps, 3),
            "frame_step_seconds": round(1.0 / fps, 6),
            "source": "processed_video_timeline",
        },
        "frame_analysis": frame_analysis,

        "v3_metrics": v3_metrics,
        "analysis_scope": {
            "fps": round(fps, 3),
            "source_frame_count": total_source_frames,
            "frames_analyzed": frames_processed,
            "max_frames_setting": max_frames,
            "analysis_truncated": analysis_truncated,
            "preferred_clip_type": "single clearly visible batting stroke",
        },
        "player_context": {
            "batting_hand": batting_hand,
            "front_foot_side": front_side,
            "back_foot_side": back_side,
            "height_cm": player_height_cm,
            "weight_kg": player_weight_kg,
            "age_group": age_group,
            "anthropometric_adjustment_applied": False,
            "normalization_note": (
                "V3.2 uses body-normalized movement features and metric-specific "
                "landmark quality checks. It does not invent age/height/weight-"
                "specific ideal technique thresholds."
            ),
        },
        "prototype_score_model": "legacy_v2_rule_based_compatibility",
        "prototype_score_disclaimer": (
            "The current top-level score is retained for V2 dashboard/history "
            "compatibility. Its thresholds are prototype rules and require "
            "expert/empirical validation before being treated as coaching standards."
        ),
        "impact_timing_disclaimer": (
            "Impact timing is a pose-based wrist-kinematics proxy. Verified "
            "bat-ball contact requires bat/ball tracking or synchronized ground truth."
        ),
        "advanced_biomechanics_disclaimer": (
            "Trunk lean is a 2D image-plane measurement. Shoulder/hip rotation, "
            "shoulder-hip separation, balance, stride and weight transfer are "
            "video-derived proxies and are not laboratory force/motion-capture measures."
        ),
    }