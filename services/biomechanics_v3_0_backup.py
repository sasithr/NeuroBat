"""
NeuroBat Biomechanics Engine V3.0
=================================

Professional pose-based cricket batting biomechanics analysis with temporal
batting-phase detection.

Pipeline
--------
1. Decode uploaded video.
2. Estimate 33 MediaPipe Pose landmarks frame-by-frame.
3. Apply landmark visibility quality gates.
4. Calculate 2D image-plane biomechanics plus optional 3D world-landmark
   measurements where MediaPipe provides them.
5. Build body-normalized wrist kinematics.
6. Detect the dominant batting stroke and segment:
       Setup -> Backlift -> Downswing -> Estimated Impact -> Follow-through
7. Calculate phase-specific biomechanics.
8. Generate an annotated processed video.
9. Preserve the V2.1 top-level fields used by the current NeuroBat database and
   dashboards.

Scientific boundary
-------------------
The V3 phase detector is pose-only. MediaPipe tracks the human body, not the
cricket bat or ball. Therefore the "impact" event in V3 is explicitly an
ESTIMATED IMPACT PROXY based on hand/wrist kinematics, not verified bat-ball
contact. Exact impact timing requires bat/ball tracking or instrumented/high-
speed ground truth and should be implemented as a later validation stage.
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

ENGINE_VERSION = "NeuroBat-Biomechanics-V3.0-PhaseDetection"
# Maximum number of frames analyzed from one uploaded video.
MAX_ANALYSIS_FRAMES = 360

# Core pose landmark visibility threshold.
MIN_LANDMARK_VISIBILITY = 0.40

# Wrist-specific thresholds used by batting phase detection.
PHASE_MIN_WRIST_VISIBILITY = 0.25
PHASE_SINGLE_WRIST_HIGH_VISIBILITY = 0.60
# 360 frames allows about 12 s at 30 fps or 6 s at 60 fps. A single, trimmed
# batting stroke is preferred for phase analysis. If a longer clip is supplied,
# the API reports whether analysis was truncated.
MAX_ANALYSIS_FRAMES = 360

# Core-body threshold used for angle/posture measurements.
MIN_LANDMARK_VISIBILITY = 0.40

# Wrist tracking can be more occlusion-prone during a two-handed cricket grip.
# V3 accepts a lower threshold for the phase signal, then penalises confidence
# when wrist coverage is poor.
PHASE_MIN_WRIST_VISIBILITY = 0.25
PHASE_SINGLE_WRIST_HIGH_VISIBILITY = 0.60

# Kept ONLY for backward compatibility with V2 dashboards and historical rows.
# They are prototype thresholds and are NOT claimed as validated coaching norms.
STANCE_WIDTH_THRESHOLDS = {
    "Very Narrow": 0.20,
    "Narrow": 0.30,
    "Balanced": 0.45,
    "Wide": 0.55,
}


# =============================================================================
# GEOMETRY HELPERS
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
    """Calculate 3D internal angle ABC in degrees using vector dot product."""
    ba = (
        a[0] - b[0],
        a[1] - b[1],
        a[2] - b[2],
    )
    bc = (
        c[0] - b[0],
        c[1] - b[1],
        c[2] - b[2],
    )

    norm_ba = math.sqrt(sum(value * value for value in ba))
    norm_bc = math.sqrt(sum(value * value for value in bc))

    if norm_ba <= 1e-9 or norm_bc <= 1e-9:
        return None

    cosine = sum(ba[i] * bc[i] for i in range(3)) / (norm_ba * norm_bc)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def calculate_line_angle(a, b):
    """
    Absolute 2D tilt relative to horizontal.

    0 degrees = horizontal
    90 degrees = vertical
    """
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    angle = abs(math.degrees(math.atan2(dy, dx)))
    if angle > 90:
        angle = 180 - angle
    return angle


def calculate_axial_orientation_proxy(a, b):
    """
    Orientation of a left-right body segment in the MediaPipe world x-z plane.

    This is a monocular 3D POSE PROXY, not laboratory-grade torso rotation.
    """
    dx = b[0] - a[0]
    dz = b[2] - a[2]
    if abs(dx) + abs(dz) <= 1e-9:
        return None
    return math.degrees(math.atan2(dz, dx))


def euclidean_distance(a, b):
    return math.sqrt(
        (a[0] - b[0]) ** 2
        + (a[1] - b[1]) ** 2
    )


def midpoint(a, b):
    return (
        (a[0] + b[0]) / 2.0,
        (a[1] + b[1]) / 2.0,
    )


def landmark_to_pixel(landmark, width, height):
    return (
        float(landmark.x) * width,
        float(landmark.y) * height,
    )


def landmark_to_world_tuple(landmark):
    return (
        float(landmark.x),
        float(landmark.y),
        float(landmark.z),
    )


def relative_point(point, origin, scale):
    if scale <= 1e-9:
        return None
    return (
        (point[0] - origin[0]) / scale,
        (point[1] - origin[1]) / scale,
    )


# =============================================================================
# NUMERIC HELPERS
# =============================================================================


def safe_mean(values, np):
    clean = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not clean:
        return 0.0
    return float(np.mean(clean))


def safe_std(values, np):
    clean = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not clean:
        return 0.0
    return float(np.std(clean))


def safe_median(values, np):
    clean = [float(value) for value in values if value is not None and np.isfinite(value)]
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
    clean = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not clean:
        return None
    radians = np.radians(clean)
    mean_sin = float(np.mean(np.sin(radians)))
    mean_cos = float(np.mean(np.cos(radians)))
    if abs(mean_sin) + abs(mean_cos) <= 1e-9:
        return None
    return float(np.degrees(np.arctan2(mean_sin, mean_cos)))


def displacement_from_baseline(points, np, baseline_count=5):
    """Maximum 2D displacement from the early-phase median point."""
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
# VIDEO WRITER
# =============================================================================


def create_video_writer(cv2, width, height, fps):
    """Create a browser-compatible annotated output writer."""
    candidates = [
        ("webm", "VP80"),
        ("mp4", "mp4v"),
    ]

    for extension, codec in candidates:
        filename = f"processed_{uuid.uuid4().hex[:10]}.{extension}"
        path = os.path.join(OUTPUTS_DIR, filename)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(
            path,
            fourcc,
            fps,
            (width, height),
        )

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
# PHASE-SPECIFIC METRICS
# =============================================================================


def _phase_metric_summary(records, start, end, wrist_speed, batting_hand, np):
    subset = records[start:end + 1]
    valid = [record for record in subset if record.get("biomechanics_valid")]

    summary = {
        "valid_biomechanics_frames": len(valid),
        "total_phase_frames": len(subset),
    }

    if not valid:
        return summary

    def values(key):
        return [record.get(key) for record in valid if record.get(key) is not None]

    summary.update({
        "left_knee_angle_2d_degrees": safe_round(safe_mean(values("left_knee_angle"), np), 1),
        "right_knee_angle_2d_degrees": safe_round(safe_mean(values("right_knee_angle"), np), 1),
        "average_knee_angle_2d_degrees": safe_round(safe_mean(values("average_knee_angle"), np), 1),
        "left_knee_angle_3d_proxy_degrees": safe_round(safe_mean(values("left_knee_angle_3d"), np), 1),
        "right_knee_angle_3d_proxy_degrees": safe_round(safe_mean(values("right_knee_angle_3d"), np), 1),
        "normalized_stance_width": safe_round(safe_median(values("normalized_stance_width_frame"), np), 3),
        "body_scaled_stance_width": safe_round(safe_median(values("body_scaled_stance_width"), np), 3),
        "shoulder_tilt_degrees": safe_round(safe_mean(values("shoulder_tilt"), np), 2),
        "hip_tilt_degrees": safe_round(safe_mean(values("hip_tilt"), np), 2),
        "shoulder_axial_orientation_3d_proxy_degrees": safe_round(circular_mean_degrees(values("shoulder_axial_orientation_3d"), np), 2),
        "hip_axial_orientation_3d_proxy_degrees": safe_round(circular_mean_degrees(values("hip_axial_orientation_3d"), np), 2),
        "average_landmark_visibility": safe_round(safe_mean(values("average_visibility"), np), 3),
        "head_movement_body_lengths": safe_round(displacement_from_baseline(values("nose_rel"), np), 3),
        "left_foot_movement_body_lengths": safe_round(displacement_from_baseline(values("left_ankle_rel"), np), 3),
        "right_foot_movement_body_lengths": safe_round(displacement_from_baseline(values("right_ankle_rel"), np), 3),
    })

    speed_slice = wrist_speed[start:end + 1] if wrist_speed is not None else []
    if len(speed_slice):
        summary["mean_wrist_speed_body_lengths_per_second"] = safe_round(float(np.mean(speed_slice)), 3)
        summary["peak_wrist_speed_body_lengths_per_second"] = safe_round(float(np.max(speed_slice)), 3)
    else:
        summary["mean_wrist_speed_body_lengths_per_second"] = None
        summary["peak_wrist_speed_body_lengths_per_second"] = None

    front_side, back_side = front_back_sides(batting_hand)
    if front_side:
        front_key = "left_foot_movement_body_lengths" if front_side == "Left" else "right_foot_movement_body_lengths"
        back_key = "right_foot_movement_body_lengths" if front_side == "Left" else "left_foot_movement_body_lengths"
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


# =============================================================================
# BACKWARD-COMPATIBLE V2 SUMMARY HELPERS
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


def legacy_prototype_score(avg_stance_width, avg_shoulder_diff, head_stability, average_knee_angle):
    """
    Preserve V2.1 scoring so current dashboards/history do not break.

    IMPORTANT: These rules are prototype compatibility rules, not validated
    universal cricket technique standards.
    """
    score = 0

    if 0.20 <= avg_stance_width <= 0.45:
        score += 1
    if avg_shoulder_diff < 0.03:
        score += 1
    if head_stability < 0.05:
        score += 1
    if 130 <= average_knee_angle <= 170:
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
            cv2,
            width,
            height,
            fps,
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
                            color=(255, 80, 80),
                            thickness=3,
                            circle_radius=4,
                        ),
                        connection_drawing_spec=mp_drawing.DrawingSpec(
                            color=(80, 200, 255),
                            thickness=3,
                            circle_radius=2,
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
                    "NeuroBat Biomechanics Engine V3",
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
                if record.get("left_knee_angle") is not None:
                    cv2.putText(
                        frame,
                        f"Left Knee: {record['left_knee_angle']:.1f} deg",
                        (25, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.62,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    y += 28

                if record.get("right_knee_angle") is not None:
                    cv2.putText(
                        frame,
                        f"Right Knee: {record['right_knee_angle']:.1f} deg",
                        (25, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.62,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    y += 28

                if record.get("normalized_stance_width_frame") is not None:
                    cv2.putText(
                        frame,
                        f"Stance Ratio: {record['normalized_stance_width_frame']:.2f}",
                        (25, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.62,
                        (96, 165, 250),
                        2,
                        cv2.LINE_AA,
                    )
                    y += 28

                if phase_name == "impact":
                    cv2.putText(
                        frame,
                        "ESTIMATED IMPACT WINDOW (POSE PROXY)",
                        (25, y + 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.64,
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
# MAIN V3 ENGINE
# =============================================================================


def process_video(
    path,
    max_frames=MAX_ANALYSIS_FRAMES,
    batting_hand=None,
    player_height_cm=None,
    player_weight_kg=None,
    age_group=None,
):
    """
    Run NeuroBat Biomechanics Engine V3 on one uploaded video.

    Parameters are backward compatible with the current V2.1 route. Player
    anthropometric fields are accepted for architecture continuity but V3 does
    not apply invented age/height/weight "ideal technique" corrections.
    Distance-like movement signals are body-normalized instead.
    """

    try:
        import cv2
        import mediapipe as mp
        import numpy as np
    except Exception as exc:
        raise RuntimeError(
            "Unable to load NeuroBat video-analysis dependencies: "
            f"{exc}"
        ) from exc

    batting_hand = normalize_batting_hand(batting_hand)

    # -------------------------------------------------------------------------
    # MediaPipe setup
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Storage
    # -------------------------------------------------------------------------

    frame_records = []
    frames_processed = 0
    poses_detected = 0
    valid_biomechanics_frames = 0

    raw_stance_widths = []
    shoulder_diffs = []
    head_x_positions = []

    left_knee_angles = []
    right_knee_angles = []
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

    # -------------------------------------------------------------------------
    # PASS 1: pose extraction + biomechanics features
    # -------------------------------------------------------------------------

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
                "phase_valid": False,
                "pose_landmarks": None,
                "average_visibility": 0.0,
                "wrist_rel": None,
                "nose_rel": None,
                "left_ankle_rel": None,
                "right_ankle_rel": None,
                "left_knee_angle": None,
                "right_knee_angle": None,
                "average_knee_angle": None,
                "left_knee_angle_3d": None,
                "right_knee_angle_3d": None,
                "shoulder_tilt": None,
                "hip_tilt": None,
                "shoulder_axial_orientation_3d": None,
                "hip_axial_orientation_3d": None,
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
            core_visibility_values = [float(lm.visibility) for lm in core_landmarks]

            average_visibility = sum(visibility_values) / len(visibility_values)
            minimum_core_visibility = min(core_visibility_values)

            record["average_visibility"] = average_visibility
            landmark_visibility_values.append(average_visibility)

            # Pixel coordinates. Using true pixels prevents width/height aspect
            # ratio distortion when calculating 2D angles.
            nose_px = landmark_to_pixel(nose, width, height)
            left_shoulder_px = landmark_to_pixel(left_shoulder, width, height)
            right_shoulder_px = landmark_to_pixel(right_shoulder, width, height)
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

            shoulder_width_px = euclidean_distance(left_shoulder_px, right_shoulder_px)
            hip_width_px = euclidean_distance(left_hip_px, right_hip_px)
            torso_length_px = euclidean_distance(shoulder_center_px, hip_center_px)

            # Robust body scale for movement features. Torso length remains usable
            # when the shoulders appear narrow because the batter is side-on.
            body_scale_px = max(
                torso_length_px,
                0.5 * (shoulder_width_px + hip_width_px),
                5.0,
            )

            # Body-relative points are less sensitive to player height, camera
            # distance and global translation than raw pixel displacement.
            record["nose_rel"] = relative_point(nose_px, hip_center_px, body_scale_px)
            record["left_ankle_rel"] = relative_point(left_ankle_px, hip_center_px, body_scale_px)
            record["right_ankle_rel"] = relative_point(right_ankle_px, hip_center_px, body_scale_px)

            nose_relative_positions.append(record["nose_rel"])
            left_ankle_relative_positions.append(record["left_ankle_rel"])
            right_ankle_relative_positions.append(record["right_ankle_rel"])

            # Grip/wrist point for phase detection. If both wrists are visible,
            # use midpoint. If one is strongly visible and the other is occluded,
            # use the stronger wrist because both hands normally move together on
            # a cricket bat handle.
            left_wrist_visibility = float(left_wrist.visibility)
            right_wrist_visibility = float(right_wrist.visibility)
            wrist_px = None

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
                record["wrist_rel"] = relative_point(wrist_px, hip_center_px, body_scale_px)
                record["phase_valid"] = record["wrist_rel"] is not None

            # Core biomechanics quality gate.
            if minimum_core_visibility >= MIN_LANDMARK_VISIBILITY:
                valid_biomechanics_frames += 1
                record["biomechanics_valid"] = True

                left_knee_angle = calculate_angle(
                    left_hip_px,
                    left_knee_px,
                    left_ankle_px,
                )
                right_knee_angle = calculate_angle(
                    right_hip_px,
                    right_knee_px,
                    right_ankle_px,
                )
                average_knee = (left_knee_angle + right_knee_angle) / 2.0

                record["left_knee_angle"] = left_knee_angle
                record["right_knee_angle"] = right_knee_angle
                record["average_knee_angle"] = average_knee

                left_knee_angles.append(left_knee_angle)
                right_knee_angles.append(right_knee_angle)

                shoulder_tilt = calculate_line_angle(left_shoulder_px, right_shoulder_px)
                hip_tilt = calculate_line_angle(left_hip_px, right_hip_px)

                record["shoulder_tilt"] = shoulder_tilt
                record["hip_tilt"] = hip_tilt
                shoulder_tilts.append(shoulder_tilt)
                hip_tilts.append(hip_tilt)

                raw_stance_width = abs(float(left_ankle.x) - float(right_ankle.x))
                shoulder_difference = abs(float(left_shoulder.y) - float(right_shoulder.y))
                ankle_separation_px = abs(left_ankle_px[0] - right_ankle_px[0])

                raw_stance_widths.append(raw_stance_width)
                shoulder_diffs.append(shoulder_difference)
                head_x_positions.append(float(nose.x))

                if shoulder_width_px > 5:
                    shoulder_widths_pixels.append(shoulder_width_px)
                    frame_stance_ratio = ankle_separation_px / shoulder_width_px
                    record["normalized_stance_width_frame"] = frame_stance_ratio
                    normalized_stance_frame_values.append(frame_stance_ratio)

                if ankle_separation_px > 1:
                    ankle_separations_pixels.append(ankle_separation_px)

                body_scaled_stance = ankle_separation_px / body_scale_px
                record["body_scaled_stance_width"] = body_scaled_stance
                body_scaled_stance_values.append(body_scaled_stance)

                nose_positions_pixels.append(nose_px)
                left_ankle_positions_pixels.append(left_ankle_px)
                right_ankle_positions_pixels.append(right_ankle_px)

                # Optional MediaPipe world landmarks: use them as additional 3D
                # monocular pose proxies, never as laboratory ground truth.
                if results.pose_world_landmarks:
                    world = results.pose_world_landmarks.landmark

                    l_shoulder_w = landmark_to_world_tuple(world[mp_pose.PoseLandmark.LEFT_SHOULDER])
                    r_shoulder_w = landmark_to_world_tuple(world[mp_pose.PoseLandmark.RIGHT_SHOULDER])
                    l_hip_w = landmark_to_world_tuple(world[mp_pose.PoseLandmark.LEFT_HIP])
                    r_hip_w = landmark_to_world_tuple(world[mp_pose.PoseLandmark.RIGHT_HIP])
                    l_knee_w = landmark_to_world_tuple(world[mp_pose.PoseLandmark.LEFT_KNEE])
                    r_knee_w = landmark_to_world_tuple(world[mp_pose.PoseLandmark.RIGHT_KNEE])
                    l_ankle_w = landmark_to_world_tuple(world[mp_pose.PoseLandmark.LEFT_ANKLE])
                    r_ankle_w = landmark_to_world_tuple(world[mp_pose.PoseLandmark.RIGHT_ANKLE])

                    left_knee_3d = calculate_angle_3d(l_hip_w, l_knee_w, l_ankle_w)
                    right_knee_3d = calculate_angle_3d(r_hip_w, r_knee_w, r_ankle_w)
                    shoulder_axial = calculate_axial_orientation_proxy(l_shoulder_w, r_shoulder_w)
                    hip_axial = calculate_axial_orientation_proxy(l_hip_w, r_hip_w)

                    record["left_knee_angle_3d"] = left_knee_3d
                    record["right_knee_angle_3d"] = right_knee_3d
                    record["shoulder_axial_orientation_3d"] = shoulder_axial
                    record["hip_axial_orientation_3d"] = hip_axial

                    if left_knee_3d is not None:
                        left_knee_angles_3d.append(left_knee_3d)
                    if right_knee_3d is not None:
                        right_knee_angles_3d.append(right_knee_3d)
                    if shoulder_axial is not None:
                        shoulder_axial_orientations_3d.append(shoulder_axial)
                    if hip_axial is not None:
                        hip_axial_orientations_3d.append(hip_axial)

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

    if valid_biomechanics_frames == 0:
        return {
            "error": (
                "Pose landmarks were detected, but landmark visibility was too "
                "low for reliable biomechanics analysis."
            )
        }

    # -------------------------------------------------------------------------
    # V3 phase detection
    # -------------------------------------------------------------------------

    phase_detection_internal = detect_batting_phases(
        frame_records,
        fps,
        np,
    )

    phase_metrics = build_phase_metrics(
        frame_records,
        phase_detection_internal,
        batting_hand,
        np,
    )

    # -------------------------------------------------------------------------
    # Backward-compatible whole-clip summary (V2.1 keys)
    # -------------------------------------------------------------------------

    avg_left_knee_angle = round(safe_mean(left_knee_angles, np), 1)
    avg_right_knee_angle = round(safe_mean(right_knee_angles, np), 1)
    average_knee_angle = round((avg_left_knee_angle + avg_right_knee_angle) / 2.0, 1)

    avg_stance_width = round(safe_mean(raw_stance_widths, np), 3)

    median_shoulder_width = safe_median(shoulder_widths_pixels, np)
    median_ankle_separation = safe_median(ankle_separations_pixels, np)

    if median_shoulder_width > 0:
        avg_normalized_stance_width = round(
            median_ankle_separation / median_shoulder_width,
            3,
        )
    else:
        avg_normalized_stance_width = 0.0

    avg_shoulder_diff = round(safe_mean(shoulder_diffs, np), 3)
    head_stability = round(safe_std(head_x_positions, np), 3)
    avg_shoulder_tilt = round(safe_mean(shoulder_tilts, np), 2)
    avg_hip_tilt = round(safe_mean(hip_tilts, np), 2)
    average_visibility = round(safe_mean(landmark_visibility_values, np), 3)

    # Legacy pixel-baseline movement metrics preserved for current DB/history.
    mean_shoulder_width = safe_mean(shoulder_widths_pixels, np)

    head_displacement = 0.0
    left_foot_movement = 0.0
    right_foot_movement = 0.0

    if mean_shoulder_width > 1 and nose_positions_pixels:
        baseline_count = min(5, len(nose_positions_pixels))
        baseline_head = (
            safe_mean([p[0] for p in nose_positions_pixels[:baseline_count]], np),
            safe_mean([p[1] for p in nose_positions_pixels[:baseline_count]], np),
        )
        distances = [
            euclidean_distance(baseline_head, point) / mean_shoulder_width
            for point in nose_positions_pixels
        ]
        if distances:
            head_displacement = round(max(distances), 3)

    if mean_shoulder_width > 1 and left_ankle_positions_pixels:
        baseline_count = min(5, len(left_ankle_positions_pixels))
        baseline = (
            safe_mean([p[0] for p in left_ankle_positions_pixels[:baseline_count]], np),
            safe_mean([p[1] for p in left_ankle_positions_pixels[:baseline_count]], np),
        )
        distances = [
            euclidean_distance(baseline, point) / mean_shoulder_width
            for point in left_ankle_positions_pixels
        ]
        if distances:
            left_foot_movement = round(max(distances), 3)

    if mean_shoulder_width > 1 and right_ankle_positions_pixels:
        baseline_count = min(5, len(right_ankle_positions_pixels))
        baseline = (
            safe_mean([p[0] for p in right_ankle_positions_pixels[:baseline_count]], np),
            safe_mean([p[1] for p in right_ankle_positions_pixels[:baseline_count]], np),
        )
        distances = [
            euclidean_distance(baseline, point) / mean_shoulder_width
            for point in right_ankle_positions_pixels
        ]
        if distances:
            right_foot_movement = round(max(distances), 3)

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
    valid_frame_rate = round((valid_biomechanics_frames / frames_processed) * 100.0, 1)

    score, rating = legacy_prototype_score(
        avg_stance_width,
        avg_shoulder_diff,
        head_stability,
        average_knee_angle,
    )
    stance_category = legacy_stance_category(avg_stance_width)

    # -------------------------------------------------------------------------
    # New V3 body-normalized / 3D proxy summary
    # -------------------------------------------------------------------------

    v3_metrics = {
        "head_displacement_body_relative": safe_round(
            displacement_from_baseline(nose_relative_positions, np),
            3,
        ),
        "left_foot_movement_body_relative": safe_round(
            displacement_from_baseline(left_ankle_relative_positions, np),
            3,
        ),
        "right_foot_movement_body_relative": safe_round(
            displacement_from_baseline(right_ankle_relative_positions, np),
            3,
        ),
        "median_frame_stance_width_ratio": safe_round(
            safe_median(normalized_stance_frame_values, np),
            3,
        ),
        "median_body_scaled_stance_width": safe_round(
            safe_median(body_scaled_stance_values, np),
            3,
        ),
        "left_knee_angle_3d_proxy_degrees": safe_round(
            safe_mean(left_knee_angles_3d, np),
            1,
        ) if left_knee_angles_3d else None,
        "right_knee_angle_3d_proxy_degrees": safe_round(
            safe_mean(right_knee_angles_3d, np),
            1,
        ) if right_knee_angles_3d else None,
        "shoulder_axial_orientation_3d_proxy_degrees": safe_round(
            circular_mean_degrees(shoulder_axial_orientations_3d, np),
            2,
        ),
        "hip_axial_orientation_3d_proxy_degrees": safe_round(
            circular_mean_degrees(hip_axial_orientations_3d, np),
            2,
        ),
    }

    # -------------------------------------------------------------------------
    # Annotated video after phase boundaries are known
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
    # Final V3 response
    # -------------------------------------------------------------------------

    return {
        # Current DB/dashboard-compatible fields -------------------------------
        "frames_processed": frames_processed,
        "poses_detected": poses_detected,
        "pose_detection_rate": pose_detection_rate,
        "valid_biomechanics_frames": valid_biomechanics_frames,
        "valid_biomechanics_frame_rate": valid_frame_rate,
        "average_landmark_visibility": average_visibility,

        "left_knee_angle": avg_left_knee_angle,
        "right_knee_angle": avg_right_knee_angle,
        "average_knee_angle": average_knee_angle,

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

        "stance_rating": rating,
        "prototype_score": score,
        "prototype_score_percent": int(score / 4 * 100),

        "processed_video_filename": output_filename,
        "processed_video_url": "/outputs/" + output_filename,
        "processed_video_codec": output_codec,

        "analysis_engine_version": ENGINE_VERSION,

        # V3 temporal analysis -------------------------------------------------
        "batting_phase_detection": public_phase_detection(phase_detection_internal),
        "phase_metrics": phase_metrics,
        "v3_metrics": v3_metrics,

        # Analysis provenance / scientific guardrails -------------------------
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
                "V3 uses body-normalized movement features instead of inventing "
                "age/height/weight-specific ideal technique thresholds."
            ),
        },
        "prototype_score_model": "legacy_v2_rule_based_compatibility",
        "prototype_score_disclaimer": (
            "The current top-level score is retained for V2 dashboard/history "
            "compatibility. Its thresholds are prototype rules and require "
            "expert/empirical validation before being treated as coaching standards."
        ),
        "impact_timing_disclaimer": (
            "V3 impact timing is a pose-based wrist-kinematics proxy. Verified "
            "bat-ball contact requires bat/ball tracking or synchronized ground truth."
        ),
    }
