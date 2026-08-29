r"""
NeuroBat Batch Dataset Processor
================================

Processes the videos listed in NeuroBat_Master_Metadata.csv through the
current NeuroBat biomechanics engine.

Expected project structure
--------------------------
D:\ICBT 2026\Final Project\
    Backend\
        backend\
            batch_process_dataset.py
            services\
                biomechanics.py
    Cut\
    Drive\
    Pull\
    Accepted\
    Rejected\
    Processed\
    Feature_Dataset\
    NeuroBat_Master_Metadata.csv

Run from:
    D:\ICBT 2026\Final Project\Backend\backend

First verify files only:
    python batch_process_dataset.py --check-only

Then process:
    python batch_process_dataset.py

Scientific / research note
--------------------------
PASS / REVIEW / FAIL in this script refers to VIDEO / ANALYSIS QUALITY only.
It does not classify batting technique as good or bad and does not treat a
mis-hit as a rejected sample.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

from services.biomechanics import (
    ENGINE_VERSION,
    OUTPUTS_DIR,
    process_video,
)


# =============================================================================
# DEFAULT PATHS
# =============================================================================

BACKEND_DIR = Path(__file__).resolve().parent

# batch_process_dataset.py is expected at:
# Final Project\Backend\backend\batch_process_dataset.py
DEFAULT_PROJECT_ROOT = BACKEND_DIR.parent.parent

DEFAULT_METADATA = "NeuroBat_Master_Metadata.csv"

SHOT_FOLDERS = {
    "cut": "Cut",
    "drive": "Drive",
    "pull": "Pull",
}

# 900 frames = ~15 seconds at 59.94 fps.
# This avoids the 360-frame / ~6-second limitation for the new 60-fps clips.
DEFAULT_MAX_FRAMES = 900


# =============================================================================
# QC RULES
# =============================================================================

# Engineering QC thresholds for deciding whether a clip needs manual review.
# These are NOT cricket coaching thresholds and are NOT expert technique norms.
PASS_MIN_POSE_DETECTION_RATE = 90.0
PASS_MIN_METRIC_USABLE_RATE = 70.0
PASS_MIN_PHASE_CONFIDENCE = 0.65


# =============================================================================
# GENERAL HELPERS
# =============================================================================


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def ensure_directories(project_root: Path) -> dict[str, Path]:
    paths = {
        "accepted": project_root / "Accepted",
        "rejected": project_root / "Rejected",
        "processed": project_root / "Processed",
        "feature_dataset": project_root / "Feature_Dataset",
        "analysis_json": project_root / "Feature_Dataset" / "Analysis_JSON",
        "frame_data": project_root / "Feature_Dataset" / "Frame_Data",
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths


def load_metadata(metadata_path: Path) -> list[dict[str, str]]:
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise RuntimeError("Metadata CSV has no header row.")

        required = {
            "video_id",
            "original_filename",
            "player_id",
            "shot_type",
            "batting_hand",
        }

        missing = required - set(reader.fieldnames)
        if missing:
            raise RuntimeError(
                "Metadata CSV is missing required columns: "
                + ", ".join(sorted(missing))
            )

        rows = []
        for row_number, row in enumerate(reader, start=2):
            cleaned = {
                key: clean_text(value)
                for key, value in row.items()
                if key is not None
            }
            cleaned["_csv_row"] = str(row_number)

            if not cleaned.get("video_id") and not cleaned.get("original_filename"):
                continue

            rows.append(cleaned)

    return rows


def validate_unique_rows(rows: list[dict[str, str]]) -> list[str]:
    problems = []

    seen_ids: dict[str, int] = {}
    seen_files: dict[str, int] = {}

    for row in rows:
        row_number = row.get("_csv_row", "?")
        video_id = row.get("video_id", "").lower()
        filename = row.get("original_filename", "").lower()

        if video_id:
            if video_id in seen_ids:
                problems.append(
                    f"Duplicate video_id '{row['video_id']}' "
                    f"(rows {seen_ids[video_id]} and {row_number})"
                )
            else:
                seen_ids[video_id] = int(row_number)

        if filename:
            if filename in seen_files:
                problems.append(
                    f"Duplicate original_filename '{row['original_filename']}' "
                    f"(rows {seen_files[filename]} and {row_number})"
                )
            else:
                seen_files[filename] = int(row_number)

    return problems


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".m4v",
}


def find_video(project_root: Path, row: dict[str, str]) -> Path | None:
    """
    Find the classified video listed in metadata.

    Matching order:
    1. Exact filename, e.g. DSC_0019.MP4
    2. Case-insensitive exact filename
    3. Same filename stem regardless of video extension,
       e.g. metadata says DSC_0019.MP4 but disk contains DSC_0019.MOV

    This keeps the batch robust when Windows Explorer hid the real extension
    while the metadata sheet was being created.
    """
    shot_type = row.get("shot_type", "").strip().lower()
    folder_name = SHOT_FOLDERS.get(shot_type)

    if not folder_name:
        return None

    folder = project_root / folder_name
    filename = row.get("original_filename", "").strip()

    if not filename or not folder.exists():
        return None

    direct = folder / filename
    if direct.exists() and direct.is_file():
        return direct

    wanted_name = filename.lower()
    wanted_stem = Path(filename).stem.lower()

    same_stem_matches = []

    for candidate in folder.iterdir():
        if not candidate.is_file():
            continue

        if candidate.name.lower() == wanted_name:
            return candidate

        if (
            candidate.stem.lower() == wanted_stem
            and candidate.suffix.lower() in VIDEO_EXTENSIONS
        ):
            same_stem_matches.append(candidate)

    # Only auto-resolve the extension when there is a unique same-stem match.
    if len(same_stem_matches) == 1:
        return same_stem_matches[0]

    return None


def copy_with_video_id(source: Path, destination_dir: Path, video_id: str) -> Path:
    destination = destination_dir / f"{video_id}{source.suffix.lower()}"
    shutil.copy2(source, destination)
    return destination


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames: list[str] = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# QC
# =============================================================================


def determine_qc_status(result: dict[str, Any]) -> tuple[str, str]:
    """
    Determine engineering quality status.

    PASS:
        Analysis is technically strong enough for the accepted pilot subset.

    REVIEW:
        Analysis completed, but one or more quality conditions need manual
        inspection (Frame Inspector / processed video).

    FAIL:
        The engine returned an error or no usable result.

    This status does NOT represent batting quality.
    """
    if "error" in result:
        return "FAIL", clean_text(result.get("error")) or "Biomechanics engine error"

    reasons = []

    pose_rate = safe_float(result.get("pose_detection_rate"))
    metric_rate = safe_float(result.get("metric_usable_frame_rate"))

    phase = result.get("batting_phase_detection") or {}
    phase_status = clean_text(phase.get("status"))
    phase_confidence = safe_float(phase.get("confidence"))

    scope = result.get("analysis_scope") or {}
    truncated = bool(scope.get("analysis_truncated"))

    if pose_rate is None or pose_rate < PASS_MIN_POSE_DETECTION_RATE:
        reasons.append(
            f"pose_detection_rate={pose_rate if pose_rate is not None else 'missing'}"
        )

    if metric_rate is None or metric_rate < PASS_MIN_METRIC_USABLE_RATE:
        reasons.append(
            f"metric_usable_frame_rate="
            f"{metric_rate if metric_rate is not None else 'missing'}"
        )

    if phase_status.lower() != "detected":
        reasons.append(f"phase_status={phase_status or 'missing'}")

    if (
        phase_confidence is None
        or phase_confidence < PASS_MIN_PHASE_CONFIDENCE
    ):
        reasons.append(
            f"phase_confidence="
            f"{phase_confidence if phase_confidence is not None else 'missing'}"
        )

    if truncated:
        reasons.append("analysis_truncated=True")

    if reasons:
        return "REVIEW", "; ".join(reasons)

    return "PASS", "Technical analysis quality passed batch QC"


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================


PHASES = (
    "setup",
    "backlift",
    "downswing",
    "impact",
    "follow_through",
)

PHASE_FEATURE_KEYS = (
    "phase_biomechanics_quality",
    "mean_metric_coverage_percent",
    "left_knee_angle_2d_degrees",
    "right_knee_angle_2d_degrees",
    "average_knee_angle_2d_degrees",
    "left_knee_angle_3d_proxy_degrees",
    "right_knee_angle_3d_proxy_degrees",
    "left_elbow_angle_2d_degrees",
    "right_elbow_angle_2d_degrees",
    "normalized_stance_width",
    "body_scaled_stance_width",
    "shoulder_tilt_degrees",
    "hip_tilt_degrees",
    "trunk_lean_2d_degrees",
    "shoulder_axial_orientation_3d_proxy_degrees",
    "hip_axial_orientation_3d_proxy_degrees",
    "shoulder_hip_separation_3d_proxy_degrees",
    "average_landmark_visibility",
    "head_movement_body_lengths",
    "left_foot_movement_body_lengths",
    "right_foot_movement_body_lengths",
    "body_center_movement_body_lengths",
    "mean_wrist_speed_body_lengths_per_second",
    "peak_wrist_speed_body_lengths_per_second",
)


def build_feature_row(
    metadata: dict[str, str],
    result: dict[str, Any],
    qc_status: str,
    qc_reason: str,
) -> dict[str, Any]:
    phase = result.get("batting_phase_detection") or {}
    scope = result.get("analysis_scope") or {}
    v3 = result.get("v3_metrics") or {}
    phase_metrics = result.get("phase_metrics") or {}

    row: dict[str, Any] = {
        # Identity / metadata
        "video_id": metadata.get("video_id"),
        "original_filename": metadata.get("original_filename"),
        "player_id": metadata.get("player_id"),
        "shot_type": metadata.get("shot_type"),
        "batting_hand": metadata.get("batting_hand"),
        "camera_view": metadata.get("camera_view"),
        "source_fps_metadata": metadata.get("fps"),
        "expert_label": metadata.get("expert_label"),
        "metadata_accepted": metadata.get("accepted"),

        # Batch QC
        "qc_status": qc_status,
        "qc_reason": qc_reason,

        # Engine / scope
        "analysis_engine_version": result.get("analysis_engine_version"),
        "frames_processed": result.get("frames_processed"),
        "source_frame_count": scope.get("source_frame_count"),
        "analysis_fps": scope.get("fps"),
        "analysis_truncated": scope.get("analysis_truncated"),

        # Pose / metric quality
        "pose_detection_rate": result.get("pose_detection_rate"),
        "metric_usable_frame_rate": result.get("metric_usable_frame_rate"),
        "valid_biomechanics_frame_rate": result.get(
            "valid_biomechanics_frame_rate"
        ),
        "average_landmark_visibility": result.get(
            "average_landmark_visibility"
        ),

        # Phase detection
        "phase_status": phase.get("status"),
        "phase_confidence": phase.get("confidence"),
        "phase_confidence_label": phase.get("confidence_label"),
        "impact_frame": phase.get("impact_frame"),
        "impact_time_seconds": phase.get("impact_time_seconds"),
        "impact_proxy_wrist_speed": phase.get("impact_proxy_wrist_speed"),
        "phase_quality_flags": "|".join(
            clean_text(item) for item in (phase.get("quality_flags") or [])
        ),

        # Whole-clip 2D / proxy biomechanics
        "left_knee_angle": result.get("left_knee_angle"),
        "right_knee_angle": result.get("right_knee_angle"),
        "average_knee_angle": result.get("average_knee_angle"),
        "left_elbow_angle": result.get("left_elbow_angle"),
        "right_elbow_angle": result.get("right_elbow_angle"),
        "trunk_lean_degrees": result.get("trunk_lean_degrees"),
        "shoulder_hip_separation_proxy_degrees": result.get(
            "shoulder_hip_separation_proxy_degrees"
        ),
        "normalized_stance_width": result.get("normalized_stance_width"),
        "shoulder_tilt_degrees": result.get("shoulder_tilt_degrees"),
        "hip_tilt_degrees": result.get("hip_tilt_degrees"),
        "head_displacement": result.get("head_displacement"),
        "left_foot_movement": result.get("left_foot_movement"),
        "right_foot_movement": result.get("right_foot_movement"),
        "stride_displacement_body_lengths": result.get(
            "stride_displacement_body_lengths"
        ),
        "balance_sway_body_lengths": result.get(
            "balance_sway_body_lengths"
        ),
        "balance_stability_proxy": result.get("balance_stability_proxy"),
        "estimated_weight_transfer_proxy_body_lengths": result.get(
            "estimated_weight_transfer_proxy_body_lengths"
        ),

        # Selected V3.2 features
        "head_displacement_body_relative": v3.get(
            "head_displacement_body_relative"
        ),
        "left_foot_movement_body_relative": v3.get(
            "left_foot_movement_body_relative"
        ),
        "right_foot_movement_body_relative": v3.get(
            "right_foot_movement_body_relative"
        ),
        "body_center_movement_body_relative": v3.get(
            "body_center_movement_body_relative"
        ),
        "median_frame_stance_width_ratio": v3.get(
            "median_frame_stance_width_ratio"
        ),
        "median_body_scaled_stance_width": v3.get(
            "median_body_scaled_stance_width"
        ),
        "left_knee_angle_3d_proxy_degrees": v3.get(
            "left_knee_angle_3d_proxy_degrees"
        ),
        "right_knee_angle_3d_proxy_degrees": v3.get(
            "right_knee_angle_3d_proxy_degrees"
        ),
        "shoulder_axial_orientation_3d_proxy_degrees": v3.get(
            "shoulder_axial_orientation_3d_proxy_degrees"
        ),
        "hip_axial_orientation_3d_proxy_degrees": v3.get(
            "hip_axial_orientation_3d_proxy_degrees"
        ),
        "shoulder_hip_separation_3d_proxy_degrees": v3.get(
            "shoulder_hip_separation_3d_proxy_degrees"
        ),
        "body_center_setup_to_impact_displacement_body_lengths": v3.get(
            "body_center_setup_to_impact_displacement_body_lengths"
        ),
        "body_center_forward_transfer_body_lengths": v3.get(
            "body_center_forward_transfer_body_lengths"
        ),
    }

    # Phase-specific features
    for phase_name in PHASES:
        phase_data = phase_metrics.get(phase_name) or {}

        for key in PHASE_FEATURE_KEYS:
            row[f"{phase_name}_{key}"] = phase_data.get(key)

        coverage = phase_data.get("metric_coverage") or {}
        for metric_name, coverage_data in coverage.items():
            if isinstance(coverage_data, dict):
                row[
                    f"{phase_name}_coverage_{metric_name}_percent"
                ] = coverage_data.get("coverage_percent")

    return row


# =============================================================================
# PER-VIDEO PROCESSING
# =============================================================================


def process_one(
    project_root: Path,
    dirs: dict[str, Path],
    metadata: dict[str, str],
    max_frames: int,
    force: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    video_id = metadata.get("video_id", "")
    source = find_video(project_root, metadata)

    report: dict[str, Any] = {
        "video_id": video_id,
        "original_filename": metadata.get("original_filename"),
        "player_id": metadata.get("player_id"),
        "shot_type": metadata.get("shot_type"),
        "batting_hand": metadata.get("batting_hand"),
        "source_path": str(source) if source else "",
        "engine_version_expected": ENGINE_VERSION,
        "qc_status": "",
        "qc_reason": "",
        "analysis_engine_version": "",
        "pose_detection_rate": "",
        "metric_usable_frame_rate": "",
        "phase_status": "",
        "phase_confidence": "",
        "analysis_truncated": "",
        "processed_video": "",
        "analysis_json": "",
        "frame_data_json": "",
    }

    if source is None:
        report["qc_status"] = "FAIL"
        report["qc_reason"] = (
            "Video file not found in classified shot folder "
            f"for shot_type='{metadata.get('shot_type')}'"
        )
        return report, None

    analysis_json_path = dirs["analysis_json"] / f"{video_id}_analysis.json"
    frame_json_path = dirs["frame_data"] / f"{video_id}_frames.json"

    result: dict[str, Any] | None = None
    used_cache = False

    if analysis_json_path.exists() and not force:
        try:
            with analysis_json_path.open("r", encoding="utf-8") as file:
                cached = json.load(file)

            if cached.get("analysis_engine_version") == ENGINE_VERSION:
                result = cached
                used_cache = True
        except Exception:
            result = None

    if result is None:
        result = process_video(
            str(source),
            max_frames=max_frames,
            batting_hand=metadata.get("batting_hand") or None,
        )

    qc_status, qc_reason = determine_qc_status(result)

    # Save frame-level data separately for the future Frame Inspector.
    frame_analysis = result.get("frame_analysis")
    if isinstance(frame_analysis, list):
        write_json(
            frame_json_path,
            {
                "video_id": video_id,
                "original_filename": metadata.get("original_filename"),
                "analysis_engine_version": result.get(
                    "analysis_engine_version"
                ),
                "frame_analysis_meta": result.get("frame_analysis_meta"),
                "frames": frame_analysis,
            },
        )
        report["frame_data_json"] = str(frame_json_path)

    # Store full analysis JSON for traceability.
    write_json(analysis_json_path, result)
    report["analysis_json"] = str(analysis_json_path)

    # Copy the engine-produced annotated video into the project Processed folder.
    processed_filename = result.get("processed_video_filename")
    if processed_filename:
        engine_output = Path(OUTPUTS_DIR) / clean_text(processed_filename)

        if engine_output.exists():
            destination = (
                dirs["processed"]
                / f"{video_id}_processed{engine_output.suffix.lower()}"
            )

            # If this was a cached result and the destination already exists,
            # leave it alone.
            if force or not destination.exists():
                shutil.copy2(engine_output, destination)

            report["processed_video"] = str(destination)
        else:
            # On a resumed run, the project Processed copy may still exist even
            # if the temporary backend output has been cleaned.
            candidates = list(
                dirs["processed"].glob(f"{video_id}_processed.*")
            )
            if candidates:
                report["processed_video"] = str(candidates[0])

    # PASS clips are copied into Accepted.
    # FAIL clips are copied into Rejected.
    # REVIEW clips remain in their classified folder for manual inspection.
    if qc_status == "PASS":
        copy_with_video_id(source, dirs["accepted"], video_id)
    elif qc_status == "FAIL":
        copy_with_video_id(source, dirs["rejected"], video_id)

    phase = result.get("batting_phase_detection") or {}
    scope = result.get("analysis_scope") or {}

    report.update({
        "qc_status": qc_status,
        "qc_reason": qc_reason,
        "analysis_engine_version": result.get("analysis_engine_version", ""),
        "pose_detection_rate": result.get("pose_detection_rate", ""),
        "metric_usable_frame_rate": result.get(
            "metric_usable_frame_rate", ""
        ),
        "phase_status": phase.get("status", ""),
        "phase_confidence": phase.get("confidence", ""),
        "phase_confidence_label": phase.get("confidence_label", ""),
        "phase_quality_flags": "|".join(
            clean_text(item) for item in (phase.get("quality_flags") or [])
        ),
        "analysis_truncated": scope.get("analysis_truncated", ""),
        "cache_used": used_cache,
    })

    if "error" in result:
        return report, None

    feature_row = build_feature_row(
        metadata,
        result,
        qc_status,
        qc_reason,
    )

    return report, feature_row


# =============================================================================
# MAIN
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch-process NeuroBat classified batting videos."
    )

    parser.add_argument(
        "--project-root",
        default=str(DEFAULT_PROJECT_ROOT),
        help=(
            "Final Project folder containing Cut/Drive/Pull and metadata CSV. "
            f"Default: {DEFAULT_PROJECT_ROOT}"
        ),
    )

    parser.add_argument(
        "--metadata",
        default=DEFAULT_METADATA,
        help=f"Metadata CSV filename. Default: {DEFAULT_METADATA}",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help=(
            "Maximum frames analyzed per clip. "
            f"Default: {DEFAULT_MAX_FRAMES}"
        ),
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only verify CSV rows and video file locations; do not analyze.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess even when a current-engine analysis JSON already exists.",
    )

    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    metadata_path = project_root / args.metadata

    print("=" * 72)
    print("NeuroBat Batch Dataset Processor")
    print("=" * 72)
    print(f"Engine:       {ENGINE_VERSION}")
    print(f"Project root: {project_root}")
    print(f"Metadata:     {metadata_path}")
    print(f"Max frames:   {args.max_frames}")
    print("=" * 72)

    if not project_root.exists():
        print(f"ERROR: Project root does not exist: {project_root}")
        return 1

    if not metadata_path.exists():
        print(f"ERROR: Metadata CSV not found: {metadata_path}")
        return 1

    try:
        rows = load_metadata(metadata_path)
    except Exception as exc:
        print(f"ERROR reading metadata: {exc}")
        return 1

    print(f"Metadata rows: {len(rows)}")

    duplicate_problems = validate_unique_rows(rows)
    if duplicate_problems:
        print("\nMetadata problems:")
        for problem in duplicate_problems:
            print(f"  - {problem}")
        print("\nFix these duplicates before processing.")
        return 1

    missing = []

    print("\nVideo lookup:")
    for row in rows:
        source = find_video(project_root, row)

        if source:
            print(
                f"  [FOUND] {row.get('video_id')} "
                f"-> {row.get('shot_type')}\\{source.name}"
            )
        else:
            print(
                f"  [MISSING] {row.get('video_id')} "
                f"-> {row.get('shot_type')}\\"
                f"{row.get('original_filename')}"
            )
            missing.append(row)

    if missing:
        print(
            f"\nERROR: {len(missing)} video(s) listed in the CSV "
            "could not be found."
        )
        print("Fix the filenames/folders before batch analysis.")
        return 1

    if args.check_only:
        print("\nCHECK COMPLETE: all metadata rows map to real video files.")
        print("No videos were analyzed.")
        return 0

    dirs = ensure_directories(project_root)

    reports: list[dict[str, Any]] = []
    all_features: list[dict[str, Any]] = []
    accepted_features: list[dict[str, Any]] = []

    total = len(rows)

    for index, metadata in enumerate(rows, start=1):
        video_id = metadata.get("video_id", f"row_{index}")

        print("\n" + "-" * 72)
        print(
            f"[{index}/{total}] {video_id} | "
            f"{metadata.get('shot_type')} | "
            f"{metadata.get('original_filename')}"
        )
        print("-" * 72)

        try:
            report, feature_row = process_one(
                project_root=project_root,
                dirs=dirs,
                metadata=metadata,
                max_frames=max(1, args.max_frames),
                force=args.force,
            )

        except KeyboardInterrupt:
            print("\nBatch processing interrupted by user.")
            break

        except Exception as exc:
            report = {
                "video_id": video_id,
                "original_filename": metadata.get("original_filename"),
                "player_id": metadata.get("player_id"),
                "shot_type": metadata.get("shot_type"),
                "batting_hand": metadata.get("batting_hand"),
                "qc_status": "FAIL",
                "qc_reason": f"Unhandled processing exception: {exc}",
                "traceback": traceback.format_exc(),
            }
            feature_row = None

        reports.append(report)

        print(f"QC status: {report.get('qc_status')}")
        print(f"Reason:    {report.get('qc_reason')}")

        if feature_row is not None:
            all_features.append(feature_row)

            if report.get("qc_status") == "PASS":
                accepted_features.append(feature_row)

        # Write progress after every video so an interrupted batch retains work.
        write_rows_csv(
            dirs["feature_dataset"] / "batch_processing_report.csv",
            reports,
        )
        write_rows_csv(
            dirs["feature_dataset"] / "all_extracted_features.csv",
            all_features,
        )
        write_rows_csv(
            dirs["feature_dataset"] / "features_dataset.csv",
            accepted_features,
        )

    pass_count = sum(
        1 for row in reports if row.get("qc_status") == "PASS"
    )
    review_count = sum(
        1 for row in reports if row.get("qc_status") == "REVIEW"
    )
    fail_count = sum(
        1 for row in reports if row.get("qc_status") == "FAIL"
    )

    print("\n" + "=" * 72)
    print("BATCH COMPLETE")
    print("=" * 72)
    print(f"Processed rows: {len(reports)}")
    print(f"PASS:           {pass_count}")
    print(f"REVIEW:         {review_count}")
    print(f"FAIL:           {fail_count}")
    print()
    print(
        "Batch report: "
        + str(dirs["feature_dataset"] / "batch_processing_report.csv")
    )
    print(
        "All features: "
        + str(dirs["feature_dataset"] / "all_extracted_features.csv")
    )
    print(
        "Accepted ML features: "
        + str(dirs["feature_dataset"] / "features_dataset.csv")
    )
    print(
        "Analysis JSON: "
        + str(dirs["analysis_json"])
    )
    print(
        "Frame data JSON: "
        + str(dirs["frame_data"])
    )
    print(
        "Processed videos: "
        + str(dirs["processed"])
    )
    print("=" * 72)

    if review_count:
        print(
            "\nIMPORTANT: REVIEW clips are not automatically rejected. "
            "Inspect their processed video / Frame Inspector data before deciding."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())