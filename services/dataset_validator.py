"""
NeuroBat Dataset Validator
==========================

Validates the ML feature dataset created by services/feature_dataset.py.

Checks:
- Dataset existence and readability
- Total rows / unique source videos
- Duplicate source-video hashes
- Required columns
- Missing player context
- Phase-detection success and confidence
- Metric coverage / missingness
- Missing expert labels
- Basic engineering readiness for later ML work

Important:
The readiness result is an engineering/data-quality gate only.
It is NOT proof that the dataset is statistically sufficient for a valid
machine-learning model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = BACKEND_DIR / "datasets" / "neurobat_features.csv"


REQUIRED_COLUMNS = (
    "source_video_sha256",
    "original_video_name",
    "analysis_engine_version",
    "batting_hand",
    "age_group",
    "phase_detection_status",
    "phase_detection_confidence",
    "phase_detection_confidence_label",
    "metric_usable_frame_rate",
    "average_landmark_visibility",
)


LABEL_COLUMNS = (
    "shot_type_label",
    "expert_technique_label",
    "weakness_label",
    "expert_score",
    "expert_notes",
)


CORE_ML_FEATURES = (
    "setup_average_knee_angle_2d_degrees",
    "backlift_left_elbow_angle_2d_degrees",
    "backlift_right_elbow_angle_2d_degrees",
    "downswing_trunk_lean_2d_degrees",
    "impact_average_knee_angle_2d_degrees",
    "impact_trunk_lean_2d_degrees",
    "impact_shoulder_hip_separation_3d_proxy_degrees",
    "impact_peak_wrist_speed_body_lengths_per_second",
    "follow_through_trunk_lean_2d_degrees",
    "stride_displacement_body_lengths",
    "balance_sway_body_lengths",
    "balance_stability_proxy",
    "estimated_weight_transfer_proxy_body_lengths",
)


def _clean(value):
    if value is None:
        return ""
    return str(value).strip()


def _is_missing(value):
    text = _clean(value).lower()
    return text in {
        "",
        "none",
        "null",
        "nan",
        "na",
        "n/a",
    }


def _to_float(value):
    if _is_missing(value):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return number


def _percentage(count, total):
    if total <= 0:
        return 0.0
    return round((count / total) * 100.0, 1)


def load_dataset(csv_path):
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:

        reader = csv.DictReader(handle)

        fieldnames = reader.fieldnames or []

        rows = [
            dict(row)
            for row in reader
        ]

    return path, fieldnames, rows


def validate_dataset(
    csv_path=DEFAULT_DATASET_PATH,
    min_unique_samples=30,
):
    """
    Return a structured NeuroBat dataset-quality report.

    min_unique_samples is a configurable engineering threshold used only to
    prevent accidental model training on an extremely small dataset.
    It is not a statistical sample-size guarantee.
    """

    path, fieldnames, rows = load_dataset(
        csv_path
    )

    total_rows = len(rows)

    missing_required_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in fieldnames
    ]

    hashes = [
        _clean(
            row.get(
                "source_video_sha256"
            )
        )
        for row in rows
        if not _is_missing(
            row.get(
                "source_video_sha256"
            )
        )
    ]

    hash_counts = Counter(hashes)

    duplicate_hashes = {
        video_hash: count
        for video_hash, count
        in hash_counts.items()
        if count > 1
    }

    unique_videos = len(
        set(hashes)
    )

    rows_without_hash = sum(
        _is_missing(
            row.get(
                "source_video_sha256"
            )
        )
        for row in rows
    )

    # --------------------------------------------------------
    # Player context
    # --------------------------------------------------------

    batting_hand_available = sum(
        not _is_missing(
            row.get(
                "batting_hand"
            )
        )
        for row in rows
    )

    age_group_available = sum(
        not _is_missing(
            row.get(
                "age_group"
            )
        )
        for row in rows
    )

    # --------------------------------------------------------
    # Phase detection
    # --------------------------------------------------------

    phase_detected = sum(
        _clean(
            row.get(
                "phase_detection_status"
            )
        ).lower()
        == "detected"
        for row in rows
    )

    high_confidence = sum(
        _clean(
            row.get(
                "phase_detection_confidence_label"
            )
        ).lower()
        == "high"
        for row in rows
    )

    phase_confidences = [
        value
        for value in (
            _to_float(
                row.get(
                    "phase_detection_confidence"
                )
            )
            for row in rows
        )
        if value is not None
    ]

    average_phase_confidence = (
        round(
            sum(phase_confidences)
            / len(phase_confidences),
            3,
        )
        if phase_confidences
        else None
    )

    # --------------------------------------------------------
    # Pose / metric quality
    # --------------------------------------------------------

    usable_rates = [
        value
        for value in (
            _to_float(
                row.get(
                    "metric_usable_frame_rate"
                )
            )
            for row in rows
        )
        if value is not None
    ]

    average_metric_usable_frame_rate = (
        round(
            sum(usable_rates)
            / len(usable_rates),
            1,
        )
        if usable_rates
        else None
    )

    visibility_values = [
        value
        for value in (
            _to_float(
                row.get(
                    "average_landmark_visibility"
                )
            )
            for row in rows
        )
        if value is not None
    ]

    average_landmark_visibility = (
        round(
            sum(visibility_values)
            / len(visibility_values),
            3,
        )
        if visibility_values
        else None
    )

    # --------------------------------------------------------
    # Label completeness
    # --------------------------------------------------------

    label_missing = {}

    for column in LABEL_COLUMNS:

        if column not in fieldnames:
            label_missing[column] = total_rows
            continue

        label_missing[column] = sum(
            _is_missing(
                row.get(column)
            )
            for row in rows
        )

    # --------------------------------------------------------
    # Feature missingness
    # --------------------------------------------------------

    feature_missingness = {}

    for column in CORE_ML_FEATURES:

        if column not in fieldnames:

            feature_missingness[column] = {
                "missing_count": total_rows,
                "missing_percent": 100.0,
                "column_present": False,
            }

            continue

        missing_count = sum(
            _is_missing(
                row.get(column)
            )
            for row in rows
        )

        feature_missingness[column] = {
            "missing_count":
                missing_count,

            "missing_percent":
                _percentage(
                    missing_count,
                    total_rows,
                ),

            "column_present":
                True,
        }

    high_missing_features = [
        column
        for column, details
        in feature_missingness.items()
        if details[
            "missing_percent"
        ] > 40.0
    ]

    # --------------------------------------------------------
    # Engineering readiness
    # --------------------------------------------------------

    blockers = []
    warnings = []

    if total_rows == 0:
        blockers.append(
            "Dataset contains no samples."
        )

    if missing_required_columns:
        blockers.append(
            "Required dataset columns are missing."
        )

    if duplicate_hashes:
        blockers.append(
            "Duplicate source-video samples are present."
        )

    if unique_videos < int(
        min_unique_samples
    ):
        blockers.append(
            "Insufficient independent source videos for the configured "
            "ML engineering gate."
        )

    if total_rows > 0 and phase_detected < total_rows:
        warnings.append(
            "Some samples do not contain a successfully detected batting stroke."
        )

    if high_missing_features:
        warnings.append(
            "Some core candidate ML features have more than 40% missing data."
        )

    if total_rows > 0 and batting_hand_available < total_rows:
        warnings.append(
            "Batting hand is missing for some samples."
        )

    if total_rows > 0 and label_missing.get(
        "shot_type_label",
        0,
    ) > 0:
        warnings.append(
            "Shot-type labels are incomplete."
        )

    if total_rows > 0 and label_missing.get(
        "weakness_label",
        0,
    ) > 0:
        warnings.append(
            "Expert weakness labels are incomplete."
        )

    training_gate = (
        "PASS"
        if not blockers
        else "NOT READY"
    )

    report = {
        "dataset_path":
            str(path),

        "dataset_summary": {
            "total_rows":
                total_rows,

            "unique_source_videos":
                unique_videos,

            "rows_without_video_hash":
                rows_without_hash,

            "duplicate_video_hashes":
                len(
                    duplicate_hashes
                ),

            "duplicate_rows":
                sum(
                    count - 1
                    for count
                    in duplicate_hashes.values()
                ),
        },

        "player_context": {
            "batting_hand_available":
                batting_hand_available,

            "batting_hand_coverage_percent":
                _percentage(
                    batting_hand_available,
                    total_rows,
                ),

            "age_group_available":
                age_group_available,

            "age_group_coverage_percent":
                _percentage(
                    age_group_available,
                    total_rows,
                ),
        },

        "phase_detection": {
            "detected_samples":
                phase_detected,

            "detected_percent":
                _percentage(
                    phase_detected,
                    total_rows,
                ),

            "high_confidence_samples":
                high_confidence,

            "high_confidence_percent":
                _percentage(
                    high_confidence,
                    total_rows,
                ),

            "average_confidence":
                average_phase_confidence,
        },

        "pose_metric_quality": {
            "average_metric_usable_frame_rate":
                average_metric_usable_frame_rate,

            "average_landmark_visibility":
                average_landmark_visibility,
        },

        "expert_labels": {
            column: {
                "missing":
                    missing_count,

                "missing_percent":
                    _percentage(
                        missing_count,
                        total_rows,
                    ),
            }
            for column, missing_count
            in label_missing.items()
        },

        "core_feature_missingness":
            feature_missingness,

        "high_missing_features":
            high_missing_features,

        "schema": {
            "column_count":
                len(fieldnames),

            "missing_required_columns":
                missing_required_columns,
        },

        "ml_engineering_gate": {
            "status":
                training_gate,

            "configured_min_unique_samples":
                int(
                    min_unique_samples
                ),

            "blockers":
                blockers,

            "warnings":
                warnings,

            "important_note": (
                "Passing this gate only means the dataset satisfies basic "
                "engineering/data-quality checks. It does not establish "
                "statistical sufficiency, biomechanical validity, model "
                "generalizability, or clinical/coaching validity."
            ),
        },
    }

    return report


def print_report(report):
    summary = report[
        "dataset_summary"
    ]

    player = report[
        "player_context"
    ]

    phases = report[
        "phase_detection"
    ]

    quality = report[
        "pose_metric_quality"
    ]

    labels = report[
        "expert_labels"
    ]

    gate = report[
        "ml_engineering_gate"
    ]

    print()
    print(
        "=" * 64
    )
    print(
        "NEUROBAT DATASET QUALITY REPORT"
    )
    print(
        "=" * 64
    )

    print()
    print(
        "DATASET"
    )
    print(
        f"Total rows:                 {summary['total_rows']}"
    )
    print(
        f"Unique source videos:       {summary['unique_source_videos']}"
    )
    print(
        f"Duplicate video hashes:     {summary['duplicate_video_hashes']}"
    )
    print(
        f"Duplicate rows:             {summary['duplicate_rows']}"
    )

    print()
    print(
        "PLAYER CONTEXT"
    )
    print(
        "Batting-hand coverage:      "
        f"{player['batting_hand_coverage_percent']}%"
    )
    print(
        "Age-group coverage:         "
        f"{player['age_group_coverage_percent']}%"
    )

    print()
    print(
        "PHASE DETECTION"
    )
    print(
        "Detected samples:           "
        f"{phases['detected_samples']} "
        f"({phases['detected_percent']}%)"
    )
    print(
        "High-confidence samples:    "
        f"{phases['high_confidence_samples']} "
        f"({phases['high_confidence_percent']}%)"
    )
    print(
        "Average confidence:         "
        f"{phases['average_confidence']}"
    )

    print()
    print(
        "POSE / METRIC QUALITY"
    )
    print(
        "Avg metric-usable rate:     "
        f"{quality['average_metric_usable_frame_rate']}"
    )
    print(
        "Avg landmark visibility:    "
        f"{quality['average_landmark_visibility']}"
    )

    print()
    print(
        "EXPERT LABELS"
    )

    for column, details in labels.items():
        print(
            f"{column:<28}"
            f"{details['missing']} missing "
            f"({details['missing_percent']}%)"
        )

    print()
    print(
        "ML ENGINEERING GATE"
    )
    print(
        f"Status:                     {gate['status']}"
    )
    print(
        "Configured minimum unique "
        f"samples: {gate['configured_min_unique_samples']}"
    )

    if gate[
        "blockers"
    ]:

        print()
        print(
            "Blockers:"
        )

        for item in gate[
            "blockers"
        ]:
            print(
                f"  - {item}"
            )

    if gate[
        "warnings"
    ]:

        print()
        print(
            "Warnings:"
        )

        for item in gate[
            "warnings"
        ]:
            print(
                f"  - {item}"
            )

    print()
    print(
        "NOTE:"
    )
    print(
        gate[
            "important_note"
        ]
    )
    print()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate the NeuroBat ML feature dataset."
        )
    )

    parser.add_argument(
        "--dataset",
        default=str(
            DEFAULT_DATASET_PATH
        ),
        help=(
            "Path to neurobat_features.csv"
        ),
    )

    parser.add_argument(
        "--min-unique-samples",
        type=int,
        default=30,
        help=(
            "Configurable engineering gate only; "
            "not a statistical sample-size guarantee."
        ),
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "Print the full report as JSON."
        ),
    )

    args = parser.parse_args()

    try:

        report = validate_dataset(
            csv_path=args.dataset,
            min_unique_samples=max(
                1,
                args.min_unique_samples,
            ),
        )

    except Exception as error:

        print(
            "[NeuroBat] Dataset validation failed:"
        )
        print(
            str(error)
        )

        raise SystemExit(
            1
        )

    if args.json:

        print(
            json.dumps(
                report,
                indent=2,
            )
        )

    else:

        print_report(
            report
        )


if __name__ == "__main__":
    main()