"""
NeuroBat metric-specific quality regression test.

This test was originally created for V3.1.
It is now used as a regression test for V3.x engines.

Run from the backend folder:

    python -m services.test_v3_1_metric_quality
"""

import numpy as np

from services.biomechanics import (
    ENGINE_VERSION,
    _phase_metric_summary,
)


def main():
    records = []

    # Simulate 10 frames where the right side is occluded.
    #
    # The engine must keep usable measurements from the visible side
    # instead of rejecting the entire biomechanics frame.
    for index in range(10):

        records.append({
            "biomechanics_valid": False,

            "left_knee_angle":
                140.0 + index * 0.1,

            "right_knee_angle":
                None,

            "average_knee_angle":
                None,

            "left_knee_angle_3d":
                138.0 + index * 0.1,

            "right_knee_angle_3d":
                None,

            "normalized_stance_width_frame":
                None,

            "body_scaled_stance_width":
                None,

            "shoulder_tilt":
                5.0 + index * 0.1,

            "hip_tilt":
                3.0 + index * 0.1,

            "shoulder_axial_orientation_3d":
                20.0 + index,

            "hip_axial_orientation_3d":
                10.0 + index,

            "average_visibility":
                0.75,

            "nose_rel":
                (0.01 * index, 0.0),

            "left_ankle_rel":
                (0.02 * index, 0.0),

            "right_ankle_rel":
                None,

            # V3.2 additional metrics.
            # Keep these unavailable in this synthetic occlusion test.
            "left_elbow_angle":
                None,

            "right_elbow_angle":
                None,

            "trunk_lean_degrees":
                None,

            "shoulder_hip_separation_3d_proxy_degrees":
                None,

            "body_center_rel":
                None,
        })

    wrist_speed = np.linspace(
        0.1,
        1.0,
        len(records),
    )

    result = _phase_metric_summary(
        records,
        0,
        len(records) - 1,
        wrist_speed,
        "Right",
        np,
    )

    # ---------------------------------------------------------
    # ENGINE VERSION
    # ---------------------------------------------------------

    assert ENGINE_VERSION.startswith(
        "NeuroBat-Biomechanics-V3."
    )

    # ---------------------------------------------------------
    # METRIC-SPECIFIC QUALITY
    # ---------------------------------------------------------

    assert result[
        "valid_biomechanics_frames"
    ] == 10

    assert result[
        "complete_core_biomechanics_frames"
    ] == 0

    # ---------------------------------------------------------
    # AVAILABLE METRICS
    # ---------------------------------------------------------

    assert result[
        "left_knee_angle_2d_degrees"
    ] is not None

    assert result[
        "shoulder_tilt_degrees"
    ] is not None

    assert result[
        "hip_tilt_degrees"
    ] is not None

    # ---------------------------------------------------------
    # OCCLUDED METRICS
    # ---------------------------------------------------------

    assert result[
        "right_knee_angle_2d_degrees"
    ] is None

    # ---------------------------------------------------------
    # COVERAGE
    # ---------------------------------------------------------

    assert result[
        "metric_coverage"
    ][
        "left_knee"
    ][
        "coverage_percent"
    ] == 100.0

    assert result[
        "metric_coverage"
    ][
        "right_knee"
    ][
        "coverage_percent"
    ] == 0.0

    # ---------------------------------------------------------
    # OUTPUT
    # ---------------------------------------------------------

    print(
        "NeuroBat metric-specific quality regression test: PASS"
    )

    print(
        "Engine:",
        ENGINE_VERSION,
    )

    print(
        "Phase quality:",
        result[
            "phase_biomechanics_quality"
        ],
    )

    print(
        "Mean metric coverage:",
        result[
            "mean_metric_coverage_percent"
        ],
    )


if __name__ == "__main__":
    main()