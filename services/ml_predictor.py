"""
NeuroBat XGBoost inference service.

Purpose
-------
Convert the current NeuroBat V3.2.1 biomechanics result into the exact
10-feature vector used by the academic XGBoost shot-type proof-of-concept,
then return Cut / Drive / Pull probabilities.

Important
---------
- This is a small academic proof-of-concept model.
- It was trained on 11 PASS clips from one player and one camera view.
- It must not be described as a generalisable production classifier.
- Weakness detection still requires real expert-labelled ground truth.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BACKEND_DIR / "models"

MODEL_PATH = MODELS_DIR / "NeuroBat_XGBoost_ShotType_POC.json"
METADATA_PATH = MODELS_DIR / "NeuroBat_XGBoost_ShotType_POC_metadata.json"


# Exact feature order used during XGBoost training.
FEATURE_NAMES = [
    "setup_left_knee_angle_2d_degrees",
    "backlift_left_elbow_angle_2d_degrees",
    "downswing_peak_wrist_speed_body_lengths_per_second",
    "impact_left_knee_angle_2d_degrees",
    "impact_trunk_lean_2d_degrees",
    "impact_shoulder_hip_separation_3d_proxy_degrees",
    "follow_through_left_elbow_angle_2d_degrees",
    "head_displacement_body_relative",
    "stride_displacement_body_lengths",
    "balance_sway_body_lengths",
]


def _read_metadata() -> dict[str, Any]:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"XGBoost metadata file not found: {METADATA_PATH}"
        )

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    classes = metadata.get("classes")
    features = metadata.get("features")

    if not isinstance(classes, list) or not classes:
        raise RuntimeError(
            "Model metadata does not contain a valid classes list."
        )

    if features != FEATURE_NAMES:
        raise RuntimeError(
            "Model metadata feature order does not match ml_predictor.py."
        )

    return metadata


def _phase_value(
    analysis_result: dict[str, Any],
    phase_name: str,
    metric_name: str,
):
    phase_metrics = analysis_result.get("phase_metrics") or {}
    phase = phase_metrics.get(phase_name) or {}
    return phase.get(metric_name)


def extract_model_features(
    analysis_result: dict[str, Any],
) -> dict[str, float | None]:
    """
    Extract the exact ten features used during model training.

    Missing measurements remain None. They are not converted to zero.
    """
    v3 = analysis_result.get("v3_metrics") or {}

    return {
        "setup_left_knee_angle_2d_degrees":
            _phase_value(
                analysis_result,
                "setup",
                "left_knee_angle_2d_degrees",
            ),

        "backlift_left_elbow_angle_2d_degrees":
            _phase_value(
                analysis_result,
                "backlift",
                "left_elbow_angle_2d_degrees",
            ),

        "downswing_peak_wrist_speed_body_lengths_per_second":
            _phase_value(
                analysis_result,
                "downswing",
                "peak_wrist_speed_body_lengths_per_second",
            ),

        "impact_left_knee_angle_2d_degrees":
            _phase_value(
                analysis_result,
                "impact",
                "left_knee_angle_2d_degrees",
            ),

        "impact_trunk_lean_2d_degrees":
            _phase_value(
                analysis_result,
                "impact",
                "trunk_lean_2d_degrees",
            ),

        "impact_shoulder_hip_separation_3d_proxy_degrees":
            _phase_value(
                analysis_result,
                "impact",
                "shoulder_hip_separation_3d_proxy_degrees",
            ),

        "follow_through_left_elbow_angle_2d_degrees":
            _phase_value(
                analysis_result,
                "follow_through",
                "left_elbow_angle_2d_degrees",
            ),

        "head_displacement_body_relative":
            v3.get("head_displacement_body_relative"),

        "stride_displacement_body_lengths":
            analysis_result.get(
                "stride_displacement_body_lengths",
                v3.get("stride_displacement_body_lengths"),
            ),

        "balance_sway_body_lengths":
            analysis_result.get(
                "balance_sway_body_lengths",
                v3.get("balance_sway_body_lengths"),
            ),
    }


def _to_float(value):
    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    # NaN check without requiring numpy at import time.
    if number != number:
        return None

    return number


def predict_shot_type(
    analysis_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Run XGBoost inference from a completed NeuroBat biomechanics result.

    Returns a structured status instead of raising for normal model/data
    availability problems so the main /analyze route can still succeed.
    """
    try:
        import numpy as np
        from xgboost import XGBClassifier
    except Exception as exc:
        return {
            "status": "unavailable",
            "available": False,
            "reason": (
                "XGBoost inference dependencies are not available: "
                f"{exc}"
            ),
            "model_type": "XGBoost",
            "target": "shot_type",
            "scope": "academic_proof_of_concept",
        }

    if not MODEL_PATH.exists():
        return {
            "status": "unavailable",
            "available": False,
            "reason": f"Model file not found: {MODEL_PATH}",
            "model_type": "XGBoost",
            "target": "shot_type",
            "scope": "academic_proof_of_concept",
        }

    try:
        metadata = _read_metadata()
    except Exception as exc:
        return {
            "status": "unavailable",
            "available": False,
            "reason": str(exc),
            "model_type": "XGBoost",
            "target": "shot_type",
            "scope": "academic_proof_of_concept",
        }

    raw_features = extract_model_features(analysis_result)

    feature_values = {}
    missing_features = []

    for name in FEATURE_NAMES:
        value = _to_float(raw_features.get(name))
        feature_values[name] = value

        if value is None:
            missing_features.append(name)

    if missing_features:
        return {
            "status": "insufficient_features",
            "available": False,
            "reason": (
                "The current clip does not contain all features required "
                "by the proof-of-concept XGBoost model."
            ),
            "missing_features": missing_features,
            "feature_count_required": len(FEATURE_NAMES),
            "feature_count_available": (
                len(FEATURE_NAMES) - len(missing_features)
            ),
            "model_type": "XGBoost",
            "target": "shot_type",
            "scope": "academic_proof_of_concept",
        }

    try:
        model = XGBClassifier()
        model.load_model(str(MODEL_PATH))

        vector = np.asarray(
            [[feature_values[name] for name in FEATURE_NAMES]],
            dtype=np.float32,
        )

        probabilities = model.predict_proba(vector)[0]

        classes = metadata["classes"]

        if len(probabilities) != len(classes):
            raise RuntimeError(
                "Prediction probability count does not match metadata classes."
            )

        best_index = int(np.argmax(probabilities))

        probability_map = {
            class_name: round(float(probability), 4)
            for class_name, probability in zip(
                classes,
                probabilities,
            )
        }

        return {
            "status": "predicted",
            "available": True,
            "model_name": metadata.get(
                "model_name",
                "NeuroBat XGBoost Shot-Type Proof-of-Concept",
            ),
            "model_type": "XGBoost",
            "target": metadata.get("target", "shot_type"),
            "predicted_shot_type": classes[best_index],
            "confidence": round(
                float(probabilities[best_index]),
                4,
            ),
            "probabilities": probability_map,
            "feature_count": len(FEATURE_NAMES),
            "features_used": {
                key: round(float(value), 4)
                for key, value in feature_values.items()
            },
            "scope": "academic_proof_of_concept",
            "validation_status": (
                "Pilot model only; not generalisable production validation."
            ),
            "training_sample_count": metadata.get("sample_count"),
            "evaluation": metadata.get("evaluation"),
            "limitations": metadata.get("limitations", []),
        }

    except Exception as exc:
        return {
            "status": "prediction_error",
            "available": False,
            "reason": str(exc),
            "model_type": "XGBoost",
            "target": "shot_type",
            "scope": "academic_proof_of_concept",
        }