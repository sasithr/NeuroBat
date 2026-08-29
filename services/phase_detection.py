"""
NeuroBat batting phase detection service.

This module performs pose-only temporal segmentation of a batting stroke using
body-normalized wrist kinematics. It intentionally does NOT claim to observe
true bat-ball contact. The "impact" event is an estimated proxy based on the
first major hand/wrist-speed peak after the backlift turning point.

Designed for NeuroBat Biomechanics Engine V3.
"""

from __future__ import annotations

import math


PHASE_ORDER = (
    "setup",
    "backlift",
    "downswing",
    "impact",
    "follow_through",
)


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def _clip(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))


def _odd_window(value, minimum=3):
    value = max(minimum, int(value))
    if value % 2 == 0:
        value += 1
    return value


def _moving_average(values, window, np):
    """Centered moving average with edge padding."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values

    window = _odd_window(min(window, max(3, values.size)))
    if window > values.size:
        window = values.size if values.size % 2 == 1 else max(1, values.size - 1)
    if window <= 1:
        return values.copy()

    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _fill_signal(values, max_short_gap, np):
    """
    Fill short internal gaps first, then produce a continuous copy for signal
    processing. The original finite-mask is retained elsewhere and is used to
    penalise low-coverage detections.
    """
    arr = np.asarray(values, dtype=float).copy()
    n = arr.size
    if n == 0:
        return arr

    finite = np.isfinite(arr)
    if not finite.any():
        return arr

    # Fill only short interior gaps in the working copy.
    i = 0
    while i < n:
        if finite[i]:
            i += 1
            continue

        gap_start = i
        while i < n and not finite[i]:
            i += 1
        gap_end = i - 1

        left = gap_start - 1
        right = i
        gap_len = gap_end - gap_start + 1

        if (
            gap_len <= max_short_gap
            and left >= 0
            and right < n
            and finite[left]
            and finite[right]
        ):
            arr[gap_start:right] = np.linspace(
                arr[left],
                arr[right],
                gap_len + 2,
            )[1:-1]

    # For derivative calculation, interpolate remaining internal gaps and hold
    # edge values. Reliability is still based on the original observation mask.
    finite2 = np.isfinite(arr)
    indices = np.arange(n, dtype=float)
    valid_idx = indices[finite2]
    valid_values = arr[finite2]

    if valid_idx.size == 1:
        return np.full(n, valid_values[0], dtype=float)

    return np.interp(indices, valid_idx, valid_values)


def _find_quiet_before(speed, peak_index, threshold, quiet_frames):
    """Find the end of the last quiet run before the candidate stroke peak."""
    quiet_frames = max(2, int(quiet_frames))
    for start in range(peak_index - quiet_frames, -1, -1):
        window = speed[start:start + quiet_frames]
        if window.size == quiet_frames and bool((window < threshold).all()):
            return start + quiet_frames
    return None



def _find_motion_onset_before(speed, peak_index, threshold, quiet_frames, active_frames, lookback_frames):
    """
    Find the earliest quiet-to-active transition within a bounded pre-peak
    window. Choosing the earliest transition prevents a brief pause at the top
    of the backlift from being mistaken for the initial setup-to-backlift onset.
    """
    quiet_frames = max(2, int(quiet_frames))
    active_frames = max(2, int(active_frames))
    start = max(quiet_frames, peak_index - max(quiet_frames + active_frames, int(lookback_frames)))
    stop = max(start + 1, peak_index - active_frames + 1)

    for transition in range(start, stop):
        quiet = speed[transition - quiet_frames:transition]
        active = speed[transition:transition + active_frames]
        if quiet.size != quiet_frames or active.size != active_frames:
            continue
        quiet_ok = bool((quiet < threshold).all())
        active_fraction = float((active >= threshold).mean())
        if quiet_ok and active_fraction >= 0.67:
            return transition

    return None

def _find_quiet_after(speed, start_index, threshold, quiet_frames):
    """Find the first quiet run after the candidate impact event."""
    quiet_frames = max(2, int(quiet_frames))
    last_start = len(speed) - quiet_frames
    for start in range(max(0, start_index), last_start + 1):
        window = speed[start:start + quiet_frames]
        if window.size == quiet_frames and bool((window < threshold).all()):
            return start
    return None


def _local_peaks(values, min_height, min_distance, np):
    """Dependency-free local maxima detector."""
    values = np.asarray(values, dtype=float)
    if values.size < 3:
        return []

    candidates = []
    for i in range(1, values.size - 1):
        if (
            values[i] >= min_height
            and values[i] >= values[i - 1]
            and values[i] > values[i + 1]
        ):
            candidates.append(i)

    # Non-maximum suppression by descending peak height.
    selected = []
    for i in sorted(candidates, key=lambda idx: values[idx], reverse=True):
        if all(abs(i - kept) >= min_distance for kept in selected):
            selected.append(i)

    return sorted(selected)


def _confidence_label(confidence):
    if confidence >= 0.80:
        return "High"
    if confidence >= 0.60:
        return "Moderate"
    return "Low"


def _duration_score(seconds, low, ideal_low, ideal_high, high):
    """Broad plausibility score for temporal segmentation, not technique quality."""
    if seconds <= low or seconds >= high:
        return 0.0
    if ideal_low <= seconds <= ideal_high:
        return 1.0
    if seconds < ideal_low:
        return (seconds - low) / max(ideal_low - low, 1e-6)
    return (high - seconds) / max(high - ideal_high, 1e-6)


# ---------------------------------------------------------------------------
# Public detector
# ---------------------------------------------------------------------------


def detect_batting_phases(frame_records, fps, np):
    """
    Detect Setup -> Backlift -> Downswing -> Impact proxy -> Follow-through.

    Expected record fields:
        frame_index: 1-based source frame number
        phase_valid: bool
        wrist_rel: (x, y) body-normalized wrist midpoint relative to hip centre
        average_visibility: float (0..1)

    The detector is deliberately conservative. If wrist tracking is inadequate,
    it returns status="unavailable" instead of inventing phase boundaries.
    """

    n = len(frame_records)
    fps = float(fps or 30.0)
    fps = fps if fps > 0 else 30.0

    base_result = {
        "status": "unavailable",
        "confidence": 0.0,
        "confidence_label": "Low",
        "method": "pose_only_body_normalized_wrist_kinematics",
        "impact_is_proxy": True,
        "impact_definition": (
            "Estimated from the first major body-normalized wrist-speed peak "
            "after the detected backlift turning point. Bat-ball contact is "
            "not directly observed because this stage does not track the bat or ball."
        ),
        "recommended_input": (
            "One clearly visible batting stroke containing setup, backlift, "
            "downswing and follow-through."
        ),
        "quality_flags": [],
        "candidate_stroke_peaks": 0,
        "phases": {},
    }

    if n < 8:
        base_result["quality_flags"].append("video_segment_too_short")
        return base_result

    x_raw = []
    y_raw = []
    observed_mask = []
    visibility = []

    for record in frame_records:
        wrist_rel = record.get("wrist_rel")
        is_valid = bool(record.get("phase_valid")) and wrist_rel is not None

        if is_valid:
            try:
                x_raw.append(float(wrist_rel[0]))
                y_raw.append(float(wrist_rel[1]))
                observed_mask.append(True)
            except (TypeError, ValueError, IndexError):
                x_raw.append(float("nan"))
                y_raw.append(float("nan"))
                observed_mask.append(False)
        else:
            x_raw.append(float("nan"))
            y_raw.append(float("nan"))
            observed_mask.append(False)

        try:
            visibility.append(float(record.get("average_visibility", 0.0)))
        except (TypeError, ValueError):
            visibility.append(0.0)

    observed_mask = np.asarray(observed_mask, dtype=bool)
    observed_count = int(observed_mask.sum())
    minimum_observations = max(10, int(round(n * 0.15)))

    if observed_count < minimum_observations:
        base_result["quality_flags"].append("insufficient_wrist_tracking")
        return base_result

    max_gap = max(2, int(round(fps * 0.10)))
    x = _fill_signal(x_raw, max_gap, np)
    y = _fill_signal(y_raw, max_gap, np)

    if not np.isfinite(x).all() or not np.isfinite(y).all():
        base_result["quality_flags"].append("wrist_signal_unusable")
        return base_result

    position_window = _odd_window(max(3, int(round(fps * 0.10))))
    x_smooth = _moving_average(x, position_window, np)
    y_smooth = _moving_average(y, position_window, np)

    dt = 1.0 / fps
    vx = np.gradient(x_smooth, dt)
    vy = np.gradient(y_smooth, dt)
    speed = np.sqrt(vx * vx + vy * vy)

    speed_window = _odd_window(max(3, int(round(fps * 0.08))))
    speed = _moving_average(speed, speed_window, np)

    # Keep the extreme edges out of peak selection because numerical gradients
    # are less stable there and a clipped video may begin/end mid-motion.
    edge = max(2, int(round(fps * 0.08)))
    eligible_start = min(edge, n - 1)
    eligible_end = max(eligible_start + 1, n - edge)
    eligible_speed = speed[eligible_start:eligible_end]

    if eligible_speed.size < 4:
        base_result["quality_flags"].append("insufficient_temporal_signal")
        return base_result

    p10 = float(np.percentile(eligible_speed, 10))
    p30 = float(np.percentile(eligible_speed, 30))
    p50 = float(np.percentile(eligible_speed, 50))
    p75 = float(np.percentile(eligible_speed, 75))
    p90 = float(np.percentile(eligible_speed, 90))
    mad = float(np.median(np.abs(eligible_speed - p50)))

    # Two velocity thresholds serve different purposes. A lower motion
    # threshold detects the relatively slow setup-to-backlift transition and
    # the end of follow-through. A stronger activity threshold is reserved for
    # the high-speed downswing / impact-proxy region. Using one threshold for
    # both can miss a deliberately slow backlift.
    motion_threshold = max(
        0.035,
        p30 + 0.06 * max(p90 - p30, 0.0),
    )

    quiet_threshold = max(
        0.030,
        p10 + 0.03 * max(p90 - p10, 0.0),
    )

    robust_threshold = min(
        p50 + 2.0 * mad,
        p75,
    )

    activity_threshold = max(
        robust_threshold,
        p30 + 0.25 * max(p90 - p30, 0.0),
        0.08,
    )

    global_peak = int(np.argmax(eligible_speed)) + eligible_start
    global_peak_speed = float(speed[global_peak])

    if global_peak_speed <= max(activity_threshold * 1.05, 0.05):
        base_result["quality_flags"].append("no_clear_batting_motion_detected")
        return base_result

    quiet_frames = max(3, int(round(fps * 0.18)))

    # Primary onset detector: departure from an early setup baseline. This is
    # intentionally position-based as well as speed-based so a slow, controlled
    # backlift is not missed simply because its velocity is much lower than the
    # subsequent downswing. The preferred NeuroBat input therefore starts with
    # a short visible setup period.
    seed_frames = min(
        max(3, int(round(fps * 0.25))),
        max(3, int(round(n * 0.15))),
    )
    seed_frames = min(seed_frames, max(3, global_peak))

    seed_x = float(np.median(x_smooth[:seed_frames]))
    seed_y = float(np.median(y_smooth[:seed_frames]))
    seed_displacement = np.sqrt(
        (x_smooth[:seed_frames] - seed_x) ** 2
        +
        (y_smooth[:seed_frames] - seed_y) ** 2
    )
    seed_noise = float(np.median(seed_displacement))
    seed_mad = float(np.median(np.abs(seed_displacement - seed_noise)))

    displacement_from_seed = np.sqrt(
        (x_smooth - seed_x) ** 2
        +
        (y_smooth - seed_y) ** 2
    )
    pre_peak_max_displacement = float(np.max(displacement_from_seed[:global_peak + 1]))
    onset_displacement_threshold = max(
        0.045,
        seed_noise + 4.0 * seed_mad + 0.015,
        0.025 * pre_peak_max_displacement,
    )

    onset_run = max(2, int(round(fps * 0.08)))
    action_start = None
    for idx in range(seed_frames, max(seed_frames, global_peak - onset_run + 2)):
        disp_window = displacement_from_seed[idx:idx + onset_run]
        if disp_window.size != onset_run:
            continue
        displaced = float((disp_window >= onset_displacement_threshold).mean()) >= 0.67
        if displaced:
            action_start = idx
            break

    if action_start is None:
        action_start = _find_motion_onset_before(
            speed,
            global_peak,
            motion_threshold,
            quiet_frames,
            max(2, int(round(fps * 0.10))),
            max(quiet_frames + 3, int(round(fps * 3.0))),
        )

    if action_start is None:
        action_start = max(0, global_peak - int(round(fps * 1.20)))
        base_result["quality_flags"].append("setup_onset_estimated_without_quiet_baseline")

    # If the action starts extremely close to the video beginning, the seed
    # probably did not contain a true quiet setup. Keep the segmentation but
    # report the limitation.
    if action_start <= max(2, int(round(fps * 0.12))):
        base_result["quality_flags"].append("limited_pre_action_setup")

    # Setup baseline is a short quiet period immediately before action onset.
    setup_target = max(4, int(round(fps * 0.40)))
    setup_start = max(0, action_start - setup_target)
    setup_end = action_start - 1

    if setup_end >= setup_start:
        setup_slice = slice(setup_start, setup_end + 1)
        baseline_x = float(np.median(x_smooth[setup_slice]))
        baseline_y = float(np.median(y_smooth[setup_slice]))
    else:
        baseline_x = float(x_smooth[action_start])
        baseline_y = float(y_smooth[action_start])
        base_result["quality_flags"].append("setup_phase_not_visible")

    # Estimate where the high-speed stroke begins. This bounds the backlift
    # turning-point search so the detector does not accidentally select a large
    # follow-forward displacement that occurs during the downswing itself.
    high_motion_threshold = max(
        activity_threshold,
        0.35 * global_peak_speed,
    )
    high_active_frames = max(2, int(round(fps * 0.06)))
    high_motion_start = None
    for idx in range(action_start + 1, max(action_start + 2, global_peak - high_active_frames + 2)):
        window = speed[idx:idx + high_active_frames]
        if (
            window.size == high_active_frames
            and float((window >= high_motion_threshold).mean()) >= 0.67
        ):
            high_motion_start = idx
            break

    if high_motion_start is None:
        high_motion_start = max(
            action_start + 2,
            global_peak - max(3, int(round(fps * 0.35))),
        )
        base_result["quality_flags"].append("downswing_acceleration_onset_estimated")

    # Detect the backlift turning point. For a visibly vertical backlift, the
    # highest wrist point (minimum image y) is a useful proxy. For lateral
    # backlifts, use maximum setup displacement, but only BEFORE the high-speed
    # downswing region. This makes the fallback more robust to lateral styles.
    pre_peak_end = min(
        global_peak - 1,
        max(action_start + 2, high_motion_start),
    )

    if pre_peak_end <= action_start + 1:
        base_result["quality_flags"].append("insufficient_pre_impact_motion")
        return base_result

    search_indices = np.arange(action_start, pre_peak_end + 1)
    pre_y = y_smooth[action_start:pre_peak_end + 1]
    vertical_excursion = float(np.percentile(pre_y, 90) - np.percentile(pre_y, 10))

    displacement = np.sqrt(
        (x_smooth - baseline_x) ** 2
        +
        (y_smooth - baseline_y) ** 2
    )

    if vertical_excursion >= 0.08:
        apex_index = int(search_indices[int(np.argmin(pre_y))])
        apex_method = "highest_wrist_turning_point"
    else:
        pre_displacement = displacement[action_start:pre_peak_end + 1]
        apex_index = int(search_indices[int(np.argmax(pre_displacement))])
        apex_method = "maximum_pre_downswing_displacement_fallback"
        base_result["quality_flags"].append("low_vertical_backlift_excursion")

    # Ensure enough temporal separation for backlift and downswing. The apex is
    # the transition event: the backlift ends immediately before it and the
    # downswing starts at/just after it.
    min_phase_frames = max(2, int(round(fps * 0.06)))
    apex_index = max(action_start + min_phase_frames, apex_index)
    apex_index = min(global_peak - min_phase_frames, apex_index)

    if apex_index <= action_start or apex_index >= global_peak:
        apex_index = max(action_start + 1, global_peak - max(2, int(round(fps * 0.25))))
        apex_method = "temporal_fallback"
        base_result["quality_flags"].append("backlift_turning_point_low_confidence")

    downswing_start = min(global_peak - 1, max(action_start + 1, apex_index + 1))

    # Impact proxy: choose the first major local speed peak after downswing onset,
    # constrained to a broad 0.8 s window. This is preferable to blindly choosing
    # a later follow-through maximum.
    impact_search_end = min(
        n - 1,
        downswing_start + max(4, int(round(fps * 0.80))),
    )
    impact_segment = speed[downswing_start:impact_search_end + 1]
    segment_peak_speed = float(np.max(impact_segment)) if impact_segment.size else 0.0
    major_threshold = max(
        activity_threshold,
        0.82 * segment_peak_speed,
    )
    local = _local_peaks(
        impact_segment,
        major_threshold,
        max(2, int(round(fps * 0.05))),
        np,
    )

    if local:
        impact_index = downswing_start + local[0]
    else:
        impact_index = downswing_start + int(np.argmax(impact_segment))
        base_result["quality_flags"].append("impact_proxy_used_interval_maximum")

    # If the selected peak is implausibly close to downswing onset, prefer the
    # strongest point a little later so the downswing remains a non-empty phase.
    if impact_index - downswing_start < min_phase_frames:
        delayed_start = min(impact_search_end, downswing_start + min_phase_frames)
        if delayed_start < impact_search_end:
            delayed_segment = speed[delayed_start:impact_search_end + 1]
            impact_index = delayed_start + int(np.argmax(delayed_segment))

    # Small impact window around the proxy event; at 30 fps this is normally
    # +/- 1 frame. It is intentionally a window, not a claim of exact contact.
    impact_radius = max(1, int(round(fps * 0.02)))
    impact_start = max(downswing_start + 1, impact_index - impact_radius)
    impact_end = min(n - 1, impact_index + impact_radius)

    # Follow-through is measured until the last sustained wrist motion inside a
    # bounded post-impact horizon. Using the last active portion rather than the
    # first quiet dip prevents a brief deceleration around the impact proxy from
    # prematurely terminating the follow-through.
    follow_start_candidate = impact_end + 1
    follow_search_end = min(
        n - 1,
        impact_end + max(6, int(round(fps * 1.50))),
    )
    follow_speed = speed[follow_start_candidate:follow_search_end + 1]
    active_follow = np.where(follow_speed >= quiet_threshold)[0]

    if active_follow.size:
        follow_end = follow_start_candidate + int(active_follow[-1])
        if follow_end >= follow_search_end and follow_search_end < n - 1:
            base_result["quality_flags"].append("follow_through_end_reached_search_horizon")
    else:
        follow_end = min(
            n - 1,
            impact_end + max(3, int(round(fps * 0.30))),
        )
        base_result["quality_flags"].append("follow_through_low_motion")

    # Make ranges non-overlapping and ordered.
    backlift_start = action_start
    backlift_end = max(backlift_start, downswing_start - 1)
    downswing_end = max(downswing_start, impact_start - 1)
    follow_start = min(n - 1, impact_end + 1)
    follow_end = max(follow_start, follow_end)

    # If setup is absent, keep it unavailable rather than stealing action frames.
    ranges = {}
    if setup_end >= setup_start:
        ranges["setup"] = (setup_start, setup_end)
    ranges["backlift"] = (backlift_start, backlift_end)
    ranges["downswing"] = (downswing_start, downswing_end)
    ranges["impact"] = (impact_start, impact_end)
    ranges["follow_through"] = (follow_start, follow_end)

    # Count widely separated strong motion peaks. More than one can indicate an
    # untrimmed multi-stroke clip; the detector still selects the dominant stroke.
    strong_peaks = _local_peaks(
        speed,
        max(p90 * 0.85, global_peak_speed * 0.65),
        max(3, int(round(fps * 0.90))),
        np,
    )
    candidate_peak_count = len(strong_peaks)
    if candidate_peak_count > 1:
        base_result["quality_flags"].append("multiple_strong_stroke_candidates")

    selected_start = ranges.get("setup", ranges["backlift"])[0]
    selected_end = ranges["follow_through"][1]
    selected_mask = observed_mask[selected_start:selected_end + 1]
    coverage = float(selected_mask.mean()) if selected_mask.size else 0.0

    selected_visibility = visibility[selected_start:selected_end + 1]
    mean_visibility = (
        float(np.mean(selected_visibility))
        if selected_visibility
        else 0.0
    )

    denom = max(p90 - p50, 1e-6)
    prominence = _clip((float(speed[impact_index]) - p50) / denom)

    setup_duration = (
        (setup_end - setup_start + 1) / fps
        if setup_end >= setup_start
        else 0.0
    )
    backlift_duration = (backlift_end - backlift_start + 1) / fps
    downswing_duration = (downswing_end - downswing_start + 1) / fps
    follow_duration = (follow_end - follow_start + 1) / fps

    # Broad ranges are used only to judge whether segmentation is temporally
    # coherent; they are NOT ideal technique targets.
    duration_quality = (
        _duration_score(backlift_duration, 0.03, 0.10, 1.20, 2.50)
        +
        _duration_score(downswing_duration, 0.03, 0.08, 0.70, 1.20)
        +
        _duration_score(follow_duration, 0.03, 0.10, 1.20, 2.50)
    ) / 3.0

    setup_quality = _clip(setup_duration / 0.25)

    confidence = (
        0.35 * coverage
        + 0.25 * prominence
        + 0.15 * _clip(mean_visibility)
        + 0.15 * duration_quality
        + 0.10 * setup_quality
    )

    if candidate_peak_count > 1:
        confidence -= 0.08
    if "setup_onset_estimated_without_quiet_baseline" in base_result["quality_flags"]:
        confidence -= 0.05
    if "backlift_turning_point_low_confidence" in base_result["quality_flags"]:
        confidence -= 0.08

    confidence = round(_clip(confidence), 3)

    phases_public = {}
    for name in PHASE_ORDER:
        if name not in ranges:
            continue
        start, end = ranges[name]
        start_frame = int(frame_records[start].get("frame_index", start + 1))
        end_frame = int(frame_records[end].get("frame_index", end + 1))
        phases_public[name] = {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time_seconds": round((start_frame - 1) / fps, 3),
            "end_time_seconds": round((end_frame - 1) / fps, 3),
            "duration_seconds": round((end - start + 1) / fps, 3),
        }

    impact_frame = int(frame_records[impact_index].get("frame_index", impact_index + 1))

    return {
        "status": "detected",
        "confidence": confidence,
        "confidence_label": _confidence_label(confidence),
        "method": "pose_only_body_normalized_wrist_kinematics",
        "backlift_turning_point_method": apex_method,
        "impact_is_proxy": True,
        "impact_definition": base_result["impact_definition"],
        "impact_frame": impact_frame,
        "impact_time_seconds": round((impact_frame - 1) / fps, 3),
        "impact_proxy_wrist_speed": round(float(speed[impact_index]), 4),
        "onset_displacement_threshold": round(float(onset_displacement_threshold), 4),
        "motion_threshold": round(float(motion_threshold), 4),
        "quiet_threshold": round(float(quiet_threshold), 4),
        "activity_threshold": round(float(activity_threshold), 4),
        "candidate_stroke_peaks": candidate_peak_count,
        "selected_stroke_start_frame": int(frame_records[selected_start].get("frame_index", selected_start + 1)),
        "selected_stroke_end_frame": int(frame_records[selected_end].get("frame_index", selected_end + 1)),
        "wrist_tracking_coverage": round(coverage, 3),
        "mean_selected_visibility": round(mean_visibility, 3),
        "recommended_input": base_result["recommended_input"],
        "quality_flags": base_result["quality_flags"],
        "phases": phases_public,
        # Internal keys consumed by biomechanics.py and removed before API output.
        "_ranges": ranges,
        "_wrist_speed": speed,
        "_impact_array_index": impact_index,
    }


def phase_name_for_array_index(array_index, detection):
    """Return public phase label for a zero-based frame-array index."""
    ranges = detection.get("_ranges", {}) if detection else {}
    for name in PHASE_ORDER:
        if name not in ranges:
            continue
        start, end = ranges[name]
        if start <= array_index <= end:
            return name
    return None


def public_phase_detection(detection):
    """Remove internal NumPy arrays/ranges before JSON serialization."""
    if not detection:
        return {}
    return {
        key: value
        for key, value in detection.items()
        if not key.startswith("_")
    }
