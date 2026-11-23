"""Pure, hardware-independent logic for the suspended G1 ankle PR test."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Sequence
import xml.etree.ElementTree as ET

import numpy as np


MODE_PR = 0
MODE_MACHINE = 4
WIRE_MOTOR_COUNT = 35
MODEL_MOTOR_COUNT = 29
ANKLE_SLOTS = (4, 5, 10, 11)
ANKLE_LIMITS_RAD = {
    4: (-0.87267, 0.5236),
    5: (-0.2618, 0.2618),
    10: (-0.87267, 0.5236),
    11: (-0.2618, 0.2618),
}
AXIS_SLOTS = {
    "left-pitch": 4,
    "left-roll": 5,
    "right-pitch": 10,
    "right-roll": 11,
}

ACTIVE_JOINT_NAMES = {
    0: "left_hip_pitch_joint", 1: "left_hip_roll_joint",
    2: "left_hip_yaw_joint", 3: "left_knee_joint",
    4: "left_ankle_pitch_joint", 5: "left_ankle_roll_joint",
    6: "right_hip_pitch_joint", 7: "right_hip_roll_joint",
    8: "right_hip_yaw_joint", 9: "right_knee_joint",
    10: "right_ankle_pitch_joint", 11: "right_ankle_roll_joint",
    12: "waist_yaw_joint", 15: "left_shoulder_pitch_joint",
    16: "left_shoulder_roll_joint", 17: "left_shoulder_yaw_joint",
    18: "left_elbow_joint", 19: "left_wrist_roll_joint",
    22: "right_shoulder_pitch_joint", 23: "right_shoulder_roll_joint",
    24: "right_shoulder_yaw_joint", 25: "right_elbow_joint",
    26: "right_wrist_roll_joint",
}


def smoothstep5(value: float) -> float:
    """Quintic interpolation with zero velocity and acceleration at each end."""
    u = float(np.clip(value, 0.0, 1.0))
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


def smoothstep5_rate(value: float) -> float:
    """Derivative of :func:`smoothstep5` with respect to its argument."""
    u = float(np.clip(value, 0.0, 1.0))
    return 30.0 * u * u * (1.0 - u) * (1.0 - u)


def load_active_joint_limits(urdf_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load finite SDK-slot position limits from the selected G1 URDF."""
    root = ET.parse(Path(urdf_path)).getroot()
    by_name = {node.attrib["name"]: node for node in root.findall("joint")}
    lower = np.full(WIRE_MOTOR_COUNT, -np.inf, dtype=np.float64)
    upper = np.full(WIRE_MOTOR_COUNT, np.inf, dtype=np.float64)
    for slot, name in ACTIVE_JOINT_NAMES.items():
        node = by_name.get(name)
        limit = None if node is None else node.find("limit")
        if limit is None or limit.get("lower") is None or limit.get("upper") is None:
            raise ValueError(f"URDF lacks finite position limits for {name!r}")
        lower[slot] = float(limit.get("lower"))
        upper[slot] = float(limit.get("upper"))
        if not (math.isfinite(lower[slot]) and math.isfinite(upper[slot])
                and lower[slot] < 0.0 < upper[slot]):
            raise ValueError(f"URDF zero is not strictly inside limits for {name!r}")
    return lower, upper


@dataclass(frozen=True)
class TrajectorySample:
    offset_rad: float
    velocity_rad_s: float
    done: bool
    segment: str
    cycle: int


@dataclass(frozen=True)
class VelocityProbeSample:
    target_dq_rad_s: float
    done: bool
    segment: str
    cycle: int


@dataclass(frozen=True)
class _Segment:
    start: float
    end: float
    duration: float
    name: str
    cycle: int
    moving: bool


class AxisTrajectory:
    """Generate +A, 0, -A, 0 with a hold after every waypoint."""

    def __init__(self, amplitude_rad: float, ramp_s: float, hold_s: float, cycles: int):
        values = (amplitude_rad, ramp_s, hold_s)
        if not all(math.isfinite(item) for item in values):
            raise ValueError("trajectory values must be finite")
        if amplitude_rad <= 0 or ramp_s <= 0 or hold_s < 0 or cycles < 1:
            raise ValueError("amplitude/ramp/cycles must be positive and hold nonnegative")
        self.amplitude_rad = float(amplitude_rad)
        self.ramp_s = float(ramp_s)
        self.hold_s = float(hold_s)
        self.cycles = int(cycles)
        self._segments: list[_Segment] = []
        current = 0.0
        for cycle in range(1, self.cycles + 1):
            for name, target in (("positive", self.amplitude_rad), ("zero-after-positive", 0.0),
                                 ("negative", -self.amplitude_rad), ("zero-after-negative", 0.0)):
                self._segments.append(_Segment(current, target, self.ramp_s,
                                               f"ramp-{name}", cycle, True))
                if self.hold_s > 0:
                    self._segments.append(_Segment(target, target, self.hold_s,
                                                   f"hold-{name}", cycle, False))
                current = target
        self.duration_s = sum(item.duration for item in self._segments)

    def sample(self, elapsed_s: float) -> TrajectorySample:
        if not math.isfinite(elapsed_s):
            raise ValueError("elapsed time must be finite")
        if elapsed_s <= 0:
            first = self._segments[0]
            return TrajectorySample(0.0, 0.0, False, first.name, first.cycle)
        remaining = float(elapsed_s)
        for item in self._segments:
            if remaining < item.duration:
                if not item.moving:
                    return TrajectorySample(item.end, 0.0, False, item.name, item.cycle)
                u = remaining / item.duration
                blend = smoothstep5(u)
                offset = item.start + (item.end - item.start) * blend
                velocity = ((item.end - item.start) * smoothstep5_rate(u)
                            / item.duration)
                return TrajectorySample(offset, velocity, False, item.name, item.cycle)
            remaining -= item.duration
        final = self._segments[-1]
        return TrajectorySample(final.end, 0.0, True, "complete", self.cycles)


class VelocityProbeTrajectory:
    """Generate a smooth 0,+V,0,-V,0 derivative-only probe.

    The command has no position component.  This is used when the official
    service publishes a zero-gain LowCmd, because copying LowState.q into a PR
    position target would assume the coordinate semantics being measured.
    """

    def __init__(self, speed_rad_s: float, ramp_s: float, hold_s: float, cycles: int):
        values = (speed_rad_s, ramp_s, hold_s)
        if not all(math.isfinite(item) for item in values):
            raise ValueError("velocity probe values must be finite")
        if speed_rad_s <= 0 or ramp_s <= 0 or hold_s < 0 or cycles < 1:
            raise ValueError("speed/ramp/cycles must be positive and hold nonnegative")
        self.speed_rad_s = float(speed_rad_s)
        self.ramp_s = float(ramp_s)
        self.hold_s = float(hold_s)
        self.cycles = int(cycles)
        self._segments: list[_Segment] = []
        current = 0.0
        for cycle in range(1, self.cycles + 1):
            for name, target in (("positive", self.speed_rad_s),
                                 ("zero-after-positive", 0.0),
                                 ("negative", -self.speed_rad_s),
                                 ("zero-after-negative", 0.0)):
                self._segments.append(_Segment(current, target, self.ramp_s,
                                               f"ramp-{name}", cycle, True))
                if self.hold_s > 0:
                    self._segments.append(_Segment(target, target, self.hold_s,
                                                   f"hold-{name}", cycle, False))
                current = target
        self.duration_s = sum(item.duration for item in self._segments)

    def sample(self, elapsed_s: float) -> VelocityProbeSample:
        if not math.isfinite(elapsed_s):
            raise ValueError("elapsed time must be finite")
        if elapsed_s <= 0:
            first = self._segments[0]
            return VelocityProbeSample(0.0, False, first.name, first.cycle)
        remaining = float(elapsed_s)
        for item in self._segments:
            if remaining < item.duration:
                if not item.moving:
                    return VelocityProbeSample(item.end, False, item.name, item.cycle)
                u = remaining / item.duration
                target = item.start + (item.end - item.start) * smoothstep5(u)
                return VelocityProbeSample(target, False, item.name, item.cycle)
            remaining -= item.duration
        return VelocityProbeSample(0.0, True, "complete", self.cycles)


class ZeroPostureTrajectory:
    """Quintic interpolation from one captured posture to SDK ``q=0``."""

    def __init__(self, initial_q, active_slots, minimum_duration_s: float,
                 max_target_speed_rad_s: float):
        self.initial_q = np.asarray(initial_q, dtype=np.float64).copy()
        self.active_slots = np.asarray(active_slots, dtype=np.int64).copy()
        if self.initial_q.shape != (WIRE_MOTOR_COUNT,):
            raise ValueError("initial posture must contain 35 wire slots")
        if (self.active_slots.ndim != 1 or self.active_slots.size == 0
                or np.any(self.active_slots < 0)
                or np.any(self.active_slots >= WIRE_MOTOR_COUNT)):
            raise ValueError("active slots are invalid")
        values = (minimum_duration_s, max_target_speed_rad_s)
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("zero-posture duration and speed must be positive")
        if not np.isfinite(self.initial_q[self.active_slots]).all():
            raise ValueError("initial posture contains non-finite active joints")
        # max(d smoothstep5/dt) = 1.875 / duration.
        speed_limited_duration = (
            1.875 * float(np.max(np.abs(self.initial_q[self.active_slots])))
            / float(max_target_speed_rad_s))
        self.duration_s = max(float(minimum_duration_s), speed_limited_duration)

    def sample(self, elapsed_s: float) -> tuple[np.ndarray, np.ndarray, float, bool]:
        if not math.isfinite(elapsed_s):
            raise ValueError("elapsed time must be finite")
        u = float(np.clip(elapsed_s / self.duration_s, 0.0, 1.0))
        blend = smoothstep5(u)
        rate = smoothstep5_rate(u) / self.duration_s
        q = self.initial_q.copy()
        dq = np.zeros(WIRE_MOTOR_COUNT, dtype=np.float64)
        q[self.active_slots] = self.initial_q[self.active_slots] * (1.0 - blend)
        dq[self.active_slots] = -self.initial_q[self.active_slots] * rate
        return q, dq, blend, bool(elapsed_s >= self.duration_s)


class SequentialAnkleZeroTrajectory:
    """Move selected ankle slots to zero one at a time.

    Every non-selected slot remains at the captured posture for the complete
    trajectory.  Each selected slot receives its own quintic segment so that
    only one ankle target changes at any instant.
    """

    def __init__(self, initial_q, ankle_slots, minimum_slot_duration_s: float,
                 max_target_speed_rad_s: float):
        self.initial_q = np.asarray(initial_q, dtype=np.float64).copy()
        self.ankle_slots = tuple(int(slot) for slot in ankle_slots)
        if self.initial_q.shape != (WIRE_MOTOR_COUNT,):
            raise ValueError("initial posture must contain 35 wire slots")
        if (not self.ankle_slots or len(set(self.ankle_slots)) != len(self.ankle_slots)
                or any(slot not in ANKLE_SLOTS for slot in self.ankle_slots)):
            raise ValueError("sequential zero slots must be unique ankle slots")
        values = (minimum_slot_duration_s, max_target_speed_rad_s)
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("ankle-zero duration and speed must be positive")
        if not np.isfinite(self.initial_q[list(self.ankle_slots)]).all():
            raise ValueError("initial posture contains non-finite ankle angles")
        self.slot_durations_s = tuple(
            max(float(minimum_slot_duration_s),
                1.875 * abs(float(self.initial_q[slot]))
                / float(max_target_speed_rad_s))
            for slot in self.ankle_slots)
        self.duration_s = float(sum(self.slot_durations_s))

    def sample(self, elapsed_s: float):
        """Return ``q, dq, progress, done, moving_slot``."""
        if not math.isfinite(elapsed_s):
            raise ValueError("elapsed time must be finite")
        elapsed = max(0.0, float(elapsed_s))
        q = self.initial_q.copy()
        dq = np.zeros(WIRE_MOTOR_COUNT, dtype=np.float64)
        segment_start = 0.0
        for index, (slot, duration) in enumerate(
                zip(self.ankle_slots, self.slot_durations_s)):
            segment_end = segment_start + duration
            if elapsed < segment_end:
                # Earlier ankles have completed; later ankles still hold their
                # captured targets.
                if index:
                    q[list(self.ankle_slots[:index])] = 0.0
                u = (elapsed - segment_start) / duration
                blend = smoothstep5(u)
                rate = smoothstep5_rate(u) / duration
                q[slot] = self.initial_q[slot] * (1.0 - blend)
                dq[slot] = -self.initial_q[slot] * rate
                progress = min(elapsed / self.duration_s, 1.0)
                return q, dq, progress, False, slot
            segment_start = segment_end
        q[list(self.ankle_slots)] = 0.0
        return q, dq, 1.0, True, None


class SpeedWarningTracker:
    """Diagnostic episodes only; no control decisions or hardware dependencies.

    Episodes are per slot/phase, use feedback sample times, and finish on a
    below-threshold sample, phase change or explicit truncation.  Duration is
    sampled duration, not an estimate of time between feedback samples.
    """

    def __init__(self, threshold_rad_s, slots):
        if not math.isfinite(threshold_rad_s) or threshold_rad_s <= 0:
            raise ValueError("speed warning threshold must be positive and finite")
        self.threshold = float(threshold_rad_s)
        self.slots = tuple(int(s) for s in slots)
        self.active = {}
        self.last_ns = None

    def finish(self, now_ns, reason="stream_end"):
        events = []
        for slot, episode in list(self.active.items()):
            events.append(("speed_warning_ended", {
                **episode, "end_ns": int(now_ns),
                "duration_s": max(0, int(now_ns) - episode["start_ns"]) * 1e-9,
                "end_reason": reason, "truncated": reason != "below_threshold"}))
            del self.active[slot]
        return events

    def update(self, sample_ns, phase, q, dq, target_q, target_dq):
        sample_ns = int(sample_ns)
        if self.last_ns is not None and sample_ns <= self.last_ns:
            return []
        self.last_ns = sample_ns
        events = []
        for slot in self.slots:
            episode = self.active.get(slot)
            exceeds = abs(float(dq[slot])) > self.threshold
            if episode is not None and (episode["phase"] != phase or not exceeds):
                reason = "phase_change" if episode["phase"] != phase else "below_threshold"
                events.append(("speed_warning_ended", {
                    **episode, "end_ns": sample_ns,
                    "duration_s": (sample_ns - episode["start_ns"]) * 1e-9,
                    "end_reason": reason, "truncated": reason != "below_threshold"}))
                del self.active[slot]
                episode = None
            if not exceeds:
                continue
            if episode is None:
                episode = dict(slot=slot, joint=ACTIVE_JOINT_NAMES.get(slot, "unknown"),
                               phase=str(phase), start_ns=sample_ns,
                               threshold_rad_s=self.threshold,
                               peak_abs_dq_rad_s=abs(float(dq[slot])),
                               peak_dq_rad_s=float(dq[slot]), peak_ns=sample_ns,
                               target_dq_at_peak_rad_s=float(target_dq[slot]),
                               position_error_at_peak_rad=float(target_q[slot] - q[slot]))
                self.active[slot] = episode
                events.append(("speed_warning_started", {
                    **episode, "dq_rad_s": float(dq[slot]),
                    "target_dq_rad_s": float(target_dq[slot]),
                    "position_error_rad": float(target_q[slot] - q[slot])}))
            if abs(float(dq[slot])) > episode["peak_abs_dq_rad_s"]:
                episode.update(peak_abs_dq_rad_s=abs(float(dq[slot])),
                               peak_dq_rad_s=float(dq[slot]), peak_ns=sample_ns,
                               target_dq_at_peak_rad_s=float(target_dq[slot]),
                               position_error_at_peak_rad=float(target_q[slot] - q[slot]))
        return events


class ReturnTrajectory:
    """Smoothly return an arbitrary current offset to the baseline."""

    def __init__(self, start_offset_rad: float, duration_s: float):
        if not math.isfinite(start_offset_rad) or not math.isfinite(duration_s):
            raise ValueError("return trajectory values must be finite")
        if duration_s <= 0:
            raise ValueError("return duration must be positive")
        self.start = float(start_offset_rad)
        self.duration_s = float(duration_s)

    def sample(self, elapsed_s: float) -> TrajectorySample:
        if elapsed_s >= self.duration_s:
            return TrajectorySample(0.0, 0.0, True, "return-complete", 0)
        u = float(np.clip(elapsed_s / self.duration_s, 0.0, 1.0))
        offset = self.start * (1.0 - smoothstep5(u))
        velocity = -self.start * smoothstep5_rate(u) / self.duration_s
        return TrajectorySample(offset, velocity, False, "return", 0)


def overlay_axis(baseline_q, baseline_dq, slot: int,
                 offset_rad: float, velocity_rad_s: float):
    """Return copies of the captured command with one virtual PR axis changed."""
    if slot not in ANKLE_SLOTS:
        raise ValueError(f"slot {slot} is not a tested ankle slot")
    q = np.asarray(baseline_q, dtype=np.float64).copy()
    dq = np.asarray(baseline_dq, dtype=np.float64).copy()
    if q.shape != (WIRE_MOTOR_COUNT,) or dq.shape != (WIRE_MOTOR_COUNT,):
        raise ValueError("baseline command must contain 35 wire slots")
    if not all(math.isfinite(item) for item in (offset_rad, velocity_rad_s)):
        raise ValueError("axis command must be finite")
    q[slot] += float(offset_rad)
    dq[slot] += float(velocity_rad_s)
    return q, dq


def _finite_command(command) -> bool:
    arrays = (command.q, command.dq, command.tau, command.kp, command.kd)
    return all(np.asarray(item).shape == (WIRE_MOTOR_COUNT,)
               and np.isfinite(item).all() for item in arrays)


def select_official_baseline(states: Sequence, commands: Sequence,
                             invalid_state_times: Iterable[int], active_slots,
                             window_s: float = 2.0, max_state_dq: float = 0.2,
                             max_command_span_rad: float = 0.10,
                             min_ankle_kp: float = 1.0):
    """Validate a stable official PR command window without assuming feedback coordinates.

    In particular, this deliberately does not compare the official PD equation with
    ``tau_est``: that comparison would assume the LowState coordinates the experiment
    is intended to identify.
    """
    if not states or not commands:
        raise RuntimeError("official capture requires both LowState and LowCmd")
    end_ns = min(states[-1].monotonic_ns, commands[-1].monotonic_ns)
    start_ns = end_ns - int(window_s * 1e9)
    selected_commands = [item for item in commands
                         if start_ns <= item.monotonic_ns <= end_ns]
    selected_states = [item for item in states
                       if start_ns <= item.monotonic_ns <= end_ns]
    if not selected_commands or not selected_states:
        raise RuntimeError("official capture window is empty")
    span_s = (selected_commands[-1].monotonic_ns
              - selected_commands[0].monotonic_ns) * 1e-9
    if span_s < 0.95 * window_s:
        raise RuntimeError(f"official LowCmd capture spans only {span_s:.3f} s")
    if any(not item.crc_valid for item in selected_commands):
        raise ValueError("official LowCmd capture contains a CRC failure")
    if any(start_ns <= int(item) <= end_ns for item in invalid_state_times):
        raise ValueError("official LowState capture contains an invalid sample")
    expected_mode = {(MODE_MACHINE, MODE_PR)}
    state_modes = {(item.mode_machine, item.mode_pr) for item in selected_states}
    command_modes = {(item.mode_machine, item.mode_pr) for item in selected_commands}
    if state_modes != expected_mode:
        raise ValueError(f"LowState mode changed in capture: {sorted(state_modes)}")
    if command_modes != expected_mode:
        raise ValueError(f"LowCmd mode changed in capture: {sorted(command_modes)}")
    if any(not _finite_command(item) for item in selected_commands):
        raise ValueError("official LowCmd capture contains non-finite or wrong-sized arrays")
    active = np.asarray(active_slots, dtype=np.int64)
    state_dq = np.stack([np.asarray(item.dq) for item in selected_states])
    if not np.isfinite(state_dq).all():
        raise ValueError("official LowState capture contains non-finite dq")
    observed_max_dq = float(np.max(np.abs(state_dq[:, active])))
    if observed_max_dq > max_state_dq:
        raise ValueError(f"robot is not stationary: max |dq|={observed_max_dq:.3f} rad/s")
    command_q = np.stack([item.q for item in selected_commands])
    command_span = np.ptp(command_q[:, active], axis=0)
    observed_span = float(np.max(command_span))
    if observed_span > max_command_span_rad:
        raise ValueError(f"official command is not stable: max q span={observed_span:.3f} rad")
    final = selected_commands[-1]
    if np.any(np.asarray(final.motor_mode)[active] == 0):
        raise ValueError("official command disables one or more active motor slots")
    if np.any(np.asarray(final.kp)[list(ANKLE_SLOTS)] < min_ankle_kp):
        raise ValueError("official ankle Kp is too small for a position-offset test")
    if np.any(np.asarray(final.kp) < 0) or np.any(np.asarray(final.kd) < 0):
        raise ValueError("official command contains negative gains")
    diagnostics = {
        "window_s": span_s,
        "command_samples": len(selected_commands),
        "state_samples": len(selected_states),
        "max_state_dq_rad_s": observed_max_dq,
        "max_command_span_rad": observed_span,
        "last_command_age_ms": float((states[-1].monotonic_ns
                                      - final.monotonic_ns) * 1e-6),
    }
    return final, diagnostics


def select_zero_command_handoff(states: Sequence, commands: Sequence,
                                invalid_state_times: Iterable[int], active_slots,
                                window_s: float = 2.0, max_state_dq: float = 0.2,
                                max_abs_command: float = 1e-6):
    """Validate the G1 ``ai`` zero-LowCmd case before releasing that service.

    This deliberately validates only the command envelope and never copies
    LowState.q into a position target.  The returned LowCmd snapshot supplies
    wire modes/reserve fields; its q values remain dynamically irrelevant
    when used by the legacy derivative-only probe.  A position-control caller
    that later copies LowState.q still assumes PR semantics during takeover;
    validation here does not establish those feedback coordinates.
    """
    if not states or not commands:
        raise RuntimeError("zero-command handoff requires LowState and LowCmd")
    end_ns = min(states[-1].monotonic_ns, commands[-1].monotonic_ns)
    start_ns = end_ns - int(window_s * 1e9)
    selected_commands = [item for item in commands
                         if start_ns <= item.monotonic_ns <= end_ns]
    selected_states = [item for item in states
                       if start_ns <= item.monotonic_ns <= end_ns]
    if not selected_commands or not selected_states:
        raise RuntimeError("zero-command capture window is empty")
    span_s = (selected_commands[-1].monotonic_ns
              - selected_commands[0].monotonic_ns) * 1e-9
    if span_s < 0.95 * window_s:
        raise RuntimeError(f"zero-command capture spans only {span_s:.3f} s")
    if any(not item.crc_valid for item in selected_commands):
        raise ValueError("zero-command capture contains a CRC failure")
    if any(start_ns <= int(item) <= end_ns for item in invalid_state_times):
        raise ValueError("zero-command LowState capture contains an invalid sample")
    expected_mode = {(MODE_MACHINE, MODE_PR)}
    state_modes = {(item.mode_machine, item.mode_pr) for item in selected_states}
    command_modes = {(item.mode_machine, item.mode_pr) for item in selected_commands}
    if state_modes != expected_mode:
        raise ValueError(f"LowState mode changed in capture: {sorted(state_modes)}")
    if command_modes != expected_mode:
        raise ValueError(f"LowCmd mode changed in capture: {sorted(command_modes)}")
    if any(not _finite_command(item) for item in selected_commands):
        raise ValueError("zero-command capture contains non-finite or wrong-sized arrays")
    active = np.asarray(active_slots, dtype=np.int64)
    state_dq = np.stack([np.asarray(item.dq) for item in selected_states])
    observed_max_dq = float(np.max(np.abs(state_dq[:, active])))
    if observed_max_dq > max_state_dq:
        raise ValueError(f"robot is not stationary: max |dq|={observed_max_dq:.3f} rad/s")
    final = selected_commands[-1]
    if np.any(np.asarray(final.motor_mode)[active] == 0):
        raise ValueError("zero-command stream disables one or more active motor slots")
    fields = ("q", "dq", "tau", "kp", "kd")
    observed_max_command = 0.0
    for command in selected_commands:
        for field in fields:
            observed_max_command = max(
                observed_max_command,
                float(np.max(np.abs(np.asarray(getattr(command, field))[active]))))
    if observed_max_command > max_abs_command:
        raise ValueError(
            f"official LowCmd is not a zero-command stream: max field="
            f"{observed_max_command:.3g}")
    diagnostics = {
        "window_s": span_s,
        "command_samples": len(selected_commands),
        "state_samples": len(selected_states),
        "max_state_dq_rad_s": observed_max_dq,
        "max_abs_official_command": observed_max_command,
        "last_command_age_ms": float((states[-1].monotonic_ns
                                      - final.monotonic_ns) * 1e-6),
    }
    return final, diagnostics
