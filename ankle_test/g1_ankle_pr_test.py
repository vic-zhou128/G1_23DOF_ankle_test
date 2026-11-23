#!/usr/bin/env python3
"""Guarded official-derived suspended-G1 ankle PR semantics test.

The default is read-only.  The active path explicitly releases the official
``ai`` service, holds the captured posture while ramping gains and verifying
settling, then waits for ``prepare ANKLE ZERO`` before moving ankle slots
4/5/10/11 to SDK ``q=0`` one at a time, checking settling after every slot.
Every other valid joint remains at its captured target.  After confirmation,
one PR ankle axis at a time is permitted.  ``mode_pr`` is always zero; no AB
command path exists.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
import queue
import signal
import threading
import time

import numpy as np

try:
    from .core import (ANKLE_LIMITS_RAD, ANKLE_SLOTS, AXIS_SLOTS, AxisTrajectory,
                       MODE_MACHINE, MODE_PR, ReturnTrajectory,
                       SequentialAnkleZeroTrajectory, ZeroPostureTrajectory, SpeedWarningTracker,
                       VelocityProbeTrajectory, WIRE_MOTOR_COUNT, overlay_axis,
                       load_active_joint_limits, select_official_baseline,
                       select_zero_command_handoff, ACTIVE_JOINT_NAMES, smoothstep5)
    from .hardware_io import (HardwareTransport, Recorder, new_lowcmd, remote_lt_b)
except ImportError:  # Support ``python ankle_test/g1_ankle_pr_test.py``.
    from core import (ANKLE_LIMITS_RAD, ANKLE_SLOTS, AXIS_SLOTS, AxisTrajectory,
                      MODE_MACHINE, MODE_PR, ReturnTrajectory,
                      SequentialAnkleZeroTrajectory, ZeroPostureTrajectory, SpeedWarningTracker,
                      VelocityProbeTrajectory, WIRE_MOTOR_COUNT, overlay_axis,
                      load_active_joint_limits, select_official_baseline,
                      select_zero_command_handoff, ACTIVE_JOINT_NAMES, smoothstep5)
    from hardware_io import HardwareTransport, Recorder, new_lowcmd, remote_lt_b


HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
REPO_ROOT = (PROJECT_DIR if (PROJECT_DIR / "models/g1_description/g1_23dof_rev_1_0.urdf").is_file()
             else PROJECT_DIR.parent)
DEFAULT_OUTPUT = REPO_ROOT / "outputs/hardware/ankle_test"
DEFAULT_URDF = REPO_ROOT / "models/g1_description/g1_23dof_rev_1_0.urdf"
ACTIVE_SLOTS = np.asarray([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    15, 16, 17, 18, 19, 22, 23, 24, 25, 26,
], dtype=np.int64)
PASSIVE_KD = np.asarray([
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0,
    1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0,
], dtype=np.float64)
POSITION_KP = np.zeros(WIRE_MOTOR_COUNT, dtype=np.float64)
POSITION_KD = np.zeros(WIRE_MOTOR_COUNT, dtype=np.float64)
# The official example uses ankle Kp=40/Kd=1 and still higher gains elsewhere.
# This experiment caps every valid joint at that ankle gain.  Non-ankle joints
# hold the captured posture; only one ankle target moves during zeroing.
POSITION_KP[ACTIVE_SLOTS] = 40.0
POSITION_KD[ACTIVE_SLOTS] = 1.0
ZERO_STRATEGY = "ankle_sequential_zero_position"
ARM_COMMAND = "arm ARM PR HOLD"
SLOT_AXES = {slot: axis for axis, slot in AXIS_SLOTS.items()}


class Phase(Enum):
    OBSERVE = "observe"
    GAIN_RAMP = "gain_ramp"
    GAIN_SETTLING = "gain_settling"
    CAPTURED_HOLD = "captured_hold"
    ZEROING = "zeroing"
    ZERO_SETTLING = "zero_settling"
    ZERO_HOLD = "zero_hold"
    HOLD = "hold"  # Legacy nonzero official-LowCmd handoff.
    RUNNING = "running"
    RETURNING = "returning"
    DEBUG_DAMPING = "debug_damping"
    PROBING = "velocity_probe"
    DAMPING = "protective_damping"
    CLOSED = "closed"


@dataclass(frozen=True)
class TestConfig:
    pitch_amplitude_rad: float
    roll_amplitude_rad: float
    ramp_s: float
    hold_s: float
    cycles: int
    state_stale_ns: int
    max_temperature: float
    speed_warning_rad_s: float
    damping_s: float
    capture_window_s: float
    max_capture_dq: float
    max_command_span_rad: float
    min_ankle_kp: float
    joint_limit_margin_rad: float
    probe_speed_rad_s: float
    probe_ramp_s: float
    probe_hold_s: float
    probe_cycles: int
    probe_ankle_kd: float
    probe_max_displacement_rad: float
    debug_settle_s: float
    zero_min_duration_s: float
    zero_max_target_speed_rad_s: float
    zero_settle_s: float
    zero_settle_timeout_s: float
    zero_position_tolerance_rad: float
    zero_speed_tolerance_rad_s: float
    max_tracking_error_rad: float
    joint_lower_rad: np.ndarray
    joint_upper_rad: np.ndarray
    gain_ramp_s: float = 3.0
    gain_settle_s: float = 1.0
    gain_settle_timeout_s: float = 5.0


class PeriodicThread:
    def __init__(self, period_s, callback):
        self.period_s = float(period_s)
        self.callback = callback
        self.stop_event = threading.Event()
        self.exception = None
        self.thread = threading.Thread(target=self._run, name="ankle-control", daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        deadline = time.perf_counter()
        try:
            while not self.stop_event.is_set():
                self.callback()
                deadline += self.period_s
                remaining = deadline - time.perf_counter()
                if remaining > 0:
                    self.stop_event.wait(remaining)
                elif remaining < -0.1:
                    deadline = time.perf_counter()
        except Exception as exc:  # The main thread reports this in the summary.
            self.exception = exc
            self.stop_event.set()

    def stop(self):
        self.stop_event.set()
        if threading.current_thread() is not self.thread:
            self.thread.join(timeout=3.0)


class AnkleController:
    def __init__(self, transport, recorder, config):
        self.transport = transport
        self.recorder = recorder
        self.config = config
        self.lock = threading.RLock()
        self.phase = Phase.OBSERVE
        self.phase_started_ns = time.monotonic_ns()
        self.baseline = None
        self.axis = ""
        self.slot = None
        self.trajectory = None
        self.current_offset = 0.0
        self.current_velocity = 0.0
        self.segment = ""
        self.cycle = 0
        self.close_after_return = False
        self.damping_until_ns = 0
        self.exit_requested = threading.Event()
        self.writer = None
        self.last_write_ns = None
        self.consecutive_write_failures = 0
        self.control_count = 0
        self.fault_reason = ""
        self.control_strategy = "unarmed"
        self.probe_origin_q = None
        self.debug_origin_q = None
        self.zero_origin_q = None
        self.zero_target_q = None
        self.zeroing_slot = None
        self.zero_slot_index = -1
        self.prepare_target_q = None
        self.zero_confirmed = False
        self.zero_settle_started_ns = None
        self.zero_settle_deadline_ns = None
        self.last_target_q = None
        self.zero_duration_s = 0.0
        self.zero_progress = 0.0
        self.gain_scale = 0.0
        self.gain_settle_started_ns = None
        self.speed_warnings = SpeedWarningTracker(config.speed_warning_rad_s, ACTIVE_SLOTS)

    def _capture_baseline(self):
        states, commands, invalid_times = self.transport.histories()
        baseline, diagnostics = select_official_baseline(
            states, commands, invalid_times, ACTIVE_SLOTS,
            window_s=self.config.capture_window_s,
            max_state_dq=self.config.max_capture_dq,
            max_command_span_rad=self.config.max_command_span_rad,
            min_ankle_kp=self.config.min_ankle_kp)
        self.recorder.event("official_baseline_validated", **diagnostics)
        return baseline

    def _capture_zero_handoff(self):
        states, commands, invalid_times = self.transport.histories()
        baseline, diagnostics = select_zero_command_handoff(
            states, commands, invalid_times, ACTIVE_SLOTS,
            window_s=self.config.capture_window_s,
            max_state_dq=self.config.max_capture_dq)
        self.recorder.event("zero_command_handoff_validated", **diagnostics)
        return baseline

    def arm(self, expected_service, handoff_kind):
        with self.lock:
            if self.phase != Phase.OBSERVE or self.writer is not None:
                raise RuntimeError("controller is already armed or closing")
        if handoff_kind == "position":
            handoff_factory = self._capture_baseline
            released_callback = self._released_position_callback
        elif handoff_kind == ZERO_STRATEGY:
            handoff_factory = self._capture_zero_handoff
            released_callback = self._released_zero_position_callback
        elif handoff_kind == "debug_velocity_probe":
            handoff_factory = self._capture_zero_handoff
            released_callback = self._released_debug_callback
        else:
            raise ValueError(f"unsupported handoff kind {handoff_kind!r}")
        try:
            self.transport.enable_commands(expected_service, handoff_factory,
                                           released_callback)
        except Exception as exc:
            if self.transport.publisher is not None:
                self.fault(f"arm failed after service release: {exc}")
                self._ensure_writer()
            raise

    def _released_position_callback(self, baseline):
        with self.lock:
            self.baseline = baseline
            self.control_strategy = "position"
            self.phase = Phase.HOLD
            self.phase_started_ns = time.monotonic_ns()
        self.recorder.event("custom_lowcmd_armed", mode_pr=MODE_PR,
                            mode_machine=MODE_MACHINE)
        state = self.transport.latest()
        if state is None:
            raise RuntimeError("LowState disappeared during handoff")
        if not self._write(self._build_hold_command(), state):
            self.fault("first reproduced official LowCmd write failed")
        self._ensure_writer()

    def _released_debug_callback(self, baseline):
        with self.lock:
            self.baseline = baseline
            self.control_strategy = "debug_velocity_probe"
            self.phase = Phase.DEBUG_DAMPING
            self.phase_started_ns = time.monotonic_ns()
            self.current_offset = 0.0
            self.current_velocity = 0.0
            self.segment = "debug-damping"
        self.recorder.event("debug_damping_armed", mode_pr=MODE_PR,
                            mode_machine=MODE_MACHINE, ankle_kp=0.0,
                            ankle_tau=0.0, ankle_kd=self.config.probe_ankle_kd)
        state = self.transport.latest()
        if state is None:
            raise RuntimeError("LowState disappeared during debug handoff")
        with self.lock:
            self.debug_origin_q = np.asarray(state.q, dtype=np.float64).copy()
        if not self._write(self._build_damping_command(), state):
            self.fault("first debug damping LowCmd write failed")
        self._ensure_writer()

    def _released_zero_position_callback(self, baseline):
        # Preserve protocol mode/reserve fields immediately so a validation
        # failure after MotionSwitcher release can still enter damping safely.
        with self.lock:
            self.baseline = baseline
        state = self.transport.latest()
        if state is None:
            raise RuntimeError("LowState disappeared during zero-position handoff")
        reason = self._safety_reason(state, time.monotonic_ns())
        if reason:
            raise RuntimeError(f"unsafe zero-position handoff: {reason}")
        initial_q = np.asarray(state.q, dtype=np.float64).copy()
        active_q = initial_q[ACTIVE_SLOTS]
        lower = self.config.joint_lower_rad[ACTIVE_SLOTS]
        upper = self.config.joint_upper_rad[ACTIVE_SLOTS]
        outside = ACTIVE_SLOTS[(active_q < lower) | (active_q > upper)]
        if outside.size:
            raise RuntimeError(
                f"cannot zero: initial feedback exceeds URDF limits in slots "
                f"{outside.tolist()}")
        trajectory = SequentialAnkleZeroTrajectory(
            initial_q, ANKLE_SLOTS, self.config.zero_min_duration_s,
            self.config.zero_max_target_speed_rad_s)
        zero_target_q = initial_q.copy()
        zero_target_q[list(ANKLE_SLOTS)] = 0.0
        with self.lock:
            self.control_strategy = ZERO_STRATEGY
            self.phase = Phase.GAIN_RAMP
            self.phase_started_ns = time.monotonic_ns()
            self.zero_origin_q = initial_q
            self.zero_target_q = zero_target_q
            self.prepare_target_q = initial_q.copy()
            self.zeroing_slot = None
            self.zero_slot_index = -1
            self.zero_confirmed = False
            self.zero_duration_s = trajectory.duration_s
            self.trajectory = None
            self.axis = ""
            self.slot = None
            self.current_offset = 0.0
            self.current_velocity = 0.0
            self.zero_progress = 0.0
            self.segment = "gain-ramp-hold"
            self.gain_scale = 0.0
            self.gain_settle_started_ns = None
        self.recorder.event(
            "ankle_sequential_zero_position_armed", mode_pr=MODE_PR,
            mode_machine=MODE_MACHINE, zero_duration_s=trajectory.duration_s,
            zero_max_target_speed_rad_s=self.config.zero_max_target_speed_rad_s,
            zero_slots=np.asarray(ANKLE_SLOTS, dtype=np.int64),
            zero_slot_durations_s=np.asarray(trajectory.slot_durations_s),
            gain_ramp_s=self.config.gain_ramp_s,
            gain_settle_s=self.config.gain_settle_s,
            gain_settle_timeout_s=self.config.gain_settle_timeout_s,
            initial_q=initial_q[ACTIVE_SLOTS], position_kp=POSITION_KP[ACTIVE_SLOTS],
            position_kd=POSITION_KD[ACTIVE_SLOTS])
        q, dq, _progress, _done, _slot = trajectory.sample(0.0)
        if not self._write(self._build_position_command(q, dq, 0.0), state):
            self.fault("first zero-position LowCmd write failed")
        self._ensure_writer()

    def _ensure_writer(self):
        with self.lock:
            if self.writer is not None:
                return
            self.writer = PeriodicThread(0.002, self._tick)
            self.writer.start()

    def _command_from_arrays(self, q, dq, tau, kp, kd):
        command = new_lowcmd()
        command.mode_pr = MODE_PR
        command.mode_machine = MODE_MACHINE
        for index, motor in enumerate(command.motor_cmd):
            motor.mode = int(self.baseline.motor_mode[index])
            motor.q = float(q[index])
            motor.dq = float(dq[index])
            motor.tau = float(tau[index])
            motor.kp = float(kp[index])
            motor.kd = float(kd[index])
            motor.reserve = int(self.baseline.motor_reserve[index])
        command.reserve = np.asarray(self.baseline.reserve, dtype=np.uint32).tolist()
        command.crc = self.transport.crc.Crc(command)
        self.last_target_q = np.asarray(q, dtype=np.float64).copy()
        return command

    def _build_position_command(self, q, dq, gain_scale=1.0):
        scale = float(np.clip(gain_scale, 0.0, 1.0))
        self.gain_scale = scale
        return self._command_from_arrays(
            q, dq, np.zeros(WIRE_MOTOR_COUNT, dtype=np.float64),
            POSITION_KP * scale, POSITION_KD * scale)

    def _build_captured_hold_command(self, gain_scale):
        return self._build_position_command(
            self.zero_origin_q, np.zeros(WIRE_MOTOR_COUNT), gain_scale)

    def _build_zero_hold_command(self):
        if self.zero_target_q is None:
            raise RuntimeError("ankle-zero target is unavailable")
        q = self.zero_target_q.copy()
        dq = np.zeros(WIRE_MOTOR_COUNT, dtype=np.float64)
        return self._build_position_command(q, dq, 1.0)

    def _build_preparation_hold_command(self):
        return self._build_position_command(self.prepare_target_q, np.zeros(WIRE_MOTOR_COUNT))

    def _start_zero_slot_locked(self, index, now_ns):
        self.zero_slot_index = index
        self.zeroing_slot = ANKLE_SLOTS[index]
        self.trajectory = ZeroPostureTrajectory(
            self.prepare_target_q, [self.zeroing_slot], self.config.zero_min_duration_s,
            self.config.zero_max_target_speed_rad_s)
        self.phase = Phase.ZEROING
        self.phase_started_ns = now_ns
        self.current_velocity = 0.0
        self.segment = f"ankle-zero-{SLOT_AXES[self.zeroing_slot]}"
        self.recorder.event("ankle_zero_slot_started", slot=self.zeroing_slot,
                            axis=SLOT_AXES[self.zeroing_slot], duration_s=self.trajectory.duration_s)
        print(f"[prepare] moving only {SLOT_AXES[self.zeroing_slot]} "
              f"(slot {self.zeroing_slot}) to q=0", flush=True)

    def prepare_ankle_zero(self):
        now_ns = time.monotonic_ns()
        state = self.transport.latest()
        with self.lock:
            if self.control_strategy != ZERO_STRATEGY or self.phase != Phase.CAPTURED_HOLD:
                raise RuntimeError("prepare requires captured_hold after gain settling")
            reason = self._safety_reason(state, now_ns)
            if reason:
                raise RuntimeError(f"cannot prepare: {reason}")
            error = float(np.max(np.abs(state.q[ACTIVE_SLOTS] - self.zero_origin_q[ACTIVE_SLOTS])))
            speed = float(np.max(np.abs(state.dq[ACTIVE_SLOTS])))
            if (error > self.config.zero_position_tolerance_rad
                    or speed > self.config.zero_speed_tolerance_rad_s):
                raise RuntimeError("captured posture is not currently settled")
            self.zero_confirmed = False
            self.recorder.event("ankle_zero_sequence_started", slots=list(ANKLE_SLOTS),
                                planned_ramp_duration_s=self.zero_duration_s)
            self._start_zero_slot_locked(0, now_ns)

    def _build_hold_command(self):
        return self._command_from_arrays(self.baseline.q, self.baseline.dq,
                                         self.baseline.tau, self.baseline.kp,
                                         self.baseline.kd)

    def _build_axis_command(self, slot, offset, velocity):
        if self.control_strategy == ZERO_STRATEGY:
            if self.zero_target_q is None:
                raise RuntimeError("ankle-zero target is unavailable")
            q = self.zero_target_q.copy()
            dq = np.zeros(WIRE_MOTOR_COUNT, dtype=np.float64)
            q[slot] = float(offset)
            dq[slot] = float(velocity)
            return self._build_position_command(q, dq, 1.0)
        q, dq = overlay_axis(self.baseline.q, self.baseline.dq, slot, offset, velocity)
        return self._command_from_arrays(q, dq, self.baseline.tau,
                                         self.baseline.kp, self.baseline.kd)

    def _damping_arrays(self):
        q = self.baseline.q.copy()
        dq = np.zeros(WIRE_MOTOR_COUNT, dtype=np.float64)
        tau = np.zeros(WIRE_MOTOR_COUNT, dtype=np.float64)
        kp = np.zeros(WIRE_MOTOR_COUNT, dtype=np.float64)
        kd = np.zeros(WIRE_MOTOR_COUNT, dtype=np.float64)
        kd[:len(PASSIVE_KD)] = PASSIVE_KD
        kd[list(ANKLE_SLOTS)] = self.config.probe_ankle_kd
        return q, dq, tau, kp, kd

    def _build_damping_command(self):
        q, dq, tau, kp, kd = self._damping_arrays()
        return self._command_from_arrays(q, dq, tau, kp, kd)

    def _build_probe_command(self, slot, target_dq):
        if slot not in ANKLE_SLOTS:
            raise ValueError(f"slot {slot} is not an ankle probe slot")
        q, dq, tau, kp, kd = self._damping_arrays()
        dq[slot] = float(target_dq)
        return self._command_from_arrays(q, dq, tau, kp, kd)

    def _write(self, command, state):
        now_ns = time.monotonic_ns()
        try:
            success = self.transport.write(command)
        except Exception as exc:
            success = False
            self.recorder.event("lowcmd_write_exception", error=f"{type(exc).__name__}: {exc}")
        if success:
            self.last_write_ns = now_ns
            self.consecutive_write_failures = 0
        else:
            self.consecutive_write_failures += 1
        motor_q = np.asarray([m.q for m in command.motor_cmd], dtype=np.float32)
        motor_dq = np.asarray([m.dq for m in command.motor_cmd], dtype=np.float32)
        motor_tau = np.asarray([m.tau for m in command.motor_cmd], dtype=np.float32)
        motor_kp = np.asarray([m.kp for m in command.motor_cmd], dtype=np.float32)
        motor_kd = np.asarray([m.kd for m in command.motor_cmd], dtype=np.float32)
        pd_tau = (motor_tau + motor_kp * (motor_q - state.q[:WIRE_MOTOR_COUNT])
                  + motor_kd * (motor_dq - state.dq[:WIRE_MOTOR_COUNT]))
        with self.lock:
            phase = self.phase.value
            axis = self.axis
            offset = self.current_offset
            target_dq = self.current_velocity
            segment = self.segment
            cycle = self.cycle
            zero_progress = self.zero_progress
            gain_scale = self.gain_scale
            if phase != Phase.DAMPING.value:
                warning_events = self.speed_warnings.update(
                    state.monotonic_ns, phase, state.q, state.dq, motor_q, motor_dq)
            else:
                warning_events = []
        self._report_speed_warnings(warning_events)
        self.recorder.submit("command", dict(
            monotonic_ns=now_ns, wall_ns=time.time_ns(), state_sequence=state.sequence,
            state_age_ms=(now_ns - state.monotonic_ns) * 1e-6, phase=phase,
            axis=axis, segment=segment, cycle=cycle, offset_rad=offset,
            target_dq_rad_s=target_dq, zero_progress=zero_progress, gain_scale=gain_scale,
            mode_pr=int(command.mode_pr), mode_machine=int(command.mode_machine),
            motor_mode=np.asarray([m.mode for m in command.motor_cmd], dtype=np.uint8),
            motor_q=motor_q, motor_dq=motor_dq, motor_tau=motor_tau,
            motor_kp=motor_kp, motor_kd=motor_kd,
            pd_tau_estimate=pd_tau.astype(np.float32),
            reserve=np.asarray(command.reserve, dtype=np.uint32),
            message_crc=int(command.crc), write_success=int(success)))
        if self.control_count % 10 == 0:
            temp = np.asarray(state.temperature)
            self.recorder.submit("analysis", dict(
                monotonic_ns=now_ns, phase=phase, axis=axis, segment=segment,
                cycle=cycle, offset_cmd_rad=offset, target_dq_rad_s=target_dq,
                zero_progress=zero_progress, gain_scale=gain_scale,
                q4=float(state.q[4]), dq4=float(state.dq[4]), tau4=float(state.tau_est[4]),
                q5=float(state.q[5]), dq5=float(state.dq[5]), tau5=float(state.tau_est[5]),
                q10=float(state.q[10]), dq10=float(state.dq[10]), tau10=float(state.tau_est[10]),
                q11=float(state.q[11]), dq11=float(state.dq[11]), tau11=float(state.tau_est[11]),
                temp4_0=int(temp[4, 0]), temp4_1=int(temp[4, 1]),
                temp5_0=int(temp[5, 0]), temp5_1=int(temp[5, 1]),
                temp10_0=int(temp[10, 0]), temp10_1=int(temp[10, 1]),
                temp11_0=int(temp[11, 0]), temp11_1=int(temp[11, 1])))
        return success

    def _report_speed_warnings(self, events):
        for kind, values in events:
            self.recorder.event(kind, **values)
            detail = (f"dq={values['dq_rad_s']:+.4f} rad/s" if kind.endswith("started")
                      else f"peak={values['peak_abs_dq_rad_s']:.4f} rad/s, "
                           f"duration={values['duration_s']:.3f} s ({values['end_reason']})")
            print(f"[speed-warning] {kind} slot={values['slot']} "
                  f"phase={values['phase']} {detail}; diagnostic only", flush=True)

    def _safety_reason(self, state, now_ns):
        if self.recorder.failed.is_set():
            return f"recorder failed: {self.recorder.failure_reason}"
        if state is None:
            return "no valid LowState"
        if now_ns - state.monotonic_ns > self.config.state_stale_ns:
            return f"LowState stale for {(now_ns-state.monotonic_ns)*1e-6:.1f} ms"
        if (state.mode_machine, state.mode_pr) != (MODE_MACHINE, MODE_PR):
            return f"mode changed to {state.mode_machine}/{state.mode_pr}"
        arrays = (state.q, state.dq, state.ddq, state.tau_est)
        if not all(np.isfinite(item).all() for item in arrays):
            return "non-finite motor feedback"
        errors = ACTIVE_SLOTS[np.asarray(state.motorstate)[ACTIVE_SLOTS] != 0]
        if errors.size:
            return f"motorstate errors in slots {errors.tolist()}"
        maximum_temperature = float(np.max(np.asarray(state.temperature)[ACTIVE_SLOTS]))
        if maximum_temperature > self.config.max_temperature:
            return (f"temperature {maximum_temperature:.1f} exceeds "
                    f"{self.config.max_temperature:.1f}")
        active_q = np.asarray(state.q)[ACTIVE_SLOTS]
        lower = self.config.joint_lower_rad[ACTIVE_SLOTS]
        upper = self.config.joint_upper_rad[ACTIVE_SLOTS]
        outside = ACTIVE_SLOTS[(active_q < lower) | (active_q > upper)]
        if outside.size:
            return f"joint feedback exceeds URDF limits in slots {outside.tolist()}"
        with self.lock:
            probing = self.phase == Phase.PROBING
            position_active = self.phase in (
                Phase.GAIN_RAMP, Phase.GAIN_SETTLING, Phase.CAPTURED_HOLD,
                Phase.ZEROING, Phase.ZERO_SETTLING, Phase.ZERO_HOLD,
                Phase.RUNNING, Phase.RETURNING)
            debug_control = (self.control_strategy == "debug_velocity_probe"
                             and self.phase in (Phase.DEBUG_DAMPING, Phase.PROBING))
            probe_origin_q = (None if self.probe_origin_q is None
                              else self.probe_origin_q.copy())
            debug_origin_q = (None if self.debug_origin_q is None
                              else self.debug_origin_q.copy())
            last_target_q = (None if self.last_target_q is None
                             else self.last_target_q.copy())
        if position_active and last_target_q is not None:
            tracking_error = np.abs(active_q - last_target_q[ACTIVE_SLOTS])
            worst = float(np.max(tracking_error))
            if worst > self.config.max_tracking_error_rad:
                slot = int(ACTIVE_SLOTS[int(np.argmax(tracking_error))])
                return (f"position tracking error {worst:.4f} rad in slot {slot} exceeds "
                        f"{self.config.max_tracking_error_rad:.4f} rad")
        if debug_control and debug_origin_q is not None:
            displacement = np.abs(np.asarray(state.q)[list(ANKLE_SLOTS)]
                                  - debug_origin_q[list(ANKLE_SLOTS)])
            maximum_displacement = float(np.max(displacement))
            if maximum_displacement > self.config.probe_max_displacement_rad:
                return (f"ankle drift since debug handoff {maximum_displacement:.4f} "
                        f"exceeds {self.config.probe_max_displacement_rad:.4f} rad")
        if probing and probe_origin_q is not None:
            displacement = np.abs(np.asarray(state.q)[list(ANKLE_SLOTS)]
                                  - probe_origin_q[list(ANKLE_SLOTS)])
            maximum_displacement = float(np.max(displacement))
            if maximum_displacement > self.config.probe_max_displacement_rad:
                return (f"ankle feedback displacement {maximum_displacement:.4f} exceeds "
                        f"{self.config.probe_max_displacement_rad:.4f} rad")
        if remote_lt_b(state.wireless_remote):
            return "remote LT+B"
        if self.consecutive_write_failures >= 3:
            return "three consecutive LowCmd write failures"
        if self.last_write_ns is not None and now_ns - self.last_write_ns > 50_000_000:
            return "no successful LowCmd write for 50 ms"
        return ""

    def _tick(self):
        now_ns = time.monotonic_ns()
        state = self.transport.latest()
        with self.lock:
            phase = self.phase
        if phase == Phase.CLOSED:
            return
        if phase != Phase.DAMPING:
            reason = self._safety_reason(state, now_ns)
            if reason:
                self.fault(reason)
                phase = Phase.DAMPING
        if state is None:
            self.transport.disable_commands()
            self.exit_requested.set()
            return
        with self.lock:
            phase = self.phase
            elapsed = (now_ns - self.phase_started_ns) * 1e-9
            if phase == Phase.GAIN_RAMP:
                scale = smoothstep5(elapsed / self.config.gain_ramp_s)
                command = self._build_captured_hold_command(scale)
                if elapsed >= self.config.gain_ramp_s:
                    self.phase = Phase.GAIN_SETTLING
                    self.phase_started_ns = now_ns
                    self.gain_settle_started_ns = None
                    self.segment = "gain-settling-hold"
                    self.recorder.event("gain_ramp_complete", kp=40.0, kd=1.0)
                    print("[gain] Kp=40/Kd=1 reached; checking captured-posture settling",
                          flush=True)
            elif phase == Phase.GAIN_SETTLING:
                command = self._build_captured_hold_command(1.0)
                errors = np.abs(np.asarray(state.q)[ACTIVE_SLOTS]
                                - self.zero_origin_q[ACTIVE_SLOTS])
                hold_error = float(np.max(errors))
                hold_speed = float(np.max(np.abs(np.asarray(state.dq)[ACTIVE_SLOTS])))
                settled = (hold_error <= self.config.zero_position_tolerance_rad
                           and hold_speed <= self.config.zero_speed_tolerance_rad_s)
                if settled:
                    if self.gain_settle_started_ns is None:
                        self.gain_settle_started_ns = now_ns
                else:
                    self.gain_settle_started_ns = None
                # A missed deadline must never unlock motion, even if the latest
                # sample happens to be settled after a delayed control cycle.
                if elapsed >= self.config.gain_settle_timeout_s:
                    self.fault(
                        f"captured posture did not settle before zeroing: "
                        f"max error={hold_error:.4f} rad, max |dq|={hold_speed:.4f} rad/s")
                    command = self._build_damping_command()
                elif (self.gain_settle_started_ns is not None
                      and now_ns - self.gain_settle_started_ns
                      >= int(self.config.gain_settle_s * 1e9)):
                    self.phase = Phase.CAPTURED_HOLD
                    self.phase_started_ns = now_ns
                    self.segment = "captured-hold-waiting-prepare"
                    self.recorder.event("captured_posture_ready",
                                        max_hold_error_rad=hold_error,
                                        max_hold_speed_rad_s=hold_speed)
                    print("[hold] captured posture ready; type 'prepare ANKLE ZERO' "
                          "to start ankle preparation", flush=True)
            elif phase == Phase.CAPTURED_HOLD:
                command = self._build_captured_hold_command(1.0)
            elif phase == Phase.ZEROING:
                q, dq, blend, done = self.trajectory.sample(elapsed)
                self.zero_progress = (self.zero_slot_index + blend) / len(ANKLE_SLOTS)
                self.current_offset = 0.0
                self.current_velocity = float(np.max(np.abs(dq[list(ANKLE_SLOTS)])))
                if done:
                    self.prepare_target_q = q.copy()
                    self.phase = Phase.ZERO_SETTLING
                    self.phase_started_ns = now_ns
                    self.zero_settle_started_ns = None
                    self.zero_settle_deadline_ns = (
                        now_ns + int(self.config.zero_settle_timeout_s * 1e9))
                    self.current_offset = 0.0
                    self.current_velocity = 0.0
                    self.segment = f"ankle-settling-{SLOT_AXES[self.zeroing_slot]}"
                    self.recorder.event("ankle_zero_slot_target_reached", slot=self.zeroing_slot)
                    command = self._build_preparation_hold_command()
                else:
                    command = self._build_position_command(q, dq, 1.0)
            elif phase == Phase.ZERO_SETTLING:
                command = self._build_preparation_hold_command()
                checked_slots = list(ANKLE_SLOTS[:self.zero_slot_index + 1])
                zero_error = float(np.max(np.abs(np.asarray(state.q)[checked_slots])))
                zero_speed = float(np.max(np.abs(np.asarray(state.dq)[checked_slots])))
                settled = (zero_error <= self.config.zero_position_tolerance_rad
                           and zero_speed <= self.config.zero_speed_tolerance_rad_s)
                if settled:
                    if self.zero_settle_started_ns is None:
                        self.zero_settle_started_ns = now_ns
                else:
                    self.zero_settle_started_ns = None
                # A late good sample cannot bypass the per-slot deadline.
                if now_ns >= self.zero_settle_deadline_ns:
                    self.fault(
                        f"ankle slot {self.zeroing_slot} did not settle: "
                        f"max |q|={zero_error:.4f} rad, max |dq|={zero_speed:.4f} rad/s")
                    command = self._build_damping_command()
                elif (self.zero_settle_started_ns is not None
                      and now_ns - self.zero_settle_started_ns >= int(self.config.zero_settle_s * 1e9)):
                    self.recorder.event("ankle_zero_slot_completed", slot=self.zeroing_slot,
                                        max_abs_q_rad=zero_error, max_abs_dq_rad_s=zero_speed)
                    if self.zero_slot_index + 1 < len(ANKLE_SLOTS):
                        self._start_zero_slot_locked(self.zero_slot_index + 1, now_ns)
                    else:
                        self.phase = Phase.ZERO_HOLD
                        self.phase_started_ns = now_ns
                        self.current_offset = 0.0
                        self.current_velocity = 0.0
                        self.segment = "zero-hold-unconfirmed"
                        self.zero_progress = 1.0
                        self.recorder.event(
                            "ankle_zero_posture_reached", max_abs_q_rad=zero_error,
                            max_abs_dq_rad_s=zero_speed)
                        print("[zero] ankle posture reached; inspect robot, then type exactly "
                              "'confirm ZERO HOLD'", flush=True)
            elif phase == Phase.PROBING:
                sample = self.trajectory.sample(elapsed)
                self.current_offset = 0.0
                self.current_velocity = sample.target_dq_rad_s
                self.segment = sample.segment
                self.cycle = sample.cycle
                if sample.done:
                    self.recorder.event("velocity_probe_complete", axis=self.axis)
                    self.phase = Phase.DEBUG_DAMPING
                    self.phase_started_ns = now_ns
                    self.current_velocity = 0.0
                    self.segment = "debug-damping"
                    command = self._build_damping_command()
                else:
                    command = self._build_probe_command(self.slot, self.current_velocity)
            elif phase == Phase.RUNNING:
                sample = self.trajectory.sample(elapsed)
                self.current_offset = sample.offset_rad
                self.current_velocity = sample.velocity_rad_s
                self.segment = sample.segment
                self.cycle = sample.cycle
                if sample.done:
                    self.recorder.event("axis_complete", axis=self.axis)
                    self.phase = (Phase.ZERO_HOLD
                                  if self.control_strategy == ZERO_STRATEGY
                                  else Phase.HOLD)
                    self.phase_started_ns = now_ns
                    self.current_offset = 0.0
                    self.current_velocity = 0.0
                    self.segment = ""
                    command = (self._build_zero_hold_command()
                               if self.control_strategy == ZERO_STRATEGY
                               else self._build_hold_command())
                else:
                    command = self._build_axis_command(
                        self.slot, self.current_offset, self.current_velocity)
            elif phase == Phase.RETURNING:
                sample = self.trajectory.sample(elapsed)
                self.current_offset = sample.offset_rad
                self.current_velocity = sample.velocity_rad_s
                self.segment = sample.segment
                if sample.done:
                    self.current_offset = 0.0
                    self.current_velocity = 0.0
                    if self.close_after_return:
                        self._enter_damping_locked("operator stop after return", now_ns)
                        command = self._build_damping_command()
                    else:
                        self.phase = (Phase.ZERO_HOLD
                                      if self.control_strategy == ZERO_STRATEGY
                                      else Phase.HOLD)
                        self.phase_started_ns = now_ns
                        command = (self._build_zero_hold_command()
                                   if self.control_strategy == ZERO_STRATEGY
                                   else self._build_hold_command())
                else:
                    command = self._build_axis_command(
                        self.slot, self.current_offset, self.current_velocity)
            elif phase in (Phase.DEBUG_DAMPING, Phase.DAMPING):
                command = self._build_damping_command()
            elif phase == Phase.ZERO_HOLD:
                command = self._build_zero_hold_command()
            else:
                command = self._build_hold_command()
        self.control_count += 1
        self._write(command, state)
        with self.lock:
            if self.phase == Phase.DAMPING and now_ns >= self.damping_until_ns:
                self.phase = Phase.CLOSED
                self.transport.disable_commands()
                self.exit_requested.set()
                if self.writer is not None:
                    self.writer.stop_event.set()

    def run_axis(self, axis):
        if axis not in AXIS_SLOTS:
            raise ValueError(f"unknown axis {axis!r}; choose {', '.join(AXIS_SLOTS)}")
        with self.lock:
            allowed_strategy = self.control_strategy in ("position", ZERO_STRATEGY)
            if not allowed_strategy:
                raise RuntimeError("run requires a position-control handoff")
            required_phase = (Phase.ZERO_HOLD
                              if self.control_strategy == ZERO_STRATEGY
                              else Phase.HOLD)
            if self.phase != required_phase:
                raise RuntimeError(
                    f"run requires {required_phase.value}, current phase is {self.phase.value}")
            if (self.control_strategy == ZERO_STRATEGY
                    and not self.zero_confirmed):
                raise RuntimeError("inspect the ankle-zero posture and type "
                                   "'confirm ZERO HOLD' first")
            if self.control_strategy == ZERO_STRATEGY:
                reason = self._safety_reason(self.transport.latest(), time.monotonic_ns())
                if reason:
                    raise RuntimeError(f"cannot run: {reason}")
            amplitude = (self.config.pitch_amplitude_rad if axis.endswith("pitch")
                         else self.config.roll_amplitude_rad)
            baseline = (0.0 if self.control_strategy == ZERO_STRATEGY
                        else float(self.baseline.q[AXIS_SLOTS[axis]]))
            lower, upper = ANKLE_LIMITS_RAD[AXIS_SLOTS[axis]]
            margin = self.config.joint_limit_margin_rad
            if baseline - amplitude < lower + margin or baseline + amplitude > upper - margin:
                raise RuntimeError(
                    f"{axis} baseline {baseline:.4f} rad with ±{amplitude:.4f} rad "
                    f"would violate the URDF limit margin")
            self.axis = axis
            self.slot = AXIS_SLOTS[axis]
            self.trajectory = AxisTrajectory(amplitude, self.config.ramp_s,
                                             self.config.hold_s, self.config.cycles)
            self.phase = Phase.RUNNING
            self.phase_started_ns = time.monotonic_ns()
            self.segment = "starting"
            self.cycle = 1
        self.recorder.event("axis_started", axis=axis, slot=self.slot,
                            amplitude_rad=amplitude, cycles=self.config.cycles)

    def confirm_zero_hold(self):
        state = self.transport.latest()
        if state is None:
            raise RuntimeError("cannot confirm without LowState")
        with self.lock:
            if self.control_strategy != ZERO_STRATEGY:
                raise RuntimeError("zero-hold confirmation is unavailable for this handoff")
            if self.phase != Phase.ZERO_HOLD:
                raise RuntimeError(
                    f"confirmation requires zero_hold, current phase is {self.phase.value}")
            reason = self._safety_reason(state, time.monotonic_ns())
            if reason:
                raise RuntimeError(f"cannot confirm: {reason}")
            zero_error = float(np.max(np.abs(np.asarray(state.q)[list(ANKLE_SLOTS)])))
            zero_speed = float(np.max(np.abs(np.asarray(state.dq)[list(ANKLE_SLOTS)])))
            if zero_error > self.config.zero_position_tolerance_rad:
                raise RuntimeError(
                    f"ankle-zero error {zero_error:.4f} rad exceeds tolerance")
            if zero_speed > self.config.zero_speed_tolerance_rad_s:
                raise RuntimeError(f"robot is still moving: max |dq|={zero_speed:.4f} rad/s")
            self.zero_confirmed = True
            self.segment = "zero-hold-confirmed"
        self.recorder.event("zero_hold_operator_confirmed",
                            max_abs_q_rad=zero_error, max_abs_dq_rad_s=zero_speed)

    def observe_motion(self, axis, motion):
        if axis not in AXIS_SLOTS or motion not in ("pitch", "roll", "mixed", "unclear"):
            raise ValueError("observe <axis> pitch|roll|mixed|unclear")
        with self.lock:
            phase, segment = self.phase.value, self.segment
        self.recorder.event("operator_observation", axis=axis, motion=motion,
                            phase=phase, segment=segment)

    def probe_axis(self, axis):
        if axis not in AXIS_SLOTS:
            raise ValueError(f"unknown axis {axis!r}; choose {', '.join(AXIS_SLOTS)}")
        state = self.transport.latest()
        if state is None:
            raise RuntimeError("cannot probe without LowState")
        now_ns = time.monotonic_ns()
        with self.lock:
            if self.control_strategy != "debug_velocity_probe":
                raise RuntimeError("probe is only available for zero-command debug handoff")
            if self.phase != Phase.DEBUG_DAMPING:
                raise RuntimeError(
                    f"probe requires DEBUG_DAMPING, current phase is {self.phase.value}")
            elapsed = (now_ns - self.phase_started_ns) * 1e-9
            if elapsed < self.config.debug_settle_s:
                raise RuntimeError(
                    f"wait {self.config.debug_settle_s - elapsed:.1f} s for debug damping")
            observed_speed = float(np.max(np.abs(np.asarray(state.dq)[ACTIVE_SLOTS])))
            if observed_speed > self.config.max_capture_dq:
                raise RuntimeError(
                    f"robot is not settled: max |dq|={observed_speed:.3f} rad/s")
            self.axis = axis
            self.slot = AXIS_SLOTS[axis]
            self.trajectory = VelocityProbeTrajectory(
                self.config.probe_speed_rad_s, self.config.probe_ramp_s,
                self.config.probe_hold_s, self.config.probe_cycles)
            self.probe_origin_q = np.asarray(state.q, dtype=np.float64).copy()
            self.phase = Phase.PROBING
            self.phase_started_ns = now_ns
            self.current_offset = 0.0
            self.current_velocity = 0.0
            self.segment = "starting"
            self.cycle = 1
        self.recorder.event(
            "velocity_probe_started", axis=axis, slot=self.slot,
            speed_rad_s=self.config.probe_speed_rad_s,
            ankle_kd=self.config.probe_ankle_kd,
            max_displacement_rad=self.config.probe_max_displacement_rad,
            q_origin=self.probe_origin_q[list(ANKLE_SLOTS)])

    def request_return(self, close_after=False, reason="operator return"):
        with self.lock:
            if self.phase == Phase.OBSERVE:
                if close_after:
                    self.phase = Phase.CLOSED
                    self.exit_requested.set()
                return
            if self.phase in (Phase.DAMPING, Phase.CLOSED):
                return
            if self.control_strategy == "debug_velocity_probe":
                self.current_offset = 0.0
                self.current_velocity = 0.0
                self.segment = "debug-damping"
                if close_after:
                    self._enter_damping_locked(reason, time.monotonic_ns())
                else:
                    self.phase = Phase.DEBUG_DAMPING
                    self.phase_started_ns = time.monotonic_ns()
                    self.recorder.event("velocity_probe_cancelled", reason=reason)
                return
            if self.control_strategy == ZERO_STRATEGY and self.phase in (
                    Phase.GAIN_RAMP, Phase.GAIN_SETTLING, Phase.ZEROING, Phase.ZERO_SETTLING):
                if close_after:
                    self._enter_damping_locked(reason, time.monotonic_ns())
                else:
                    raise RuntimeError("gain acquisition/zeroing cannot be reversed; use stop or exit")
                return
            if self.phase == Phase.CAPTURED_HOLD:
                if close_after:
                    self._enter_damping_locked(reason, time.monotonic_ns())
                return  # 'return' must not bypass preparation/confirmation.
            if abs(self.current_offset) < 1e-9:
                if close_after:
                    self._enter_damping_locked(reason, time.monotonic_ns())
                else:
                    self.phase = (Phase.ZERO_HOLD
                                  if self.control_strategy == ZERO_STRATEGY
                                  else Phase.HOLD)
                return
            self.trajectory = ReturnTrajectory(self.current_offset, self.config.ramp_s)
            self.phase = Phase.RETURNING
            self.phase_started_ns = time.monotonic_ns()
            self.close_after_return = bool(close_after)
            self.segment = "return"
        self.recorder.event("return_requested", reason=reason, close_after=close_after)

    def _enter_damping_locked(self, reason, now_ns):
        self._report_speed_warnings(self.speed_warnings.finish(now_ns, "protective_damping"))
        self.gain_scale = 0.0
        self.phase = Phase.DAMPING
        self.phase_started_ns = now_ns
        self.damping_until_ns = now_ns + int(self.config.damping_s * 1e9)
        self.fault_reason = self.fault_reason or str(reason)
        self.segment = "damping"
        self.recorder.event("protective_damping", reason=str(reason),
                            duration_s=self.config.damping_s)

    def fault(self, reason):
        with self.lock:
            if self.phase in (Phase.DAMPING, Phase.CLOSED):
                return
            self.fault_reason = str(reason)
            self._enter_damping_locked(reason, time.monotonic_ns())
        print(f"[FAULT] {reason}; entering bounded protective damping", flush=True)

    def status(self):
        state = self.transport.latest()
        with self.lock:
            summary = (f"phase={self.phase.value} axis={self.axis or '-'} "
                       f"strategy={self.control_strategy} "
                       f"segment={self.segment or '-'} "
                       f"offset={math.degrees(self.current_offset):+.3f} deg "
                       f"dq_cmd={math.degrees(self.current_velocity):+.3f} deg/s "
                       f"zero_progress={100.0*self.zero_progress:.1f}% "
                       f"gain_scale={self.gain_scale:.3f} "
                       f"zero_confirmed={self.zero_confirmed}")
        if state is None:
            return summary + " LowState=missing"
        temperature = np.asarray(state.temperature)
        fields = []
        for slot in ANKLE_SLOTS:
            fields.append(f"{slot}:q={state.q[slot]:+.4f},dq={state.dq[slot]:+.4f},"
                          f"tau={state.tau_est[slot]:+.2f},T={temperature[slot].tolist()}")
        return (summary + f" mode={state.mode_machine}/{state.mode_pr} "
                + " | ".join(fields))

    def close(self):
        if self.transport.publisher is not None and self.phase != Phase.CLOSED:
            self.fault("process cleanup")
            self.exit_requested.wait(self.config.damping_s + 1.0)
        if self.writer is not None:
            self.writer.stop()
        self._report_speed_warnings(self.speed_warnings.finish(time.monotonic_ns()))
        self.transport.disable_commands()


def input_worker(commands, stop_event):
    while not stop_event.is_set():
        try:
            commands.put(input())
        except EOFError:
            return


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", required=True, help="physical Ethernet interface")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--enable-lowcmd", action="store_true",
                        help="allow the explicit console arm command to release MotionSwitcher")
    parser.add_argument("--observe-duration", type=float, default=30.0,
                        help="read-only duration; ignored after low-level arming")
    parser.add_argument("--pitch-amplitude-deg", type=float, default=5.0)
    parser.add_argument("--roll-amplitude-deg", type=float, default=3.0)
    parser.add_argument("--ramp-duration", type=float, default=3.0)
    parser.add_argument("--hold-duration", type=float, default=2.0)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--state-stale-ms", type=float, default=50.0)
    parser.add_argument("--max-temperature", type=float, default=75.0)
    parser.add_argument("--speed-warning-rad-s", type=float, default=0.25,
                        help="feedback speed diagnostic only; does not stop motion")
    parser.add_argument("--max-abs-dq", help=argparse.SUPPRESS)
    parser.add_argument("--damping-duration", type=float, default=1.0)
    parser.add_argument("--capture-window", type=float, default=2.0)
    parser.add_argument("--max-capture-dq", type=float, default=0.2)
    parser.add_argument("--max-command-span-deg", type=float, default=5.0)
    parser.add_argument("--min-ankle-kp", type=float, default=1.0)
    parser.add_argument("--joint-limit-margin-deg", type=float, default=2.0)
    parser.add_argument("--zero-min-duration", type=float, default=8.0,
                        help="minimum duration for each sequential ankle-zero segment")
    parser.add_argument("--gain-ramp-duration", type=float, default=3.0,
                        help="ramp gains while holding the captured posture; at least 3 seconds")
    parser.add_argument("--gain-settle-duration", type=float, default=1.0)
    parser.add_argument("--gain-settle-timeout", type=float, default=5.0)
    parser.add_argument("--zero-max-target-speed-deg-s", type=float, default=3.0,
                        help="automatically lengthen zeroing to respect this target speed")
    parser.add_argument("--zero-settle-duration", type=float, default=1.0)
    parser.add_argument("--zero-settle-timeout", type=float, default=5.0)
    parser.add_argument("--zero-position-tolerance-deg", type=float, default=2.0)
    parser.add_argument("--zero-speed-tolerance-deg-s", type=float, default=5.0)
    parser.add_argument("--max-tracking-error-deg", type=float, default=8.0)
    args = parser.parse_args(argv)
    if args.max_abs_dq is not None:
        parser.error("--max-abs-dq is retired; use --speed-warning-rad-s for "
                     "diagnostics only (feedback speed no longer aborts motion)")
    del args.max_abs_dq
    finite_positive = (args.observe_duration, args.pitch_amplitude_deg,
                       args.roll_amplitude_deg, args.ramp_duration,
                       args.state_stale_ms, args.max_temperature, args.speed_warning_rad_s,
                       args.damping_duration, args.capture_window,
                       args.max_capture_dq, args.max_command_span_deg, args.min_ankle_kp)
    finite_positive += (args.joint_limit_margin_deg, args.zero_min_duration,
                        args.zero_max_target_speed_deg_s,
                        args.zero_settle_duration, args.zero_settle_timeout,
                        args.zero_position_tolerance_deg,
                        args.zero_speed_tolerance_deg_s, args.max_tracking_error_deg,
                        args.gain_ramp_duration, args.gain_settle_duration,
                        args.gain_settle_timeout)
    if not all(math.isfinite(item) and item > 0 for item in finite_positive):
        parser.error("durations, limits, gains, and amplitudes must be positive and finite")
    if not math.isfinite(args.hold_duration) or args.hold_duration < 0:
        parser.error("--hold-duration must be finite and nonnegative")
    if args.cycles < 1 or args.cycles > 3:
        parser.error("--cycles must be in [1,3]")
    if args.pitch_amplitude_deg > 10.0 or args.roll_amplitude_deg > 5.0:
        parser.error("this entry caps Pitch at 10 degrees and Roll at 5 degrees")
    if args.ramp_duration < 1.0:
        parser.error("--ramp-duration must be at least 1 second")
    if args.zero_min_duration < 6.0:
        parser.error("--zero-min-duration must be at least 6 seconds")
    if args.gain_ramp_duration < 3.0:
        parser.error("--gain-ramp-duration must be at least 3 seconds")
    if args.gain_settle_duration < 1.0:
        parser.error("--gain-settle-duration must be at least 1 second")
    if args.gain_settle_timeout <= args.gain_settle_duration:
        parser.error("--gain-settle-timeout must exceed --gain-settle-duration")
    if args.zero_settle_timeout <= args.zero_settle_duration:
        parser.error("--zero-settle-timeout must exceed --zero-settle-duration")
    if args.zero_max_target_speed_deg_s > 10.0:
        parser.error("--zero-max-target-speed-deg-s is capped at 10 deg/s")
    if args.zero_position_tolerance_deg > 3.0:
        parser.error("--zero-position-tolerance-deg is capped at 3 degrees")
    if args.max_tracking_error_deg > 10.0:
        parser.error("--max-tracking-error-deg is capped at 10 degrees")
    if args.enable_lowcmd and args.network == "lo":
        parser.error("real LowCmd cannot use the loopback interface")
    return args


def make_config(args):
    joint_lower, joint_upper = load_active_joint_limits(args.urdf)
    return TestConfig(
        math.radians(args.pitch_amplitude_deg), math.radians(args.roll_amplitude_deg),
        args.ramp_duration, args.hold_duration, args.cycles,
        int(args.state_stale_ms * 1e6), args.max_temperature, args.speed_warning_rad_s,
        args.damping_duration, args.capture_window, args.max_capture_dq,
        math.radians(args.max_command_span_deg), args.min_ankle_kp,
        math.radians(args.joint_limit_margin_deg),
        math.radians(2.0), 1.0, 0.5, 1, 3.0, math.radians(5.0), 3.0,
        args.zero_min_duration, math.radians(args.zero_max_target_speed_deg_s),
        args.zero_settle_duration, args.zero_settle_timeout,
        math.radians(args.zero_position_tolerance_deg),
        math.radians(args.zero_speed_tolerance_deg_s),
        math.radians(args.max_tracking_error_deg), joint_lower, joint_upper,
        args.gain_ramp_duration, args.gain_settle_duration, args.gain_settle_timeout)


def main(argv=None):
    args = parse_args(argv)
    config = make_config(args)
    metadata = {
        "experiment": "G1 suspended SDK ankle PR feedback semantics",
        "real_lowcmd_enabled": bool(args.enable_lowcmd),
        "network": args.network,
        "mode_pr_fixed": MODE_PR,
        "mode_machine_required": MODE_MACHINE,
        "active_slots": ACTIVE_SLOTS,
        "ankle_slots": {name: slot for name, slot in AXIS_SLOTS.items()},
        "workflow_version": 2,
        "feedback_speed_policy": "warning_only",
        "evidence_scope": "qualitative q/dq; no automatic PR/AB verdict or torque calibration",
        "arguments": vars(args),
    }
    recorder = Recorder(args.output_dir, args.run_name, metadata)
    transport = None
    controller = None
    handoff_kind = None
    arm_command = None
    outcome = "error"
    command_queue = queue.Queue()
    input_stop = threading.Event()
    try:
        transport = HardwareTransport(args.network, recorder)
        print(f"[preflight] waiting on {args.network}; "
              f"lowcmd={'allowed after explicit arm' if args.enable_lowcmd else 'read-only'}",
              flush=True)
        preflight = transport.wait_preflight(ACTIVE_SLOTS,
                                             max_temperature=args.max_temperature)
        recorder.event("preflight_passed", **preflight)
        print(f"[preflight] passed: {preflight}", flush=True)
        service = transport.check_motion_mode()
        service_name = service.get("name", "")
        controller = AnkleController(transport, recorder, config)
        if args.enable_lowcmd:
            if not service_name:
                raise RuntimeError("no official MotionSwitcher service owns the robot")
            # Validate once for operator visibility; arm validates a fresh window again.
            try:
                baseline, diagnostics = select_zero_command_handoff(
                    *transport.histories(), ACTIVE_SLOTS,
                    window_s=config.capture_window_s,
                    max_state_dq=config.max_capture_dq)
                del baseline
                handoff_kind = ZERO_STRATEGY
                arm_command = ARM_COMMAND
                print(f"[capture] official service={service_name!r}, "
                      f"zero LowCmd verified: {diagnostics}", flush=True)
                print("[console] 'arm ARM PR HOLD' releases ai, ramps gains, and waits "
                      "in captured_hold; then 'prepare ANKLE ZERO' moves ankles only", flush=True)
                print(f"[speed-policy] |dq| > {config.speed_warning_rad_s:.3f} rad/s "
                      "is recorded as a warning only; no feedback-speed automatic abort", flush=True)
                print("[console] after '[zero] ankle posture reached': status | "
                      "confirm ZERO HOLD | run left-pitch|left-roll|right-pitch|right-roll",
                      flush=True)
            except (RuntimeError, ValueError) as zero_error:
                raise RuntimeError(
                    "official zero-LowCmd validation failed; refusing the ankle-zero "
                    f"takeover: {zero_error}") from zero_error
        else:
            print("[console] read-only: status | mark <label> | exit", flush=True)
        print("[console] common commands: status | mark <label> | "
              "observe <axis> pitch|roll|mixed|unclear | return | stop | exit",
              flush=True)
        threading.Thread(target=input_worker, args=(command_queue, input_stop),
                         name="ankle-console", daemon=True).start()

        signal_count = 0

        def on_signal(signum, _frame):
            nonlocal signal_count
            signal_count += 1
            recorder.event("signal", signum=signum, count=signal_count)
            command_queue.put("stop" if signal_count == 1 else "exit")

        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)
        readonly_deadline = (None if args.enable_lowcmd else
                             time.monotonic() + args.observe_duration)
        while not controller.exit_requested.is_set():
            if readonly_deadline is not None and time.monotonic() >= readonly_deadline:
                outcome = "observe_complete"
                controller.request_return(close_after=True, reason="read-only duration complete")
                break
            if controller.writer is not None and controller.writer.exception is not None:
                raise RuntimeError(f"control thread failed: {controller.writer.exception}")
            try:
                line = command_queue.get(timeout=0.1).strip()
            except queue.Empty:
                continue
            if not line:
                continue
            try:
                if line == "status":
                    print(controller.status(), flush=True)
                elif line.startswith("mark "):
                    label = line[5:].strip()
                    if not label:
                        raise ValueError("mark requires a nonempty label")
                    recorder.event("operator_mark", label=label)
                    print(f"[mark] {label}", flush=True)
                elif line == ARM_COMMAND:
                    if not args.enable_lowcmd:
                        raise RuntimeError("restart with --enable-lowcmd to permit arming")
                    if line != arm_command:
                        raise RuntimeError(f"this capture requires exact command {arm_command!r}")
                    controller.arm(service_name, handoff_kind)
                    print(f"[armed] fixed mode_pr=0; strategy={handoff_kind}; "
                          f"phase={controller.phase.value}", flush=True)
                    if handoff_kind == ZERO_STRATEGY:
                        print(f"[gain] ramp={config.gain_ramp_s:.2f} s, "
                              f"stable hold required={config.gain_settle_s:.2f} s; "
                              "then captured_hold waiting for prepare; "
                              "stop/exit/LT+B aborts to protective damping", flush=True)
                elif line.startswith("arm "):
                    raise RuntimeError(f"old/unknown arm command; use '{ARM_COMMAND}', "
                                       "then 'prepare ANKLE ZERO' after captured_hold")
                elif line == "prepare ANKLE ZERO":
                    controller.prepare_ankle_zero()
                    print("[prepare] sequential ankle preparation started", flush=True)
                elif line.startswith("observe "):
                    parts = line.split()
                    if len(parts) != 3:
                        raise ValueError("observe <axis> pitch|roll|mixed|unclear")
                    controller.observe_motion(parts[1], parts[2])
                    print(f"[observe] {parts[1]} {parts[2]}", flush=True)
                elif line == "confirm ZERO HOLD":
                    controller.confirm_zero_hold()
                    print("[confirmed] zero hold; single-axis PR run commands enabled",
                          flush=True)
                elif line.startswith("run "):
                    controller.run_axis(line[4:].strip())
                    print(f"[run] {line[4:].strip()}", flush=True)
                elif line.startswith("probe "):
                    raise RuntimeError("velocity probe is retired; use the ankle-zero path")
                elif line == "return":
                    controller.request_return(close_after=False)
                elif line == "stop":
                    controller.request_return(close_after=True, reason="operator stop")
                    outcome = "operator_stop"
                elif line == "exit":
                    controller.request_return(close_after=True, reason="operator exit")
                    outcome = "operator_exit"
                else:
                    print(f"[console] unknown or incomplete command: {line!r}", flush=True)
            except Exception as exc:
                print(f"[console] rejected: {exc}", flush=True)
                recorder.event("console_command_rejected", command=line, error=str(exc))
        if controller.writer is not None:
            controller.exit_requested.wait(
                config.damping_s + max(config.ramp_s, config.probe_ramp_s) + 2.0)
        if outcome == "error" and controller.fault_reason:
            outcome = "fault"
        elif outcome == "error" and controller.phase == Phase.CLOSED:
            outcome = "closed"
    except Exception as exc:
        if controller is not None:
            controller.fault(f"main exception: {type(exc).__name__}: {exc}")
        recorder.event("fatal_exception", error=f"{type(exc).__name__}: {exc}")
        print(f"[fatal] {type(exc).__name__}: {exc}", flush=True)
        outcome = "error"
    finally:
        input_stop.set()
        if controller is not None:
            controller.close()
        if transport is not None:
            transport.close()
        extra = {}
        if controller is not None:
            extra = {"final_phase": controller.phase.value,
                     "fault_reason": controller.fault_reason,
                     "control_thread_exception": ("" if controller.writer is None
                         or controller.writer.exception is None
                         else str(controller.writer.exception))}
        recorder.close(outcome, extra)
        print(f"[done] outcome={outcome}; log={recorder.run_dir}", flush=True)
    return 0 if outcome in ("observe_complete", "operator_exit", "operator_stop", "closed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
