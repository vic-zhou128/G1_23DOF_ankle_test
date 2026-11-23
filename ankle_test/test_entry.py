"""Offline controller-message tests; no DDS participant is created."""
from __future__ import annotations

import math
from dataclasses import replace
import io
from contextlib import redirect_stderr
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from unitree_sdk2py.utils.crc import CRC

from .g1_ankle_pr_test import (ACTIVE_SLOTS, AnkleController, Phase, TestConfig,
                               ZERO_STRATEGY, parse_args, make_config)
from .core import ANKLE_SLOTS, ZeroPostureTrajectory
from .hardware_io import LowCmdSnapshot, StateSnapshot


class FakeRecorder:
    def __init__(self):
        self.events = []
        self.records = []
        self.failed = SimpleNamespace(is_set=lambda: False)
        self.failure_reason = ""

    def event(self, kind, **values):
        self.events.append((kind, values))

    def submit(self, stream, record):
        self.records.append((stream, record))
        return True


class FakeTransport:
    def __init__(self):
        self.crc = CRC()
        self.publisher = object()
        self.writes = []
        self.disabled = False
        self.state = make_state()

    def latest(self):
        return self.state

    def write(self, command):
        self.writes.append(command)
        return True

    def disable_commands(self):
        self.publisher = None
        self.disabled = True


def make_baseline():
    count = 35
    return LowCmdSnapshot(
        sequence=1, monotonic_ns=1, wall_ns=1, mode_pr=0, mode_machine=4,
        motor_mode=np.ones(count, dtype=np.uint8),
        q=np.linspace(-0.2, 0.2, count), dq=np.zeros(count),
        tau=np.linspace(-0.1, 0.1, count), kp=np.full(count, 20.0),
        kd=np.full(count, 1.0), motor_reserve=np.zeros(count, dtype=np.uint32),
        reserve=np.zeros(4, dtype=np.uint32), crc_valid=True)


def make_state():
    count = 35
    return StateSnapshot(
        sequence=1, monotonic_ns=1, wall_ns=1, tick=1, mode_pr=0, mode_machine=4,
        q=np.zeros(count), dq=np.zeros(count), ddq=np.zeros(count),
        tau_est=np.zeros(count), motor_mode=np.ones(count, dtype=np.uint8),
        motorstate=np.zeros(count, dtype=np.uint32),
        temperature=np.full((count, 2), 30, dtype=np.int16),
        wireless_remote=np.zeros(40, dtype=np.uint8))


def config():
    lower = np.full(35, -math.pi)
    upper = np.full(35, math.pi)
    return TestConfig(
        pitch_amplitude_rad=math.radians(3), roll_amplitude_rad=math.radians(2),
        ramp_s=2.0, hold_s=1.0, cycles=2, state_stale_ns=50_000_000,
        max_temperature=75.0, speed_warning_rad_s=0.25, damping_s=1.0,
        capture_window_s=2.0, max_capture_dq=0.2,
        max_command_span_rad=math.radians(5), min_ankle_kp=1.0,
        joint_limit_margin_rad=math.radians(2),
        probe_speed_rad_s=math.radians(2), probe_ramp_s=1.0,
        probe_hold_s=0.5, probe_cycles=1, probe_ankle_kd=3.0,
        probe_max_displacement_rad=math.radians(5), debug_settle_s=3.0,
        zero_min_duration_s=8.0,
        zero_max_target_speed_rad_s=math.radians(3), zero_settle_s=1.0,
        zero_settle_timeout_s=5.0,
        zero_position_tolerance_rad=math.radians(2),
        zero_speed_tolerance_rad_s=math.radians(1),
        max_tracking_error_rad=math.radians(8),
        joint_lower_rad=lower, joint_upper_rad=upper)


class MessageConstructionTests(unittest.TestCase):
    def setUp(self):
        self.transport = FakeTransport()
        self.recorder = FakeRecorder()
        self.controller = AnkleController(self.transport, self.recorder, config())
        self.controller.baseline = make_baseline()
        self.controller.control_strategy = "position"
        self.controller.phase = Phase.HOLD

    def test_hold_reproduces_all_35_slots_and_forces_pr(self):
        command = self.controller._build_hold_command()
        self.assertEqual(command.mode_pr, 0)
        self.assertEqual(command.mode_machine, 4)
        self.assertEqual(len(command.motor_cmd), 35)
        np.testing.assert_allclose([item.q for item in command.motor_cmd],
                                   self.controller.baseline.q)
        np.testing.assert_allclose([item.tau for item in command.motor_cmd],
                                   self.controller.baseline.tau)
        self.assertEqual(int(command.crc), int(self.transport.crc.Crc(command)))

    def test_axis_command_changes_only_selected_pr_slot(self):
        command = self.controller._build_axis_command(4, 0.03, 0.02)
        q = np.asarray([item.q for item in command.motor_cmd])
        dq = np.asarray([item.dq for item in command.motor_cmd])
        changed_q = np.flatnonzero(np.abs(q - self.controller.baseline.q) > 1e-7)
        changed_dq = np.flatnonzero(np.abs(dq - self.controller.baseline.dq) > 1e-7)
        np.testing.assert_array_equal(changed_q, [4])
        np.testing.assert_array_equal(changed_dq, [4])
        self.assertEqual(command.mode_pr, 0)

    def test_ankle_axis_command_holds_non_ankles_at_captured_targets(self):
        self.controller.control_strategy = ZERO_STRATEGY
        self.controller.phase = Phase.ZERO_HOLD
        self.controller.zero_confirmed = True
        captured = np.linspace(-0.3, 0.3, 35)
        target = captured.copy()
        target[list(ANKLE_SLOTS)] = 0.0
        self.controller.zero_origin_q = captured
        self.controller.zero_target_q = target
        command = self.controller._build_axis_command(4, 0.03, 0.02)
        q = np.asarray([item.q for item in command.motor_cmd])
        dq = np.asarray([item.dq for item in command.motor_cmd])
        kp = np.asarray([item.kp for item in command.motor_cmd])
        expected_q = target.copy()
        expected_q[4] = 0.03
        np.testing.assert_allclose(q, expected_q)
        np.testing.assert_array_equal(np.flatnonzero(np.abs(dq) > 1e-12), [4])
        np.testing.assert_allclose(kp[ACTIVE_SLOTS], 40.0)
        self.assertEqual(command.mode_pr, 0)

    def test_zero_position_run_requires_operator_confirmation(self):
        self.controller.control_strategy = ZERO_STRATEGY
        self.controller.phase = Phase.ZERO_HOLD
        self.controller.zero_confirmed = False
        with self.assertRaisesRegex(RuntimeError, "confirm ZERO HOLD"):
            self.controller.run_axis("left-pitch")

    def test_final_slot_reaches_settling_then_zero_hold(self):
        import time
        now = time.monotonic_ns()
        self.transport.state = StateSnapshot(
            **{**make_state().__dict__, "monotonic_ns": now})
        self.controller.control_strategy = ZERO_STRATEGY
        self.controller.phase = Phase.ZEROING
        initial = np.zeros(35)
        initial[11] = math.radians(-10)
        self.controller.zero_origin_q = initial.copy()
        self.controller.zero_target_q = initial.copy()
        self.controller.zero_target_q[list(ANKLE_SLOTS)] = 0.0
        self.controller.prepare_target_q = initial.copy()
        self.controller.zero_slot_index = 3
        self.controller.zeroing_slot = 11
        self.controller.trajectory = ZeroPostureTrajectory(
            initial, [11], 1.0, math.radians(20))
        self.controller.phase_started_ns = now - int(5e9)
        self.controller._tick()
        self.assertEqual(self.controller.phase, Phase.ZERO_SETTLING)
        self.controller.zero_settle_started_ns = time.monotonic_ns() - int(2e9)
        self.controller._tick()
        self.assertEqual(self.controller.phase, Phase.ZERO_HOLD)
        self.assertFalse(self.controller.zero_confirmed)

    def test_position_tracking_error_is_a_fault_reason(self):
        import time
        now = time.monotonic_ns()
        moved = make_state().q.copy()
        moved[4] = math.radians(9)
        state = StateSnapshot(**{**make_state().__dict__, "q": moved,
                                 "monotonic_ns": now})
        self.controller.control_strategy = ZERO_STRATEGY
        self.controller.phase = Phase.ZERO_HOLD
        self.controller.last_target_q = np.zeros(35)
        reason = self.controller._safety_reason(state, now)
        self.assertIn("tracking error", reason)

    def test_damping_has_zero_kp_tau_and_only_bounded_kd(self):
        command = self.controller._build_damping_command()
        np.testing.assert_allclose([item.kp for item in command.motor_cmd], 0.0)
        np.testing.assert_allclose([item.tau for item in command.motor_cmd], 0.0)
        kd = np.asarray([item.kd for item in command.motor_cmd])
        self.assertTrue(np.all(kd >= 0))
        self.assertTrue(np.all(kd[29:] == 0))
        self.assertEqual(command.mode_pr, 0)

    def test_debug_probe_has_zero_kp_tau_and_changes_only_selected_dq(self):
        self.controller.control_strategy = "debug_velocity_probe"
        command = self.controller._build_probe_command(4, math.radians(2))
        q = np.asarray([item.q for item in command.motor_cmd])
        dq = np.asarray([item.dq for item in command.motor_cmd])
        kp = np.asarray([item.kp for item in command.motor_cmd])
        tau = np.asarray([item.tau for item in command.motor_cmd])
        np.testing.assert_allclose(q, self.controller.baseline.q)
        np.testing.assert_allclose(kp, 0.0)
        np.testing.assert_allclose(tau, 0.0)
        np.testing.assert_array_equal(np.flatnonzero(np.abs(dq) > 1e-12), [4])
        self.assertAlmostEqual(dq[4], math.radians(2))

    def test_position_run_rejected_for_debug_probe_strategy(self):
        self.controller.control_strategy = "debug_velocity_probe"
        self.controller.phase = Phase.DEBUG_DAMPING
        with self.assertRaisesRegex(RuntimeError, "position-control handoff"):
            self.controller.run_axis("left-pitch")

    def test_probe_displacement_limit_is_a_fault_reason(self):
        state = make_state()
        origin = state.q.copy()
        moved = state.q.copy()
        moved[5] = math.radians(6)
        state = StateSnapshot(**{**state.__dict__, "q": moved,
                                 "monotonic_ns": __import__("time").monotonic_ns()})
        self.controller.control_strategy = "debug_velocity_probe"
        self.controller.phase = Phase.PROBING
        self.controller.probe_origin_q = origin
        reason = self.controller._safety_reason(state, state.monotonic_ns)
        self.assertIn("feedback displacement", reason)

    def test_debug_damping_drift_limit_is_a_fault_reason(self):
        state = make_state()
        origin = state.q.copy()
        moved = state.q.copy()
        moved[10] = math.radians(6)
        state = StateSnapshot(**{**state.__dict__, "q": moved,
                                 "monotonic_ns": __import__("time").monotonic_ns()})
        self.controller.control_strategy = "debug_velocity_probe"
        self.controller.phase = Phase.DEBUG_DAMPING
        self.controller.debug_origin_q = origin
        reason = self.controller._safety_reason(state, state.monotonic_ns)
        self.assertIn("drift since debug handoff", reason)

    def test_mode_change_is_a_fault(self):
        state = make_state()
        state = StateSnapshot(**{**state.__dict__, "mode_pr": 1,
                                 "monotonic_ns": __import__("time").monotonic_ns()})
        reason = self.controller._safety_reason(state, state.monotonic_ns)
        self.assertIn("mode changed", reason)

    def test_nonzero_active_motorstate_is_a_fault(self):
        state = make_state()
        errors = state.motorstate.copy()
        errors[int(ACTIVE_SLOTS[3])] = 7
        state = StateSnapshot(**{**state.__dict__, "motorstate": errors,
                                 "monotonic_ns": __import__("time").monotonic_ns()})
        reason = self.controller._safety_reason(state, state.monotonic_ns)
        self.assertIn("motorstate errors", reason)

    def test_axis_rejected_near_joint_limit(self):
        self.controller.baseline.q[5] = math.radians(14.0)
        with self.assertRaisesRegex(RuntimeError, "limit margin"):
            self.controller.run_axis("left-roll")


class GainAcquisitionTests(unittest.TestCase):
    """Exercise commands and transitions with a fake clock and transport only."""

    def setUp(self):
        self.transport = FakeTransport()
        self.recorder = FakeRecorder()
        self.controller = AnkleController(
            self.transport, self.recorder, config())
        self.epoch = 100_000_000_000
        self.q0 = np.zeros(35)
        self.q0[17] = 1.65
        self.q0[4] = -0.31
        self.transport.state = replace(make_state(), q=self.q0.copy(),
                                       monotonic_ns=self.epoch)
        with patch.object(self.controller, "_ensure_writer"), patch(
                "ankle_test.g1_ankle_pr_test.time.monotonic_ns", return_value=self.epoch):
            self.controller._released_zero_position_callback(make_baseline())

    def tick(self, seconds, q=None, dq=None, stale=False):
        now = self.epoch + round(seconds * 1e9)
        self.transport.state = replace(
            make_state(), q=self.q0.copy() if q is None else q.copy(),
            dq=np.zeros(35) if dq is None else dq.copy(),
            monotonic_ns=now - (100_000_000 if stale else 0))
        # These are selected control instants, with healthy writes between them.
        self.controller.last_write_ns = now - 2_000_000
        with patch("ankle_test.g1_ankle_pr_test.time.monotonic_ns", return_value=now):
            self.controller._tick()
        return self.transport.writes[-1]

    def test_gain_ramp_keeps_captured_targets_and_zero_velocity(self):
        self.assertEqual(self.controller.phase, Phase.GAIN_RAMP)
        for seconds, scale in ((0, 0), (1.5, 0.5), (3, 1)):
            command = self.tick(seconds)
            np.testing.assert_allclose([m.q for m in command.motor_cmd], self.q0)
            np.testing.assert_allclose([m.dq for m in command.motor_cmd], 0)
            np.testing.assert_allclose([m.tau for m in command.motor_cmd], 0)
            kp = np.array([m.kp for m in command.motor_cmd])
            kd = np.array([m.kd for m in command.motor_cmd])
            np.testing.assert_allclose(kp[ACTIVE_SLOTS], 40 * scale)
            np.testing.assert_allclose(kd[ACTIVE_SLOTS], scale)
            inactive = np.setdiff1d(np.arange(35), ACTIVE_SLOTS)
            np.testing.assert_allclose(kp[inactive], 0)
            np.testing.assert_allclose(kd[inactive], 0)
            self.assertEqual(command.mode_pr, 0)
            self.assertEqual(command.crc, self.transport.crc.Crc(command))
            self.assertEqual(self.controller.zero_progress, 0)
        self.assertEqual(self.controller.phase, Phase.GAIN_SETTLING)

    def test_gain_settling_waits_for_explicit_prepare_and_holds_captured_angles(self):
        self.tick(3)
        self.tick(3.002)
        self.tick(4.0)
        self.assertEqual(self.controller.phase, Phase.GAIN_SETTLING)
        self.tick(4.004)
        self.assertEqual(self.controller.phase, Phase.CAPTURED_HOLD)
        self.assertEqual(self.controller.phase_started_ns, self.epoch + 4_004_000_000)
        np.testing.assert_allclose(self.controller.zero_origin_q, self.q0)
        self.assertGreater(self.controller.zero_duration_s, 35)
        command = self.tick(5.004)
        np.testing.assert_allclose([m.q for m in command.motor_cmd], self.q0)
        self.assertIsNone(self.controller.trajectory)
        self.controller.request_return()
        self.assertEqual(self.controller.phase, Phase.CAPTURED_HOLD)
        self.tick(10)
        with patch('ankle_test.g1_ankle_pr_test.time.monotonic_ns', return_value=self.epoch + 10_000_000_000):
            self.controller.prepare_ankle_zero()
        command = self.tick(11)
        self.assertEqual(self.controller.phase, Phase.ZEROING)
        self.assertAlmostEqual(command.motor_cmd[17].q, self.q0[17])
        self.assertGreater(command.motor_cmd[4].q, self.q0[4])
        self.assertLess(command.motor_cmd[4].q, 0)
        for slot in (5, 10, 11):
            self.assertAlmostEqual(command.motor_cmd[slot].q, self.q0[slot])
        np.testing.assert_allclose(
            np.array([m.kp for m in command.motor_cmd])[ACTIVE_SLOTS], 40)
        np.testing.assert_allclose(
            np.array([m.kd for m in command.motor_cmd])[ACTIVE_SLOTS], 1)
        self.assertLessEqual(max(abs(m.dq) for m in command.motor_cmd), math.radians(3))
        self.assertFalse(self.controller.zero_confirmed)

    def test_motion_or_position_error_resets_stable_timer(self):
        for field in ("q", "dq"):
            with self.subTest(field=field):
                self.setUp()
                self.tick(3)
                self.tick(3.002)
                bad_q = self.q0.copy()
                bad_dq = np.zeros(35)
                if field == "q":
                    bad_q[17] += math.radians(3)
                else:
                    bad_dq[17] = math.radians(2)
                self.tick(3.8, q=bad_q, dq=bad_dq)
                self.assertIsNone(self.controller.gain_settle_started_ns)
                self.tick(4.0)
                self.tick(4.8)
                self.assertEqual(self.controller.phase, Phase.GAIN_SETTLING)
                self.tick(5.01)
                self.assertEqual(self.controller.phase, Phase.CAPTURED_HOLD)

    def test_settle_timeout_enters_damping_and_never_starts_zeroing(self):
        self.tick(3)
        drifting = self.q0.copy()
        drifting[17] += math.radians(3)
        self.tick(7.9, q=drifting)
        command = self.tick(8.0, q=drifting)
        self.assertEqual(self.controller.phase, Phase.DAMPING)
        self.assertIn("did not settle", self.controller.fault_reason)
        np.testing.assert_allclose([m.kp for m in command.motor_cmd], 0)
        self.assertNotIn("ankle_zero_sequence_started",
                         [kind for kind, _ in self.recorder.events])
        self.tick(9.01, q=drifting)
        self.assertTrue(self.transport.disabled)
        self.assertTrue(self.controller.exit_requested.is_set())

    def test_delayed_settled_sample_cannot_bypass_deadline(self):
        self.tick(3)
        self.tick(3.002)
        self.tick(8.0)
        self.assertEqual(self.controller.phase, Phase.DAMPING)

    def test_speed_warning_identifies_slot_and_does_not_abort(self):
        dq = np.zeros(35)
        dq[17] = -0.267
        self.tick(0.1, dq=dq)
        self.assertEqual(self.controller.phase, Phase.GAIN_RAMP)
        self.assertFalse(self.controller.fault_reason)
        self.tick(.102, dq=dq)
        self.tick(.104)
        warnings = [(kind, values) for kind, values in self.recorder.events if kind.startswith('speed_warning')]
        self.assertEqual([k for k, _ in warnings], ['speed_warning_started', 'speed_warning_ended'])
        self.assertEqual(warnings[0][1]['slot'], 17)
        self.assertEqual(warnings[-1][1]['peak_dq_rad_s'], -0.267)

    def test_tracking_and_stale_faults_are_active_during_gain_ramp(self):
        moved = self.q0.copy()
        moved[17] += math.radians(9)
        self.tick(0.1, q=moved)
        self.assertIn("tracking error", self.controller.fault_reason)
        self.setUp()
        self.tick(0.1, stale=True)
        self.assertIn("stale", self.controller.fault_reason)

    def test_stop_and_return_cannot_skip_gain_acquisition(self):
        for seconds in (1.0, 3.0):
            with self.subTest(seconds=seconds):
                self.setUp()
                self.tick(seconds)
                phase = self.controller.phase
                with self.assertRaises(RuntimeError):
                    self.controller.request_return()
                self.assertEqual(self.controller.phase, phase)
                with self.assertRaises(RuntimeError):
                    self.controller.run_axis("left-pitch")
                with self.assertRaises(RuntimeError):
                    self.controller.confirm_zero_hold()
                self.controller.request_return(close_after=True)
                self.assertEqual(self.controller.phase, Phase.DAMPING)

    def test_stale_handoff_never_sends_position_command(self):
        controller = AnkleController(self.transport, self.recorder, config())
        count = len(self.transport.writes)
        with patch("ankle_test.g1_ankle_pr_test.time.monotonic_ns",
                   return_value=self.epoch + 100_000_000):
            with self.assertRaisesRegex(RuntimeError, "stale"):
                controller._released_zero_position_callback(make_baseline())
        self.assertEqual(len(self.transport.writes), count)


class WorkflowTests(unittest.TestCase):
    setUp = GainAcquisitionTests.setUp
    tick = GainAcquisitionTests.tick

    def ready(self):
        self.tick(3)
        self.tick(3.002)
        self.tick(4.004)

    def prepare(self):
        self.ready()
        with patch('ankle_test.g1_ankle_pr_test.time.monotonic_ns', return_value=self.epoch + 4_004_000_000):
            self.controller.prepare_ankle_zero()

    def test_prepare_run_and_confirm_are_gated(self):
        with self.assertRaisesRegex(RuntimeError, 'captured_hold'):
            self.controller.prepare_ankle_zero()
        self.ready()
        with self.assertRaises(RuntimeError):
            self.controller.run_axis('left-pitch')
        with self.assertRaises(RuntimeError):
            self.controller.confirm_zero_hold()

    def test_each_ankle_must_settle_before_next_one_and_deadline_is_strict(self):
        self.prepare()
        duration = self.controller.trajectory.duration_s
        target = self.controller.trajectory.sample(duration)[0]
        # Simulate successful following at small intervals.
        for t in np.arange(4.104, 4.004 + duration, .1):
            self.tick(float(t), q=self.controller.last_target_q)
        t = 4.004 + duration + .01
        self.tick(t, q=target)
        self.assertEqual(self.controller.phase, Phase.ZERO_SETTLING)
        self.assertEqual(self.controller.zeroing_slot, 4)
        unsettled = target.copy()
        unsettled[4] = math.radians(-3)
        self.tick(t + .1, q=unsettled)
        self.assertEqual(self.controller.phase, Phase.ZERO_SETTLING)
        self.tick(t + .2, q=target)
        # A control callback delayed until the deadline cannot advance to slot 5.
        self.tick(t + 5.0, q=target)
        self.assertEqual(self.controller.phase, Phase.DAMPING)
        self.assertEqual(self.controller.zeroing_slot, 4)

    def test_full_workflow_and_two_cycles_keep_non_ankles_captured(self):
        self.prepare()
        non_ankles = np.setdiff1d(ACTIVE_SLOTS, ANKLE_SLOTS)
        elapsed = 4.004
        for _ in range(600):
            elapsed += .1
            cmd = self.tick(elapsed, q=self.controller.last_target_q)
            np.testing.assert_allclose(np.array([m.q for m in cmd.motor_cmd])[non_ankles], self.q0[non_ankles])
            self.assertEqual(cmd.crc, self.transport.crc.Crc(cmd))
            if self.controller.phase == Phase.ZERO_HOLD:
                break
        self.assertEqual(self.controller.phase, Phase.ZERO_HOLD)
        started = [v['slot'] for k, v in self.recorder.events if k == 'ankle_zero_slot_started']
        completed = [v['slot'] for k, v in self.recorder.events if k == 'ankle_zero_slot_completed']
        self.assertEqual(started, [4, 5, 10, 11])
        self.assertEqual(completed, started)
        with patch('ankle_test.g1_ankle_pr_test.time.monotonic_ns', return_value=self.epoch + round(elapsed*1e9)):
            self.controller.confirm_zero_hold()
            self.controller.run_axis('left-pitch')
        for _ in range(300):
            elapsed += .1
            cmd = self.tick(elapsed, q=self.controller.last_target_q)
            np.testing.assert_allclose(np.array([m.q for m in cmd.motor_cmd])[non_ankles], self.q0[non_ankles])
            if self.controller.phase == Phase.ZERO_HOLD:
                break
        self.assertEqual(self.controller.phase, Phase.ZERO_HOLD)
        self.assertEqual(sum(k == 'axis_complete' for k, _ in self.recorder.events), 1)

    def test_stop_during_preparation_immediately_enters_damping(self):
        self.prepare()
        self.controller.request_return(close_after=True)
        cmd = self.tick(4.1)
        self.assertEqual(self.controller.phase, Phase.DAMPING)
        np.testing.assert_allclose([m.kp for m in cmd.motor_cmd], 0)

    def test_speed_warning_during_zeroing_does_not_cancel_preparation(self):
        self.prepare()
        dq = np.zeros(35); dq[4] = .272321
        cmd = self.tick(4.1, dq=dq)
        self.assertEqual(self.controller.phase, Phase.ZEROING)
        self.assertGreater(cmd.motor_cmd[4].kp, 0)
        self.assertIn('speed_warning_started', [k for k, _ in self.recorder.events])

    def test_normal_axis_exit_returns_before_damping_and_closes_publisher(self):
        self.ready()
        target = self.q0.copy(); target[list(ANKLE_SLOTS)] = 0
        self.controller.phase = Phase.ZERO_HOLD
        self.controller.last_target_q = target.copy()
        self.controller.zero_confirmed = True
        now = self.epoch + 5_000_000_000
        self.transport.state = replace(make_state(), q=target, monotonic_ns=now)
        self.controller.last_write_ns = now - 2_000_000
        with patch('ankle_test.g1_ankle_pr_test.time.monotonic_ns', return_value=now):
            self.controller.run_axis('left-pitch')
        self.tick(6, q=target)
        self.assertGreater(self.controller.current_offset, 0)
        with patch('ankle_test.g1_ankle_pr_test.time.monotonic_ns', return_value=self.epoch + 6_000_000_000):
            self.controller.request_return(close_after=True)
        self.assertEqual(self.controller.phase, Phase.RETURNING)
        self.tick(8.1, q=target)
        self.assertEqual(self.controller.phase, Phase.DAMPING)
        self.tick(9.2, q=target)
        self.assertEqual(self.controller.phase, Phase.CLOSED)
        self.assertTrue(self.transport.disabled)

    def test_other_faults_still_abort_even_with_high_speed(self):
        for cause in ('motorstate', 'temperature', 'nonfinite', 'limit', 'remote', 'write', 'recorder'):
            with self.subTest(cause=cause):
                self.setUp()
                state = self.transport.state
                dq = state.dq.copy(); dq[4] = .8
                values = dict(dq=dq)
                if cause == 'motorstate':
                    a = state.motorstate.copy(); a[4] = 1; values['motorstate'] = a
                elif cause == 'temperature':
                    a = state.temperature.copy(); a[4, 1] = 80; values['temperature'] = a
                elif cause == 'nonfinite':
                    a = state.q.copy(); a[4] = np.nan; values['q'] = a
                elif cause == 'limit':
                    a = state.q.copy(); a[4] = 4; values['q'] = a
                elif cause == 'remote':
                    a = state.wireless_remote.copy(); a[2] = 32; a[3] = 2; values['wireless_remote'] = a
                elif cause == 'write':
                    self.controller.consecutive_write_failures = 3
                else:
                    self.recorder.failed = SimpleNamespace(is_set=lambda: True)
                    self.recorder.failure_reason = 'disk failure'
                self.transport.state = replace(state, **values)
                with patch('ankle_test.g1_ankle_pr_test.time.monotonic_ns', return_value=self.epoch):
                    self.controller._tick()
                self.assertEqual(self.controller.phase, Phase.DAMPING)


class CliSafetyTests(unittest.TestCase):
    def test_default_cli_is_read_only(self):
        args = parse_args(["--network", "enp3s0", "--run-name", "test"])
        self.assertFalse(args.enable_lowcmd)
        self.assertEqual(args.pitch_amplitude_deg, 5.0)
        self.assertEqual(args.roll_amplitude_deg, 3.0)
        self.assertEqual(args.ramp_duration, 3.0)
        self.assertEqual(args.hold_duration, 2.0)
        self.assertEqual(args.zero_min_duration, 8.0)
        self.assertEqual(args.zero_max_target_speed_deg_s, 3.0)
        self.assertEqual(args.gain_ramp_duration, 3.0)
        self.assertEqual(args.speed_warning_rad_s, 0.25)
        self.assertEqual(args.zero_speed_tolerance_deg_s, 5.0)
        cfg = make_config(args)
        self.assertEqual(cfg.gain_ramp_s, 3.0)
        self.assertEqual(cfg.gain_settle_s, 1.0)
        self.assertEqual(cfg.gain_settle_timeout_s, 5.0)
        self.assertAlmostEqual(cfg.zero_max_target_speed_rad_s, math.radians(3))

    def test_rejects_unsafe_gain_acquisition_parameters(self):
        for extra in (["--gain-ramp-duration", "2"],
                      ["--gain-ramp-duration", "nan"],
                      ["--gain-settle-duration", "0.1"],
                      ["--gain-settle-timeout", "1"],
                      ["--gain-settle-timeout", "inf"],
                      ["--max-abs-dq", "0.01"]):
            with self.subTest(extra=extra), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(["--network", "eno1", "--run-name", "offline"] + extra)

    def test_old_speed_parameter_has_explicit_migration_error(self):
        error = io.StringIO()
        with redirect_stderr(error), self.assertRaises(SystemExit):
            parse_args(['--network', 'eno1', '--run-name', 'test', '--max-abs-dq', '.25'])
        self.assertIn('--speed-warning-rad-s', error.getvalue())
        self.assertIn('no longer aborts', error.getvalue())

    def test_rejects_large_first_stage_amplitude(self):
        with self.assertRaises(SystemExit):
            parse_args(["--network", "enp3s0", "--run-name", "test",
                        "--pitch-amplitude-deg", "11"])

    def test_rejects_fast_ramp(self):
        with self.assertRaises(SystemExit):
            parse_args(["--network", "enp3s0", "--run-name", "test",
                        "--ramp-duration", "0.5"])

    def test_rejects_real_commands_on_loopback(self):
        with self.assertRaises(SystemExit):
            parse_args(["--network", "lo", "--run-name", "test", "--enable-lowcmd"])


if __name__ == "__main__":
    unittest.main()
