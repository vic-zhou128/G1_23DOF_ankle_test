"""Offline regression tests; these tests never initialize DDS or send LowCmd."""
from __future__ import annotations

from dataclasses import dataclass
import math
from types import SimpleNamespace
import unittest

import numpy as np

try:
    from .core import (ANKLE_SLOTS, AXIS_SLOTS, AxisTrajectory, ReturnTrajectory,
                       SequentialAnkleZeroTrajectory, ZeroPostureTrajectory,
                       VelocityProbeTrajectory, WIRE_MOTOR_COUNT, overlay_axis,
                       select_official_baseline, select_zero_command_handoff,
                       smoothstep5, smoothstep5_rate)
except ImportError:
    from core import (ANKLE_SLOTS, AXIS_SLOTS, AxisTrajectory, ReturnTrajectory,
                      SequentialAnkleZeroTrajectory, ZeroPostureTrajectory,
                      VelocityProbeTrajectory, WIRE_MOTOR_COUNT, overlay_axis,
                      select_official_baseline, select_zero_command_handoff,
                      smoothstep5, smoothstep5_rate)


ACTIVE = np.asarray([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    15, 16, 17, 18, 19, 22, 23, 24, 25, 26,
], dtype=np.int64)


def command_at(monotonic_ns, q_shift=0.0, crc_valid=True, mode_pr=0, gain=20.0):
    q = np.zeros(WIRE_MOTOR_COUNT)
    q[4] = q_shift
    return SimpleNamespace(
        monotonic_ns=monotonic_ns, mode_machine=4, mode_pr=mode_pr,
        q=q, dq=np.zeros(WIRE_MOTOR_COUNT), tau=np.zeros(WIRE_MOTOR_COUNT),
        kp=np.full(WIRE_MOTOR_COUNT, gain),
        kd=np.full(WIRE_MOTOR_COUNT, 0.0 if gain == 0 else 1.0),
        motor_mode=np.ones(WIRE_MOTOR_COUNT, dtype=np.uint8),
        crc_valid=crc_valid)


def state_at(monotonic_ns, dq=0.0, mode_pr=0):
    return SimpleNamespace(
        monotonic_ns=monotonic_ns, mode_machine=4, mode_pr=mode_pr,
        q=np.zeros(WIRE_MOTOR_COUNT),
        dq=np.full(WIRE_MOTOR_COUNT, dq),
        tau_est=np.zeros(WIRE_MOTOR_COUNT))


class SmoothTrajectoryTests(unittest.TestCase):
    def test_smoothstep_endpoints(self):
        self.assertEqual(smoothstep5(0.0), 0.0)
        self.assertEqual(smoothstep5(1.0), 1.0)
        self.assertEqual(smoothstep5_rate(0.0), 0.0)
        self.assertEqual(smoothstep5_rate(1.0), 0.0)

    def test_axis_sequence_and_duration(self):
        trajectory = AxisTrajectory(math.radians(3), 2.0, 1.0, 2)
        self.assertAlmostEqual(trajectory.duration_s, 24.0)
        self.assertAlmostEqual(trajectory.sample(0).offset_rad, 0.0)
        self.assertAlmostEqual(trajectory.sample(2.0).offset_rad, math.radians(3))
        self.assertAlmostEqual(trajectory.sample(5.0).offset_rad, 0.0)
        self.assertAlmostEqual(trajectory.sample(8.0).offset_rad, -math.radians(3))
        final = trajectory.sample(24.0)
        self.assertTrue(final.done)
        self.assertAlmostEqual(final.offset_rad, 0.0)
        samples = [trajectory.sample(value / 100.0) for value in range(2401)]
        self.assertLessEqual(max(abs(item.offset_rad) for item in samples),
                             math.radians(3) + 1e-12)

    def test_return_is_smooth_and_monotonic(self):
        trajectory = ReturnTrajectory(0.05, 2.0)
        samples = [trajectory.sample(value / 100.0) for value in range(201)]
        offsets = np.asarray([item.offset_rad for item in samples])
        self.assertTrue(np.all(np.diff(offsets) <= 1e-12))
        self.assertAlmostEqual(samples[0].velocity_rad_s, 0.0)
        self.assertTrue(samples[-1].done)
        self.assertAlmostEqual(samples[-1].velocity_rad_s, 0.0)

    def test_overlay_changes_exactly_one_slot(self):
        baseline_q = np.linspace(-0.1, 0.1, WIRE_MOTOR_COUNT)
        baseline_dq = np.linspace(-0.01, 0.01, WIRE_MOTOR_COUNT)
        for _, slot in AXIS_SLOTS.items():
            q, dq = overlay_axis(baseline_q, baseline_dq, slot, 0.02, -0.03)
            changed_q = np.flatnonzero(np.abs(q - baseline_q) > 1e-12)
            changed_dq = np.flatnonzero(np.abs(dq - baseline_dq) > 1e-12)
            np.testing.assert_array_equal(changed_q, [slot])
            np.testing.assert_array_equal(changed_dq, [slot])
        np.testing.assert_array_equal(sorted(AXIS_SLOTS.values()), sorted(ANKLE_SLOTS))

    def test_velocity_probe_is_symmetric_and_ends_at_zero(self):
        trajectory = VelocityProbeTrajectory(math.radians(2), 1.0, 0.5, 1)
        samples = [trajectory.sample(value / 100.0)
                   for value in range(int(trajectory.duration_s * 100) + 1)]
        targets = np.asarray([item.target_dq_rad_s for item in samples])
        self.assertLessEqual(np.max(targets), math.radians(2) + 1e-12)
        self.assertGreaterEqual(np.min(targets), -math.radians(2) - 1e-12)
        self.assertAlmostEqual(float(np.sum(targets)), 0.0, places=9)
        self.assertTrue(samples[-1].done)
        self.assertEqual(samples[-1].target_dq_rad_s, 0.0)

    def test_zero_posture_trajectory_is_speed_limited_and_ends_at_zero(self):
        initial = np.zeros(WIRE_MOTOR_COUNT)
        initial[ACTIVE] = np.linspace(-1.0, 1.0, len(ACTIVE))
        trajectory = ZeroPostureTrajectory(
            initial, ACTIVE, minimum_duration_s=8.0,
            max_target_speed_rad_s=0.1)
        self.assertGreaterEqual(trajectory.duration_s, 18.75)
        q0, dq0, blend0, done0 = trajectory.sample(0.0)
        np.testing.assert_allclose(q0, initial)
        np.testing.assert_allclose(dq0, 0.0)
        self.assertEqual(blend0, 0.0)
        self.assertFalse(done0)
        q1, dq1, blend1, done1 = trajectory.sample(trajectory.duration_s)
        np.testing.assert_allclose(q1[ACTIVE], 0.0, atol=1e-12)
        np.testing.assert_allclose(dq1, 0.0, atol=1e-12)
        self.assertEqual(blend1, 1.0)
        self.assertTrue(done1)
        speeds = []
        for elapsed in np.linspace(0.0, trajectory.duration_s, 1001):
            _, dq, _, _ = trajectory.sample(float(elapsed))
            speeds.append(float(np.max(np.abs(dq[ACTIVE]))))
        self.assertLessEqual(max(speeds), 0.1 + 1e-12)

    def test_sequential_ankle_zero_moves_exactly_one_ankle_at_a_time(self):
        initial = np.linspace(-0.3, 0.3, WIRE_MOTOR_COUNT)
        initial[4] = -0.30
        initial[5] = 0.10
        initial[10] = -0.20
        initial[11] = -0.08
        trajectory = SequentialAnkleZeroTrajectory(
            initial, ANKLE_SLOTS, minimum_slot_duration_s=2.0,
            max_target_speed_rad_s=0.1)
        non_ankles = np.setdiff1d(np.arange(WIRE_MOTOR_COUNT), ANKLE_SLOTS)
        previous_q = initial.copy()
        for elapsed in np.linspace(0.0, trajectory.duration_s, 2001):
            q, dq, progress, done, moving_slot = trajectory.sample(float(elapsed))
            np.testing.assert_allclose(q[non_ankles], initial[non_ankles])
            moving = np.flatnonzero(np.abs(dq) > 1e-12)
            self.assertLessEqual(len(moving), 1)
            if moving_slot is not None and moving.size:
                self.assertEqual(int(moving[0]), moving_slot)
            self.assertLessEqual(float(np.max(np.abs(dq))), 0.1 + 1e-12)
            self.assertGreaterEqual(progress, 0.0)
            self.assertLessEqual(progress, 1.0)
            previous_q = q
        np.testing.assert_allclose(previous_q[list(ANKLE_SLOTS)], 0.0, atol=1e-12)
        self.assertTrue(done)
        self.assertIsNone(moving_slot)

    def test_sequential_ankle_zero_preserves_later_ankles_until_their_turn(self):
        initial = np.zeros(WIRE_MOTOR_COUNT)
        initial[list(ANKLE_SLOTS)] = [-0.2, 0.1, -0.15, -0.05]
        trajectory = SequentialAnkleZeroTrajectory(initial, ANKLE_SLOTS, 2.0, 0.2)
        first_midpoint = trajectory.slot_durations_s[0] / 2.0
        q, dq, _progress, done, moving_slot = trajectory.sample(first_midpoint)
        self.assertEqual(moving_slot, 4)
        self.assertFalse(done)
        np.testing.assert_allclose(q[[5, 10, 11]], initial[[5, 10, 11]])
        np.testing.assert_allclose(dq[[5, 10, 11]], 0.0)


class BaselineCaptureTests(unittest.TestCase):
    def setUp(self):
        # 500 Hz histories spanning just over two seconds.
        self.states = [state_at(index * 2_000_000) for index in range(1101)]
        self.commands = [command_at(index * 2_000_000) for index in range(1101)]

    def select(self, **overrides):
        values = dict(window_s=2.0, max_state_dq=0.2,
                      max_command_span_rad=0.1, min_ankle_kp=1.0)
        values.update(overrides)
        return select_official_baseline(self.states, self.commands, [], ACTIVE, **values)

    def test_accepts_stable_pr_capture_without_using_tau_residual(self):
        final, diagnostics = self.select()
        self.assertEqual(final.mode_pr, 0)
        self.assertGreaterEqual(diagnostics["window_s"], 1.9)
        self.assertEqual(diagnostics["max_state_dq_rad_s"], 0.0)

    def test_rejects_any_ab_mode(self):
        self.commands[-1] = command_at(self.commands[-1].monotonic_ns, mode_pr=1)
        with self.assertRaisesRegex(ValueError, "LowCmd mode changed"):
            self.select()

    def test_rejects_crc_failure(self):
        self.commands[-10] = command_at(self.commands[-10].monotonic_ns, crc_valid=False)
        with self.assertRaisesRegex(ValueError, "CRC"):
            self.select()

    def test_rejects_motion(self):
        self.states = [state_at(index * 2_000_000, dq=0.25) for index in range(1101)]
        with self.assertRaisesRegex(ValueError, "not stationary"):
            self.select()

    def test_rejects_unstable_official_command(self):
        for index in range(900, 1101):
            self.commands[index] = command_at(index * 2_000_000,
                                               q_shift=0.2 * (index % 2))
        with self.assertRaisesRegex(ValueError, "not stable"):
            self.select()

    def test_rejects_invalid_lowstate_in_window(self):
        with self.assertRaisesRegex(ValueError, "invalid sample"):
            select_official_baseline(self.states, self.commands, [1_500_000_000],
                                     ACTIVE, window_s=2.0)


class ZeroCommandHandoffTests(unittest.TestCase):
    def setUp(self):
        self.states = [state_at(index * 2_000_000) for index in range(1101)]
        self.commands = [command_at(index * 2_000_000, gain=0.0)
                         for index in range(1101)]

    def select(self):
        return select_zero_command_handoff(
            self.states, self.commands, [], ACTIVE,
            window_s=2.0, max_state_dq=0.2)

    def test_accepts_stable_zero_lowcmd(self):
        final, diagnostics = self.select()
        self.assertEqual(final.mode_pr, 0)
        self.assertEqual(diagnostics["max_abs_official_command"], 0.0)

    def test_rejects_nonzero_position_target(self):
        self.commands[-1] = command_at(
            self.commands[-1].monotonic_ns, q_shift=1e-3, gain=0.0)
        with self.assertRaisesRegex(ValueError, "not a zero-command stream"):
            self.select()

    def test_rejects_ab_mode(self):
        self.commands[-1] = command_at(
            self.commands[-1].monotonic_ns, gain=0.0, mode_pr=1)
        with self.assertRaisesRegex(ValueError, "LowCmd mode changed"):
            self.select()


if __name__ == "__main__":
    unittest.main()
