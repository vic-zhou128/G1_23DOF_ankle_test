"""Offline analysis regression fixtures; no robot or DDS participant."""
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .analyze_run import analyze, delay_diagnostic, replay_speed_warnings, window_derivative
from .core import AxisTrajectory, SpeedWarningTracker


class SpeedWarningTests(unittest.TestCase):
    def test_phase_and_stream_boundaries_close_episodes_and_keep_peak_context(self):
        tracker = SpeedWarningTracker(.25, [4])
        q = np.zeros(35); target = q.copy(); target[4] = .1
        dq = q.copy(); dq[4] = .27
        events = tracker.update(10_000_000, 'zeroing', q, dq, target, q)
        self.assertEqual(events[0][0], 'speed_warning_started')
        dq[4] = .4
        self.assertFalse(tracker.update(12_000_000, 'zeroing', q, dq, target, q))
        events = tracker.update(14_000_000, 'zero_settling', q, dq, target, q)
        self.assertEqual([e[0] for e in events], ['speed_warning_ended', 'speed_warning_started'])
        self.assertEqual(events[0][1]['peak_abs_dq_rad_s'], .4)
        self.assertEqual(events[0][1]['position_error_at_peak_rad'], .1)
        self.assertTrue(events[0][1]['truncated'])
        self.assertEqual(tracker.finish(16_000_000)[0][1]['duration_s'], .002)

    def test_replay_includes_trigger_sample_but_excludes_damping_motion(self):
        states = dict(sequence=np.arange(3), monotonic_ns=np.array([100, 200, 300]),
                      motor_q=np.zeros((3, 35)), motor_dq=np.zeros((3, 35)))
        states['motor_dq'][1:, 4] = [.27, 2.0]
        commands = dict(monotonic_ns=np.array([110, 210, 310]), state_sequence=np.arange(3),
            phase=np.array(['zeroing', 'protective_damping', 'protective_damping']),
            motor_q=np.zeros((3, 35)), motor_dq=np.zeros((3, 35)), write_success=np.ones(3))
        events = replay_speed_warnings(states, commands, .25, cutoff_ns=205)
        ended = [e for e in events if e['kind'] == 'speed_warning_ended']
        self.assertEqual(len(ended), 1)
        self.assertEqual(ended[0]['peak_abs_dq_rad_s'], .27)
        self.assertEqual(ended[0]['phase'], 'zeroing')


class AnalysisTests(unittest.TestCase):
    def test_derivative_on_linear_angle_is_correct_and_does_not_bridge_gaps(self):
        seconds = np.r_[np.arange(0, .2, .001), np.arange(.4, .6, .001)]
        stamps = (seconds * 1e9).astype(np.int64)
        velocity = window_derivative(stamps, (seconds * .3)[:, None], .02)[:, 0]
        np.testing.assert_allclose(velocity[np.isfinite(velocity)], .3, atol=1e-12)
        self.assertTrue(np.isnan(velocity[195]))

    def test_delay_diagnostic_recovers_known_shift(self):
        t = np.arange(0, 5, .01)
        ns = (1e9 + t * 1e9).astype(np.int64)
        q = .1 * np.sin(2 * np.pi * .4 * t)
        delayed = .1 * np.sin(2 * np.pi * .4 * (t - .06))
        metric = delay_diagnostic(ns, q, ns, delayed)
        self.assertAlmostEqual(metric['best_shift_ms'], 60, delta=2.1)

    def test_two_cycles_endpoints_repeatability_and_no_automatic_verdict(self):
        trajectory = AxisTrajectory(math.radians(5), 3, 2, 2)
        t = np.arange(0, trajectory.duration_s, .01)
        samples = [trajectory.sample(float(x)) for x in t]
        count = len(t)
        stamps = (1e9 + t * 1e9).astype(np.int64)
        q = np.zeros((count, 35)); dq = q.copy()
        q[:, 4] = [s.offset_rad for s in samples]
        dq[:, 4] = [s.velocity_rad_s for s in samples]
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            np.savez(p / 'state_000000.npz', monotonic_ns=stamps, sequence=np.arange(count),
                     motor_q=q, motor_dq=dq, crc_valid=np.ones(count), mode_pr=np.zeros(count),
                     mode_machine=np.full(count, 4))
            np.savez(p / 'command_000000.npz', monotonic_ns=stamps, state_sequence=np.arange(count),
                     motor_q=q, motor_dq=dq, phase=np.full(count, 'running'),
                     axis=np.full(count, 'left-pitch'), segment=[s.segment for s in samples],
                     cycle=[s.cycle for s in samples], write_success=np.ones(count))
            report, _, _, _ = analyze(p)
        self.assertEqual(report['verdict'], 'not_determined')
        self.assertEqual(len(report['axes']), 1)
        axis = report['axes'][0]
        self.assertAlmostEqual(axis['feedback_peak_to_peak_deg'], 10)
        self.assertEqual(axis['companion_peak_to_peak_deg'], 0)
        self.assertEqual(len(axis['endpoints']), 8)
        for values in axis['repeatability_deg'].values():
            self.assertLess(values['range_deg'], 1e-10)
        self.assertLess(report['dq_consistency']['4']['rmse_rad_s'], 1e-5)
        json.dumps(report, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
