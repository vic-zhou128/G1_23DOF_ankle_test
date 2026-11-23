#!/usr/bin/env python3
"""Offline qualitative q/dq analysis and diagnostic speed replay. No SDK/DDS imports."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

try:
    from .core import ACTIVE_JOINT_NAMES, ANKLE_SLOTS, AXIS_SLOTS, SpeedWarningTracker
except ImportError:
    from core import ACTIVE_JOINT_NAMES, ANKLE_SLOTS, AXIS_SLOTS, SpeedWarningTracker


def load_stream(directory, prefix, fields):
    chunks = []
    for path in sorted(Path(directory).glob(f"{prefix}_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            chunks.append({name: data[name] for name in fields})
    if not chunks:
        return {}
    return {name: np.concatenate([chunk[name] for chunk in chunks]) for name in fields}


def correlation(a, b):
    if len(a) < 3 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def window_derivative(t_ns, q, window_s):
    """Centered secants on raw received q, NaN at edges/gaps; no q smoothing."""
    t = (t_ns - t_ns[0]) * 1e-9
    half = window_s / 2
    left = np.maximum(0, np.searchsorted(t, t - half, side="right") - 1)
    right = np.searchsorted(t, t + half, side="left")
    right = np.minimum(right, len(t) - 1)
    dt = t[right] - t[left]
    valid = ((t >= half) & (t <= t[-1] - half) & (dt >= window_s * .5)
             & (dt <= window_s * 2))
    # Do not differentiate across missing feedback intervals (>50 ms).
    gaps = np.r_[0, np.cumsum(np.diff(t) > .05)]
    valid &= gaps[right] == gaps[left]
    result = np.full_like(q, np.nan, dtype=float)
    # Interpolate to symmetric window endpoints instead of comparing an
    # asymmetric secant with dq at the center (irregular host receive times).
    for column in range(q.shape[1]):
        before = np.interp(t - half, t, q[:, column])
        after = np.interp(t + half, t, q[:, column])
        result[valid, column] = (after[valid] - before[valid]) / window_s
    return result


def match_command_states(states, commands):
    """Match exactly the feedback snapshot used by each attempted write."""
    indexes = np.searchsorted(states['sequence'], commands['state_sequence'])
    clipped = np.minimum(indexes, len(states['sequence']) - 1)
    valid = ((indexes < len(states['sequence']))
             & (states['sequence'][clipped] == commands['state_sequence']))
    return clipped, valid


def replay_speed_warnings(states, commands, threshold, cutoff_ns=None):
    """Replay only control checks before damping; no extrapolation after a stop."""
    tracker = SpeedWarningTracker(threshold, ACTIVE_JOINT_NAMES)
    indexes, matched = match_command_states(states, commands)
    events = []
    last_active = None
    for i, row in enumerate(indexes):
        phase = str(commands['phase'][i])
        now = int(commands['monotonic_ns'][i])
        if phase in ('protective_damping', 'closed') or not matched[i]:
            # Old controllers abort before writing the triggering command.
            # Include only that pre-fault snapshot, using the last sent target
            # as context, never treating the subsequent damping as a motion trial.
            if (matched[i] and last_active is not None and cutoff_ns is not None
                    and int(states['monotonic_ns'][row]) <= cutoff_ns):
                j = last_active
                events.extend(tracker.update(states['monotonic_ns'][row], str(commands['phase'][j]),
                    states['motor_q'][row], states['motor_dq'][row],
                    commands['motor_q'][j], commands['motor_dq'][j]))
            events.extend(tracker.finish(now, 'excluded_or_unmatched'))
            continue
        events.extend(tracker.update(
            states['monotonic_ns'][row], phase, states['motor_q'][row],
            states['motor_dq'][row], commands['motor_q'][i], commands['motor_dq'][i]))
        if commands['write_success'][i] == 1:
            last_active = i
    if len(commands['monotonic_ns']):
        events.extend(tracker.finish(int(commands['monotonic_ns'][-1]), 'recording_end'))
    return [dict(kind=kind, **values) for kind, values in events]


def delay_diagnostic(command_ns, command_q, state_ns, state_q):
    """Best sampled nonnegative shift by RMSE, not a physical delay calibration."""
    if len(command_ns) < 20 or np.ptp(command_q) < math.radians(.5):
        return None
    t = (command_ns - command_ns[0]) * 1e-9
    s = (state_ns - command_ns[0]) * 1e-9
    keep = (t >= .5) & (t <= min(t[-1], s[-1]) - .5)
    if np.count_nonzero(keep) < 20:
        return None
    scores = []
    delays = np.arange(0, .502, .002)
    for delay in delays:
        feedback = np.interp(t[keep] + delay, s, state_q)
        scores.append(float(np.mean((feedback - command_q[keep]) ** 2)))
    best = int(np.argmin(scores))
    return dict(best_shift_ms=float(delays[best] * 1000),
                rmse_deg=math.degrees(math.sqrt(scores[best])),
                at_search_boundary=best in (0, len(delays) - 1),
                method='RMSE shift 0..500 ms, 2 ms grid; includes tracking dynamics')


def summarize_axis(commands, states, state_indexes, selected):
    axis = str(commands['axis'][selected[0]])
    slot = AXIS_SLOTS[axis]
    companion = slot + 1 if slot in (4, 10) else slot - 1
    rows = state_indexes[selected]
    start = int(commands['monotonic_ns'][selected[0]])
    end = int(commands['monotonic_ns'][selected[-1]])
    in_run = (states['monotonic_ns'] >= start) & (states['monotonic_ns'] <= end)
    q = states['motor_q'][in_run]
    if not len(q):
        return None
    result = dict(axis=axis, slot=slot, companion_slot=companion, start_ns=start, end_ns=end,
                  target_peak_to_peak_deg=math.degrees(float(np.ptp(commands['motor_q'][selected, slot]))),
                  feedback_peak_to_peak_deg=math.degrees(float(np.ptp(q[:, slot]))),
                  companion_peak_to_peak_deg=math.degrees(float(np.ptp(q[:, companion]))),
                  tracking_rmse_deg=math.degrees(float(np.sqrt(np.mean(
                      (commands['motor_q'][selected, slot] - states['motor_q'][rows, slot]) ** 2)))),
                  delay_diagnostic=delay_diagnostic(
                      commands['monotonic_ns'][selected], commands['motor_q'][selected, slot],
                      states['monotonic_ns'][in_run], q[:, slot]), endpoints=[])
    # Aggregate the final half of each endpoint hold, grouped by cycle/segment.
    for cycle in np.unique(commands['cycle'][selected]):
        for segment in np.unique(commands['segment'][selected]):
            if not str(segment).startswith('hold-'):
                continue
            idx = selected[(commands['cycle'][selected] == cycle)
                           & (commands['segment'][selected] == segment)]
            if not len(idx):
                continue
            midpoint = (int(commands['monotonic_ns'][idx[0]]) + int(commands['monotonic_ns'][idx[-1]])) // 2
            mask = ((states['monotonic_ns'] >= midpoint)
                    & (states['monotonic_ns'] <= commands['monotonic_ns'][idx[-1]]))
            if not np.any(mask):
                continue
            target_deg = math.degrees(float(commands['motor_q'][idx[-1], slot]))
            mean_deg = math.degrees(float(np.mean(states['motor_q'][mask, slot])))
            result['endpoints'].append(dict(
                cycle=int(cycle), segment=str(segment),
                target_deg=target_deg, q_mean_deg=mean_deg,
                tracking_bias_deg=mean_deg - target_deg,
                return_zero_error_deg=mean_deg if abs(target_deg) < 1e-8 else None,
                companion_mean_deg=math.degrees(float(np.mean(states['motor_q'][mask, companion]))),
                q_std_deg=math.degrees(float(np.std(states['motor_q'][mask, slot])))))
    result['repeatability_deg'] = {}
    for segment in sorted({e['segment'] for e in result['endpoints']}):
        values = [e['q_mean_deg'] for e in result['endpoints'] if e['segment'] == segment]
        result['repeatability_deg'][segment] = dict(cycles=len(values),
            range_deg=float(np.ptp(values)) if len(values) >= 2 else None)
    return result


def analyze(directory, threshold=.25, window_s=.02):
    directory = Path(directory)
    states = load_stream(directory, 'state', ('monotonic_ns', 'sequence', 'motor_q', 'motor_dq',
                                            'crc_valid', 'mode_pr', 'mode_machine'))
    commands = load_stream(directory, 'command', ('monotonic_ns', 'state_sequence', 'phase', 'axis',
        'segment', 'cycle', 'motor_q', 'motor_dq', 'write_success'))
    if not states:
        raise ValueError('no recorded LowState stream')
    valid = ((states['crc_valid'] == 1) & (states['mode_pr'] == 0) & (states['mode_machine'] == 4)
             & np.isfinite(states['motor_q']).all(axis=1) & np.isfinite(states['motor_dq']).all(axis=1))
    excluded = int(np.count_nonzero(~valid))
    states = {name: values[valid] for name, values in states.items()}
    if not len(states['sequence']):
        raise ValueError('no finite CRC-valid mode=4/0 states for analysis')
    report = dict(run=str(directory.resolve()), verdict='not_determined',
        evidence_scope='qualitative q/dq; operator observation required; no torque coordinate verdict',
        limitations=[
            'Tracking/URDF guards provisionally assume PR; passing them does not establish PR semantics.',
            'Visual direction alone does not calibrate angle scale or distinguish all AB mappings.',
            'Host receive timestamps differ from sensor sampling times.',
            'Historical runs stop at their old fault; no normal-motion inference after damping.'],
        speed_warning_rad_s=threshold, derivative_window_s=window_s, axes=[], dq_consistency={},
        excluded_state_samples=excluded,
        speed_warning_events=[], operator_observations=[])
    events_path = directory / 'events.jsonl'
    if events_path.exists():
        events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
        report['operator_observations'] = [e for e in events if e.get('kind') == 'operator_observation']
        report['operator_marks'] = [e for e in events if e.get('kind') == 'operator_mark']
        report['recorded_faults'] = [e for e in events if e.get('kind') == 'protective_damping']
    derived = window_derivative(states['monotonic_ns'], states['motor_q'][:, list(ANKLE_SLOTS)], window_s)
    if commands:
        faults = report.get('recorded_faults', [])
        cutoff = int(faults[0]['monotonic_ns']) if faults else None
        report['speed_warning_events'] = replay_speed_warnings(states, commands, threshold, cutoff)
        rows, matched = match_command_states(states, commands)
        report['unmatched_command_samples'] = int(np.count_nonzero(~matched))
        active = ~np.isin(commands['phase'], ['protective_damping', 'closed'])
        report['failed_write_samples'] = int(np.count_nonzero(commands['write_success'] != 1))
        if np.any(active):
            lo = int(commands['monotonic_ns'][active][0])
            # Exclude post-fault feedback and derivative windows crossing into damping.
            stops = commands['monotonic_ns'][~active]
            hi = int(stops[0] if len(stops) else commands['monotonic_ns'][-1])
            valid_time = ((states['monotonic_ns'] >= lo + int(window_s * 1e9))
                          & (states['monotonic_ns'] < hi - int(window_s * 1e9)))
        else:
            valid_time = np.zeros(len(states['sequence']), dtype=bool)
        # Split running episodes, including repeated commands for the same axis.
        running = (commands['phase'] == 'running') & matched & (commands['write_success'] == 1)
        starts = np.flatnonzero(running & ~np.r_[False, running[:-1]
            & (commands['axis'][:-1] == commands['axis'][1:])])
        for start in starts:
            end = start + 1
            while end < len(running) and running[end] and commands['axis'][end] == commands['axis'][start]:
                end += 1
            result = summarize_axis(commands, states, rows, np.arange(start, end))
            if result is not None:
                report['axes'].append(result)
    else:
        valid_time = np.ones(len(states['sequence']), dtype=bool)
        report['note'] = 'read-only run; no sent commands or speed replay'
    for i, slot in enumerate(ANKLE_SLOTS):
        valid = valid_time & np.isfinite(derived[:, i])
        delta = derived[valid, i] - states['motor_dq'][valid, slot]
        report['dq_consistency'][str(slot)] = dict(
            samples=int(np.count_nonzero(valid)),
            rmse_rad_s=float(np.sqrt(np.mean(delta ** 2))) if len(delta) else None,
            correlation=correlation(derived[valid, i], states['motor_dq'][valid, slot]),
            raw_q_peak_to_peak_deg=math.degrees(float(np.ptp(states['motor_q'][valid, slot]))) if len(delta) else None)
    return report, states, commands, derived


def save_artifacts(destination, report, states, commands, derived):
    """Raw q/dq plus analysis-window derivative; never overwrite raw NPZ files."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    if commands:
        margin = int(report['derivative_window_s'] * 1e9)
        mask = ((states['monotonic_ns'] >= commands['monotonic_ns'][0] - margin)
                & (states['monotonic_ns'] <= commands['monotonic_ns'][-1] + margin))
        states = {name: values[mask] for name, values in states.items()}
        derived = derived[mask]
    (destination / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    with (destination / 'samples.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['monotonic_ns'] + [f'{kind}_{slot}' for slot in ANKLE_SLOTS
                                            for kind in ('raw_q_rad', 'raw_dq_rad_s', 'window_dq_rad_s')])
        for row, now in enumerate(states['monotonic_ns']):
            writer.writerow([int(now)] + [value for i, slot in enumerate(ANKLE_SLOTS)
                for value in (states['motor_q'][row, slot], states['motor_dq'][row, slot], derived[row, i])])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    origin = states['monotonic_ns'][0]
    t = (states['monotonic_ns'] - origin) * 1e-9
    for i, slot in enumerate(ANKLE_SLOTS):
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(t, np.degrees(states['motor_q'][:, slot]), label='raw feedback q', lw=.8)
        axes[1].plot(t, states['motor_dq'][:, slot], label='raw feedback dq', lw=.6, alpha=.7)
        axes[1].plot(t, derived[:, i], label=f'q secant ({report["derivative_window_s"]*1000:g} ms)', lw=.8)
        if commands:
            ct = (commands['monotonic_ns'] - origin) * 1e-9
            ok = (commands['write_success'] == 1) & ~np.isin(commands['phase'], ['protective_damping', 'closed'])
            axes[0].plot(ct, np.where(ok, np.degrees(commands['motor_q'][:, slot]), np.nan), label='sent q', lw=.8)
            axes[1].plot(ct, np.where(ok, commands['motor_dq'][:, slot], np.nan), label='sent dq', lw=.8)
            stops = commands['monotonic_ns'][commands['phase'] == 'protective_damping']
            if len(stops):
                for ax in axes:
                    ax.axvline((stops[0] - origin)*1e-9, color='r', ls='--', label='damping begins')
        axes[0].set(ylabel='q [deg]', title=f'Slot {slot}; coordinate semantics not determined')
        axes[1].set(ylabel='dq [rad/s]', xlabel='host receive time since first state [s]')
        for ax in axes:
            ax.legend(loc='best'); ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(destination / f'slot_{slot}.png', dpi=150)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--speed-warning-rad-s', type=float, default=.25)
    parser.add_argument('--derivative-window-ms', type=float, default=20)
    parser.add_argument('--output-dir', type=Path, help='new directory for JSON, raw CSV and plots; otherwise stdout only')
    args = parser.parse_args(argv)
    if not all(math.isfinite(x) and x > 0 for x in (args.speed_warning_rad_s, args.derivative_window_ms)):
        parser.error('warning threshold and derivative window must be positive and finite')
    report, states, commands, derived = analyze(args.run_dir, args.speed_warning_rad_s, args.derivative_window_ms / 1000)
    if args.output_dir:
        save_artifacts(args.output_dir, report, states, commands, derived)
        print(f'[analysis] {args.output_dir.resolve()}; verdict=not_determined')
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
