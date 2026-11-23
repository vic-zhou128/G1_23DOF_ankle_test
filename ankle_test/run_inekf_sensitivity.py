#!/usr/bin/env python3
"""Offline P2 input sensitivity audit. No DDS, SDK, simulator or robot control imports."""
from __future__ import annotations

import argparse
import contextlib
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT))
from g1_estimator import (DEFAULT_URDF, FEATURE_NAMES, G1EstimatorInput,
                          G1Kinematics, ProprioceptiveContactPipeline)
from p2_evaluate_offline import evaluate

SLOTS = [4, 5, 10, 11]
TRIALS = ('no_push', 'right_40N')
MEASUREMENT_KEYS = {p: [f'{p}_{i}' for i in range(29)] for p in ('q', 'dq', 'tau')}
MEASUREMENT_KEYS.update({p: [f'{p}_{a}' for a in 'xyz'] for p in ('gyro', 'accel')})
KIN_KEYS = [f'{p}_{a}' for p in ('imu_to_left_sole', 'imu_to_right_sole',
                                'com_pelvis', 'com_velocity_pelvis') for a in 'xyz']
RULE_KEYS = ['rule_usable', 'rule_mode'] + [f'rule_{p}_{s}' for p in
    ('probability', 'slip', 'contact') for s in ('left', 'right')]
FEATURE_KEYS = [f'feature_{s}_{n}' for s in ('left', 'right') for n in FEATURE_NAMES]
MUTABLE_KEYS = set(sum(MEASUREMENT_KEYS.values(), []) + KIN_KEYS + RULE_KEYS + FEATURE_KEYS)
METRICS = ('base_position_xy_m', 'base_velocity_xy_m_s', 'com_position_xy_m',
           'com_velocity_xy_m_s', 'dcm_xy_m', 'tilt_error_deg')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    # A failed run must remain reportable even if a diagnostic value is nonfinite.
    def finite_json(x):
        if isinstance(x, dict): return {k:finite_json(v) for k,v in x.items()}
        if isinstance(x, (list,tuple)): return [finite_json(v) for v in x]
        if isinstance(x, float) and not np.isfinite(x): return None
        return x
    Path(path).write_text(json.dumps(finite_json(value), indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def scenarios():
    return ([dict(id='baseline', kind='baseline')]
            + [dict(id=f'sign_{s}', kind='sign', slot=s) for s in SLOTS]
            + [dict(id=f'swap_{a}_{b}', kind='swap', slots=[a, b]) for a, b in ((4, 5), (10, 11))]
            + [dict(id='dq_residual', kind='dq_residual')]
            + [dict(id=f'delay_{ms}ms', kind='delay', delay_ms=ms) for ms in (2, 10, 20)]
            + [dict(id=f'noise_seed_{seed}', kind='noise', seed=seed) for seed in range(5)])


def hardware_profile(run):
    """Statistics from last ten seconds before exit damping, preserving both provenance and units."""
    run = Path(run)
    summary = json.loads((run / 'summary.json').read_text())
    if summary['outcome'] != 'operator_exit' or summary['dropped'] or summary['failed']:
        raise ValueError('hardware profile requires a complete, clean operator-exit run')
    events = [json.loads(s) for s in (run / 'events.jsonl').read_text().splitlines() if s.strip()]
    exits = [e for e in events if e['kind'] == 'protective_damping']
    if len(exits) != 1 or exits[0]['reason'] != 'operator exit':
        raise ValueError('expected one operator-exit damping event')
    end = exits[0]['monotonic_ns']; start = end - 10_000_000_000
    arrays = {k: [] for k in ('time', 'q', 'dq', 'gyro', 'accel')}
    sources = []
    for path in sorted(run.glob('state_*.npz')):
        with np.load(path, allow_pickle=False) as z:
            keep = (z['monotonic_ns'] >= start) & (z['monotonic_ns'] < end)
            if not np.any(keep):
                continue
            if not np.all((z['crc_valid'][keep] == 1) & (z['mode_pr'][keep] == 0)
                          & (z['mode_machine'][keep] == 4)):
                raise ValueError('invalid hardware profile feedback')
            for key, field in [('time', 'monotonic_ns'), ('q', 'motor_q'), ('dq', 'motor_dq'),
                               ('gyro', 'imu_gyroscope'), ('accel', 'imu_accelerometer')]:
                values = z[field][keep]
                arrays[key].append(values[:, SLOTS] if key in ('q', 'dq') else values)
            sources.append(dict(path=str(path.resolve()), sha256=digest(path)))
    arrays = {k: np.concatenate(v) for k, v in arrays.items()}
    if len(arrays['time']) < 100 or np.ptp(arrays['time']) < 9.9e9:
        raise ValueError('incomplete ten-second hardware profile window')
    if any(not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('nonfinite hardware profile')
    command_samples = 0
    for path in sorted(run.glob('command_*.npz')):
        with np.load(path, allow_pickle=False) as z:
            keep = (z['monotonic_ns'] >= start) & (z['monotonic_ns'] < end)
            if np.any(keep):
                if (not np.all(z['phase'][keep] == 'zero_hold')
                        or np.any(z['motor_q'][keep][:, SLOTS] != 0)):
                    raise ValueError('profile window is not an ankle zero hold')
                command_samples += int(np.count_nonzero(keep))
                sources.append(dict(path=str(path.resolve()), sha256=digest(path)))
    if not command_samples:
        raise ValueError('no hold commands in profile window')
    return dict(run=str(run.resolve()), start_ns=start, end_ns=end,
                samples=len(arrays['time']), slots=SLOTS, command_samples=command_samples,
                dq_mean_rad_s=arrays['dq'].mean(axis=0, dtype=np.float64).tolist(),
                std={k: v.std(axis=0, dtype=np.float64).tolist() for k, v in arrays.items() if k != 'time'},
                units=dict(q='rad', dq='rad/s', gyro='rad/s', accel='m/s^2'),
                sources=sources + [dict(path=str((run/name).resolve()),sha256=digest(run/name))
                                   for name in ('summary.json','events.jsonl','metadata.json')],
                limitations=['One suspended hold is not a stationary sensor calibration.',
                             'Independent per-sample Gaussian draws omit colored noise, bias drift and cross-channel correlation.',
                             'Measured dq means are diagnostic residuals, not calibrated constant offsets.'])


def load_dataset(path, config):
    path = Path(path)
    meta = json.loads(path.with_suffix('.metadata.json').read_text())
    if digest(path) != meta['dataset_sha256']:
        raise ValueError(f'dataset hash mismatch: {path}')
    if meta['model_identity'] != config['model_variant'] or meta['status'] != 'done':
        raise ValueError('source model or completion status mismatch')
    if (meta['truth_contact_definition'] != config['truth_contact']
            or meta['rule_contact_definition'] != config['rule_contact']):
        raise ValueError('source contact definitions differ from fixed P2 configuration')
    with path.open() as f:
        reader = csv.DictReader(f); rows = list(reader); fields = reader.fieldnames
    t = np.array([float(r['time_s']) for r in rows])
    if not np.all(np.diff(t) > 0) or not np.allclose(np.diff(t), config['physics_dt_s'], atol=1e-10, rtol=0):
        raise ValueError('source is not a strictly increasing P2 physics-rate stream')
    measurements = {p: np.array([[float(r[k]) for k in keys] for r in rows])
                    for p, keys in MEASUREMENT_KEYS.items()}
    if any(not np.isfinite(v).all() for v in measurements.values()):
        raise ValueError('source measurements contain nonfinite values')
    return dict(path=path, metadata=meta, rows=rows, fields=fields, time=t, measurements=measurements)


def perturb(time, measurements, case, profile, initialization_time):
    """Start strictly after baseline initialization. Delays use only old physics-rate samples."""
    out = {k: v.copy() for k, v in measurements.items()}
    active = time > initialization_time + 1e-9
    kind = case['kind']
    if kind == 'sign':
        for k in ('q', 'dq'):
            out[k][active, case['slot']] *= -1
    elif kind == 'swap':
        a, b = case['slots']
        for k in ('q', 'dq'):
            out[k][np.ix_(active, [a, b])] = measurements[k][np.ix_(active, [b, a])]
    elif kind == 'dq_residual':
        out['dq'][np.ix_(active, SLOTS)] += profile['dq_mean_rad_s']
    elif kind == 'delay':
        # Epsilon accommodates accumulated floating point in the exact 2 ms source grid.
        old = np.searchsorted(time, time - case['delay_ms'] / 1000 + 1e-10, side='right') - 1
        old = np.maximum(old, 0)
        if np.any(time[old[active]] > time[active] + 1e-10):
            raise ValueError('delay attempted future-sample access')
        for k in ('q', 'dq', 'tau'):
            out[k][active] = measurements[k][old[active]]
    elif kind == 'noise':
        rng = np.random.default_rng(case['seed'])
        n = int(active.sum())
        for k in ('q', 'dq'):
            out[k][np.ix_(active, SLOTS)] += rng.normal(size=(n, 4)) * profile['std'][k]
        for k in ('gyro', 'accel'):
            out[k][active] += rng.normal(size=(n, 3)) * profile['std'][k]
    elif kind != 'baseline':
        raise ValueError(f'unknown perturbation {kind}')
    return out


def recompute_row(row, time, measurements, index, solver, rule, config):
    q, dq, tau = (measurements[k][index] for k in ('q', 'dq', 'tau'))
    kin = solver.compute(q, dq)
    if not kin.valid:
        raise ValueError(f'kinematics at {time}: {kin.reason}')
    quat = np.array([float(row[k]) for k in ('quat_w', 'quat_x', 'quat_y', 'quat_z')])
    sample = G1EstimatorInput(round(time * 1e9), round(time * 1000), quat,
        measurements['gyro'][index], measurements['accel'][index], q, dq, tau)
    contact = rule.update(sample, kin)
    if not contact.valid:
        raise ValueError('invalid recomputed contact features')
    for prefix, values in [('imu_to_left_sole', kin.imu_T_left_sole[:3, 3]),
                           ('imu_to_right_sole', kin.imu_T_right_sole[:3, 3]),
                           ('com_pelvis', kin.com_pelvis), ('com_velocity_pelvis', kin.com_velocity_pelvis)]:
        row.update({f'{prefix}_{a}': float(values[i]) for i, a in enumerate('xyz')})
    for side, side_name in enumerate(('left', 'right')):
        p = [contact.left_probability, contact.right_probability][side]
        slip = [contact.slip_probability_left, contact.slip_probability_right][side]
        contacting = contact.mode in (('DOUBLE_SUPPORT', 'LEFT_SUPPORT') if side == 0
                                      else ('DOUBLE_SUPPORT', 'RIGHT_SUPPORT'))
        row.update({f'rule_probability_{side_name}': p, f'rule_slip_{side_name}': slip,
                    f'rule_contact_{side_name}': int(contacting and slip <= config['maximum_slip_probability'])})
        row.update({f'feature_{side_name}_{n}': float(contact.features[side, i])
                    for i, n in enumerate(FEATURE_NAMES)})
    row.update(rule_usable=int(contact.usable_by_estimator), rule_mode=contact.mode)


def derive(source, output, case, profile, config, urdf, init_time):
    changed = perturb(source['time'], source['measurements'], case, profile, init_time)
    solver = G1Kinematics(urdf)
    rule = ProprioceptiveContactPipeline(config['rule_contact']['enter_probability'],
                                        config['rule_contact']['exit_probability'])
    foot_errors = [[], []]
    with Path(output).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=source['fields']); writer.writeheader()
        for i, old in enumerate(source['rows']):
            row = old.copy(); now = source['time'][i]
            if now > init_time + 1e-9:
                for p, keys in MEASUREMENT_KEYS.items():
                    row.update(zip(keys, map(float, changed[p][i])))
            if old['kinematic_update'] == '1':
                # Warm the temporal contact pipeline on the pristine prefix too.
                recompute_row(row, now, changed, i, solver, rule, config['rule_contact'])
                if now <= init_time + 1e-9:
                    # Prove the regenerated prefix agrees before retaining original bytes/values.
                    for k in KIN_KEYS + RULE_KEYS + FEATURE_KEYS:
                        if k == 'rule_mode':
                            if row[k] != old[k]: raise ValueError('baseline contact prefix mismatch')
                        elif not np.isclose(float(row[k]), float(old[k]), atol=1e-12, rtol=1e-12):
                            raise ValueError(f'baseline prefix mismatch: {k}')
                    row = old.copy()
                if now > init_time + 1e-9:
                    for j, side in enumerate(('left', 'right')):
                        keys = [f'imu_to_{side}_sole_{a}' for a in 'xyz']
                        foot_errors[j].append(float(np.linalg.norm(
                            [float(row[k])-float(old[k]) for k in keys])))
            # Truth, contact references, timestamps, phase and quaternion must remain untouched.
            if any(row[k] != old[k] for k in source['fields'] if k not in MUTABLE_KEYS):
                raise ValueError('protected source field was changed')
            writer.writerow(row)
    meta = dict(source['metadata'], schema='g1_p2_sensitivity_dataset_v1',
                dataset=str(Path(output).resolve()), dataset_sha256=digest(output),
                parent_dataset=str(source['path'].resolve()), parent_sha256=digest(source['path']),
                sensitivity=dict(case=case, initialization_time_s=init_time,
                    starts='strictly after initialization', urdf_sha256=digest(urdf),
                    feedback_semantics='PR assumed for sensitivity analysis, not identified by this experiment'))
    write_json(Path(output).with_suffix('.metadata.json'), meta)
    return {side: dict(rmse_m=float(np.sqrt(np.mean(np.square(values)))), max_m=float(max(values)))
            for side, values in zip(('left', 'right'), foot_errors)}


def estimate_health(path, init_time):
    with Path(path).open() as f:
        rows = list(csv.DictReader(f))
    initialized = next((i for i, r in enumerate(rows) if r['valid'] == '1'), None)
    if initialized is None:
        return dict(status='invalid', reason='never initialized')
    times = np.array([float(r['time_s']) for r in rows])
    finite = all(np.isfinite(float(v)) for r in rows[initialized:] for v in r.values())
    invalid = sum(r['valid'] != '1' for r in rows[initialized:])
    covariance = [float(r['covariance_trace']) for r in rows[initialized:]]
    health = dict(status='ok', first_valid_time_s=float(times[initialized]),
                  invalid_samples_after_initialization=invalid, finite=finite,
                  maximum_covariance_trace=float(max(covariance)))
    if invalid or not finite or min(covariance) < 0 or abs(times[initialized] - init_time) > 1e-9:
        health.update(status='invalid', reason='invalid states or initialization changed')
    return health


def compare_baseline(actual, reference):
    with Path(actual).open() as f: a = list(csv.DictReader(f))
    with Path(reference).open() as f: b = list(csv.DictReader(f))
    if len(a) != len(b): raise ValueError('baseline row count changed')
    maximum = max(abs(float(x[k])-float(y[k])) for x, y in zip(a, b)
                  for k in x if x[k] and y[k])
    if maximum > 1e-10:
        raise ValueError(f'zero-perturbation baseline differs from historical replay: {maximum}')
    return dict(reference=str(Path(reference).resolve()), reference_sha256=digest(reference),
                maximum_absolute_numeric_difference=maximum, tolerance=1e-10)


def run_pair(dataset, destination, contact_source, config_path, binary, init_time, reference=None):
    output = destination / f'{contact_source}.csv'
    cmd = [sys.executable, str(PROJECT/'p2_run_inekf.py'), '--p2-config', str(config_path),
           '--binary', str(binary), '--input', str(dataset), '--output', str(output),
           '--contact-source', contact_source]
    result = dict(contact_source=contact_source, status='failed')
    try:
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        (destination/f'{contact_source}.log').write_text(process.stdout + '\n' + process.stderr)
        if process.returncode: raise RuntimeError(f'runner exited {process.returncode}: {process.stderr[-1000:]}')
        result.update(estimate_health(output, init_time))
        if result['status'] != 'ok': return result
        if reference is not None:
            result['baseline_equivalence'] = compare_baseline(output, reference)
        with contextlib.redirect_stdout(io.StringIO()):
            report = evaluate(dataset, output, config_path, destination/f'{contact_source}_evaluation.json')
        result.update(cruise=report['cruise'], policy_active=report['policy_active'],
                      valid_fraction_cruise=report['valid_fraction_cruise'],
                      disturbance_windows=report['disturbance_windows'],
                      contacts=report['contact_rule']['cruise'])
        if report['cruise'] is None or report['valid_fraction_cruise'] != 1:
            result.update(status='invalid', reason='incomplete cruise coverage')
    except Exception as exc:
        result.update(status='failed', reason=f'{type(exc).__name__}: {exc}')
    return result


def artifacts(destination, manifest, records, complete=False):
    for r in records:
        baseline = next((b for b in records if b['trial']==r['trial'] and b['case']=='baseline'
                         and b['contact_source']==r['contact_source'] and b['status']=='ok'), None)
        if r['status']=='ok' and baseline:
            r['cruise_delta_rmse'] = {k:r['cruise'][k]['rmse']-baseline['cruise'][k]['rmse'] for k in METRICS}
    summary = dict(manifest=manifest, complete=complete, attempted=len(records),
                   status_counts=dict((s, sum(r['status']==s for r in records)) for s in ('ok','invalid','failed')),
                   records=records)
    write_json(destination/'summary.json', summary)
    columns = ['trial', 'case', 'contact_source', 'status', 'reason', 'valid_fraction_cruise']
    columns += [f'{k}_rmse' for k in METRICS] + [f'{k}_delta_rmse' for k in METRICS]
    columns += [f'{s}_foot_input_rmse_m' for s in ('left','right')]
    columns += [f'{s}_{k}' for s in ('left','right') for k in ('false_positive_rate_given_no_truth','miss_rate_given_truth')]
    with (destination/'summary.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for r in records:
            row={k:r.get(k,'') for k in columns[:6]}
            for k in METRICS:
                row[f'{k}_rmse']=r.get('cruise',{}).get(k,{}).get('rmse','')
                row[f'{k}_delta_rmse']=r.get('cruise_delta_rmse',{}).get(k,'')
            for side in ('left','right'):
                row[f'{side}_foot_input_rmse_m']=r.get('foot_input_difference',{}).get(side,{}).get('rmse_m','')
                for k in ('false_positive_rate_given_no_truth','miss_rate_given_truth'):
                    row[f'{side}_{k}']=r.get('contacts',{}).get(side,{}).get(k,'')
            writer.writerow(row)
    if complete:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        cases=manifest['cases']; trials=manifest['trials']
        fig, axes=plt.subplots(len(trials),2,figsize=(13,5*len(trials)),squeeze=False)
        for i, trial in enumerate(trials):
            for j, metric in enumerate(('com_velocity_xy_m_s','dcm_xy_m')):
                ax=axes[i,j]
                for source in ('truth_stable','rule'):
                    rows=[next((r for r in records if r['trial']==trial and r['case']==case
                                and r['contact_source']==source),{}) for case in cases]
                    values=[r.get('cruise',{}).get(metric,{}).get('rmse',np.nan) if r.get('status')=='ok' else np.nan for r in rows]
                    ax.plot(range(len(cases)),values,'o-',label=source)
                ax.set(title=trial, ylabel=f'{metric} RMSE', xticks=range(len(cases)))
                ax.set_xticklabels(cases,rotation=60,ha='right',fontsize=8);ax.grid(alpha=.2);ax.legend()
        fig.suptitle('Fixed-trajectory input sensitivity; development data, not hardware coordinate identification')
        fig.tight_layout();fig.savefig(destination/'sensitivity.pdf');fig.savefig(destination/'sensitivity.png',dpi=160)
        plt.close(fig)
        lines=['G1 InEKF 输入敏感性分析（开发轮）',
               f'完成回放：{len(records)}/{manifest["planned_replays"]}；状态：{summary["status_counts"]}',
               '使用固定轨迹和固定滤波参数；扰动在基线初始化之后开始。',
               '单位：位置/DCM 为 m，速度为 m/s，倾角为 deg。以下数值为匀速段 RMSE。',
               'case | trial | contact | base velocity | CoM velocity | DCM | DCM difference from baseline']
        for r in records:
            prefix=f'{r["case"]} | {r["trial"]} | {r["contact_source"]}'
            if r['status']=='ok':
                lines.append(prefix+' | '+' | '.join(f'{r["cruise"][k]["rmse"]:.8f}' for k in
                    ('base_velocity_xy_m_s','com_velocity_xy_m_s','dcm_xy_m'))
                    +f' | {r.get("cruise_delta_rmse",{}).get("dcm_xy_m",0):+.8f}')
            else: lines.append(prefix+f' | {r["status"]}: {r.get("reason", "")}')
        lines += ['', '解释边界：', *manifest['limitations'],
                  '误差变小只表示该轨迹上的误差抵消，不能据此选择错误坐标或调参。',
                  '本测试不能确认实机 PR/AB 语义、力矩精度或闭环收益；不能用两条开发轨迹宣称泛化。',
                  '详细的足端差异、分窗估计误差、接触误报漏报、协方差及失败分母见 summary.json。']
        (destination/'report.txt').write_text('\n'.join(lines)+'\n')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True,help='new directory; never overwrites a previous audit')
    parser.add_argument('--p2-config',type=Path,default=PROJECT/'assets/p2_offline_estimation.yaml')
    parser.add_argument('--simulation-dir',type=Path,default=ROOT/'outputs/simulation')
    parser.add_argument('--hardware-run',type=Path,default=ROOT/'outputs/hardware/ankle_test/ankle_pr_qualitative_08')
    parser.add_argument('--urdf',type=Path,default=DEFAULT_URDF)
    parser.add_argument('--binary',type=Path,default=PROJECT/'build/p2_py311/g1_inekf_offline')
    parser.add_argument('--cases',nargs='+',choices=[c['id'] for c in scenarios()],help='optional subset; default all 16')
    parser.add_argument('--trials',nargs='+',choices=TRIALS,default=list(TRIALS))
    args=parser.parse_args(argv)
    config_path=args.p2_config.resolve(); cfg=yaml.safe_load(config_path.read_text())
    if args.output_dir.exists(): parser.error('output directory already exists')
    if cfg['schema']!='g1_p2_offline_estimation_v1': raise ValueError('unsupported P2 schema')
    if not args.binary.is_file(): raise FileNotFoundError(args.binary)
    upstream=ROOT/'third_party/invariant-ekf'
    head=subprocess.run(['git','-C',str(upstream),'rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()
    if head!=cfg['inekf']['upstream_commit']: raise ValueError('upstream commit differs from fixed P2 configuration')
    selected=[c for c in scenarios() if args.cases is None or c['id'] in args.cases]
    # Always retain a matched baseline for subset invocations.
    if selected[0]['id']!='baseline': selected.insert(0,scenarios()[0])
    trials=list(dict.fromkeys(args.trials))
    sources={t:load_dataset(args.simulation_dir/f'p2_{t}_seed_20261118_v4.csv',cfg) for t in trials}
    profile=hardware_profile(args.hardware_run.resolve())
    dest=args.output_dir.resolve(); dest.mkdir(parents=True)
    manifest=dict(schema='g1_inekf_input_sensitivity_v1', status='development_not_acceptance',
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  cases=[c['id'] for c in selected],scenario_definitions=selected,trials=trials,
                  planned_replays=len(selected)*len(trials)*2,
                  initialization_time_s=cfg['inekf']['estimator_start_time_s'],
                  sources={t:dict(path=str(s['path'].resolve()),sha256=digest(s['path']),
                      metadata_sha256=digest(s['path'].with_suffix('.metadata.json'))) for t,s in sources.items()},
                  identities={str(p.resolve()):digest(p) for p in (config_path,args.urdf,args.binary,Path(__file__),
                      PROJECT/'g1_estimator.py',PROJECT/'p2_run_inekf.py',PROJECT/'p2_evaluate_offline.py',PROJECT/'cpp/g1_inekf_offline.cpp')},
                  implementation='RossHartley/invariant-ekf',upstream_commit=head,
                  limitations=['Fixed-trajectory offline sensitivity, not estimator/controller-in-the-loop simulation.',
                    'Two development trajectories; five noise seeds are not independent locomotion trials.',
                    'No PR/AB verdict: sign and swap faults are not physical AB kinematics.',
                    'Noise statistics from a suspended ten-second hold are diagnostic, not calibrated sensor specifications.',
                    'Fixed initialization and filter parameters; no tuning on individual injected errors.'])
    write_json(dest/'manifest.json',manifest);write_json(dest/'hardware_profile.json',profile)
    records=[]
    for trial, source in sources.items():
        for case in selected:
            case_dir=dest/trial/case['id'];case_dir.mkdir(parents=True)
            dataset=case_dir/'input.csv'
            try:
                feet=derive(source,dataset,case,profile,cfg,args.urdf,manifest['initialization_time_s'])
                preparation_error=None
            except Exception as exc:
                preparation_error=f'{type(exc).__name__}: {exc}';feet={}
            for contact_source,label in [('truth_stable','truth'),('rule','rule')]:
                reference=args.simulation_dir/f'p2_{trial}_{label}_inekf_20261118_v4.csv' if case['kind']=='baseline' else None
                result=(dict(status='failed',contact_source=contact_source,reason='dataset preparation: '+preparation_error)
                        if preparation_error else run_pair(dataset,case_dir,contact_source,config_path,args.binary.resolve(),
                                                          manifest['initialization_time_s'],reference))
                result.update(trial=trial,case=case['id'],foot_input_difference=feet)
                records.append(result)
                artifacts(dest,manifest,records)
                print(f'[{len(records)}/{manifest["planned_replays"]}] {trial} {case["id"]} {contact_source}: {result["status"]}',flush=True)
            # A mismatching baseline makes all derived comparisons unreliable.
            if case['kind']=='baseline' and any(r['status']!='ok' for r in records[-2:]):
                raise RuntimeError('baseline failed; see summary.json; no sensitivity results accepted')
    if any(digest(Path(path))!=value for path,value in manifest['identities'].items()):
        raise RuntimeError('code/configuration changed during audit')
    if any(digest(s['path'])!=manifest['sources'][t]['sha256'] for t,s in sources.items()):
        raise RuntimeError('source dataset changed during audit')
    if any(digest(Path(s['path']))!=s['sha256'] for s in profile['sources']):
        raise RuntimeError('hardware profile source changed during audit')
    artifacts(dest,manifest,records,complete=True)
    print(f'[done] {dest}; {len(records)} replays; development only')
    return 0 if all(r['status']=='ok' for r in records) else 2


if __name__=='__main__':
    raise SystemExit(main())
