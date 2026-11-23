"""Meaningful offline invariants for sensitivity injection and truth isolation."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_inekf_sensitivity as audit


class SensitivityTests(unittest.TestCase):
    def setUp(self):
        self.time = np.arange(0, .102, .002)
        self.input = {k: np.arange(len(self.time)*n,dtype=float).reshape(-1,n)/1000
                      for k,n in [('q',29),('dq',29),('tau',29),('gyro',3),('accel',3)]}
        self.profile = dict(dq_mean_rad_s=[.001,-.003,.0002,-.002],
                            std=dict(q=[.001]*4,dq=[.003]*4,gyro=[.001]*3,accel=[.01]*3))
        self.init = .02

    def test_all_16_cases_preserve_prefix_and_input(self):
        before={k:v.copy() for k,v in self.input.items()}
        for case in audit.scenarios():
            result=audit.perturb(self.time,self.input,case,self.profile,self.init)
            for k in self.input:
                np.testing.assert_array_equal(result[k][self.time<=self.init],self.input[k][self.time<=self.init])
                np.testing.assert_array_equal(self.input[k],before[k])
        self.assertEqual(len(audit.scenarios()),16)

    def test_sign_swap_and_residual_affect_only_specified_fields(self):
        cases={c['id']:c for c in audit.scenarios()}
        for name in ('sign_4','swap_4_5','dq_residual'):
            result=audit.perturb(self.time,self.input,cases[name],self.profile,self.init)
            for k in ('tau','gyro','accel'):
                np.testing.assert_array_equal(result[k],self.input[k])
            for k in ('q','dq'):
                np.testing.assert_array_equal(result[k][:,[0,1,6,12,15,28]],self.input[k][:,[0,1,6,12,15,28]])
        sign=audit.perturb(self.time,self.input,cases['sign_4'],self.profile,self.init)
        self.assertEqual(sign['q'][-1,4],-self.input['q'][-1,4])
        swap=audit.perturb(self.time,self.input,cases['swap_4_5'],self.profile,self.init)
        np.testing.assert_array_equal(swap['dq'][-1,[4,5]],self.input['dq'][-1,[5,4]])
        bias=audit.perturb(self.time,self.input,cases['dq_residual'],self.profile,self.init)
        np.testing.assert_allclose(bias['dq'][-1,audit.SLOTS]-self.input['dq'][-1,audit.SLOTS],self.profile['dq_mean_rad_s'])
        np.testing.assert_array_equal(bias['q'],self.input['q'])

    def test_delay_uses_old_raw_samples_including_non_correction_rows(self):
        for ms in (2,10,20):
            result=audit.perturb(self.time,self.input,dict(kind='delay',delay_ms=ms),self.profile,self.init)
            for k in ('q','dq','tau'):
                np.testing.assert_array_equal(result[k][-1],self.input[k][-1-ms//2])
            np.testing.assert_array_equal(result['gyro'],self.input['gyro'])
        # A large hypothetical future spike cannot leak into delayed input.
        self.input['q'][-1]=1e6
        result=audit.perturb(self.time,self.input,dict(kind='delay',delay_ms=2),self.profile,self.init)
        np.testing.assert_array_equal(result['q'][-1],self.input['q'][-2])

    def test_noise_is_reproducible_and_seed_sensitive(self):
        a=audit.perturb(self.time,self.input,dict(kind='noise',seed=0),self.profile,self.init)
        b=audit.perturb(self.time,self.input,dict(kind='noise',seed=0),self.profile,self.init)
        c=audit.perturb(self.time,self.input,dict(kind='noise',seed=1),self.profile,self.init)
        for k in a: np.testing.assert_array_equal(a[k],b[k])
        self.assertFalse(np.array_equal(a['gyro'],c['gyro']))
        np.testing.assert_array_equal(a['tau'],self.input['tau'])

    def test_rederived_feet_com_contact_and_protected_truth(self):
        import yaml
        from p2_generate_dataset import row_header
        cfg=yaml.safe_load((audit.PROJECT/'assets/p2_offline_estimation.yaml').read_text())
        time=np.array([0.,.02,.04,.06])
        values={k:np.zeros((len(time),n)) for k,n in [('q',29),('dq',29),('tau',29),('gyro',3),('accel',3)]}
        values['q'][:,4]=.1;values['q'][:,5]=.05;values['accel'][:,2]=9.81
        fields=row_header();rows=[]
        solver=audit.G1Kinematics();rule=audit.ProprioceptiveContactPipeline()
        for i,t in enumerate(time):
            r={k:'' for k in fields}
            r.update(time_s=str(t),physics_step=str(i*10),phase='cruise',kinematic_update='1',
                     quat_w='1',quat_x='0',quat_y='0',quat_z='0')
            for k,keys in audit.MEASUREMENT_KEYS.items():r.update(zip(keys,map(str,values[k][i])))
            for k in fields:
                if k.startswith('truth_') or k in ('left_normal_N','right_normal_N',
                    'left_contact_tangent_speed_m_s','right_contact_tangent_speed_m_s'):
                    r[k]='0'
            audit.recompute_row(r,t,values,i,solver,rule,cfg['rule_contact'])
            rows.append({k:str(v) for k,v in r.items()})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source_path=root/'source.csv'
            with source_path.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
            source=dict(path=source_path,fields=fields,rows=rows,time=time,measurements=values,metadata={})
            errors=audit.derive(source,root/'derived.csv',dict(id='sign_4',kind='sign',slot=4),
                                self.profile,cfg,audit.DEFAULT_URDF,.02)
            self.assertGreater(errors['left']['rmse_m'],.001)
            with (root/'derived.csv').open() as f:out=list(csv.DictReader(f))
            self.assertEqual(out[0],rows[0]);self.assertEqual(out[1],rows[1])
            for r,old in zip(out,rows):
                for k in fields:
                    if k not in audit.MUTABLE_KEYS:self.assertEqual(r[k],old[k])
            self.assertNotEqual(out[-1]['imu_to_left_sole_x'],rows[-1]['imu_to_left_sole_x'])
            self.assertNotEqual(out[-1]['com_pelvis_x'],rows[-1]['com_pelvis_x'])
            self.assertNotEqual(out[-1]['feature_left_sole_z'],rows[-1]['feature_left_sole_z'])

    def test_invalid_results_remain_in_summary_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            records=[dict(trial='t',case='baseline',contact_source='rule',status='failed',reason='never initialized')]
            audit.artifacts(Path(tmp),dict(cases=['baseline'],trials=['t']),records)
            summary=json.loads((Path(tmp)/'summary.json').read_text())
            self.assertEqual(summary['attempted'],1)
            self.assertEqual(summary['status_counts']['failed'],1)


if __name__=='__main__':unittest.main()
