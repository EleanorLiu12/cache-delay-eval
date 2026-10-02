import runpy
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
calibrate=runpy.run_path(str(ROOT/'scripts/calibrate_eviction_costs.py'))['calibrate']


class CostTests(unittest.TestCase):
    def sample(self, run, session, time=1501):
        return dict(device='NVIDIA A30', timing_scope='complete_iteration_wall', session_ids=[session],
                    source_measurements='test-only fabricated timing fixture',
                    samples=[dict(signature='[["prefill",16,0]]',duration_ns=time,run_id=run)])

    def test_disjointness_required(self):
        with self.assertRaises(ValueError):
            calibrate(self.sample('a','same'),self.sample('b','same'))
        with self.assertRaises(ValueError):
            calibrate(self.sample('same','a'),self.sample('same','b'))

    def test_rounding_error_and_unknown_signature(self):
        a,b=self.sample('a','a'),self.sample('b','b',2000)
        r=calibrate(a,b)
        self.assertEqual(r['cost_table']['[["prefill",16,0]]'],2)
        self.assertTrue(r['accepted'])
        self.assertFalse(r['trajectory_validated'])
        b['samples'][0]['duration_ns']=4000
        self.assertFalse(calibrate(a,b)['accepted'])
        b['samples'][0]['signature']='[["decode",1,16]]'
        self.assertEqual(calibrate(a,b)['coverage'],0)

    def test_admission_time_not_accepted_as_service_time(self):
        a=self.sample('a','a'); a['timing_scope']='admission_section_elapsed_ns'
        with self.assertRaises(ValueError):
            calibrate(a,self.sample('b','b'))


if __name__=='__main__':
    unittest.main()
