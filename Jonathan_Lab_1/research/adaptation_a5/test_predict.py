"""Cutoff and probability-contract checks for the new temporal correction."""
import unittest
from types import SimpleNamespace as Row
import numpy as np
from predict import local_rates,replace_win


class TemporalTests(unittest.TestCase):
    def test_future_and_other_episodes_do_not_change_prediction(self):
        q=[Row(episode_id='e',machine_id='m',cutoff=50)]
        h=[Row(episode_id='e',machine_id='m',tick=t,payout=int(t%3==0)) for t in range(40)]
        P=np.array([[.6,.2,.1,.08,.02]])
        expected=local_rates(q,h,P,64,4,True)
        extra=[Row(episode_id='e',machine_id='m',tick=50,payout=20),Row(episode_id='e',machine_id='m',tick=51,payout=1),Row(episode_id='other',machine_id='m',tick=49,payout=1)]
        np.testing.assert_array_equal(expected,local_rates(q,h+extra,P,64,4,True))

    def test_missing_ticks_and_probability_normalization(self):
        q=[Row(episode_id='e',machine_id='m',cutoff=50)]
        h=[Row(episode_id='e',machine_id='m',tick=t,payout=1) for t in (1,7,32,40)]
        P=np.array([[.6,.2,.1,.08,.02]])
        rates=local_rates(q,h,P,64,4,True)
        result=replace_win(P,rates,.5)
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue((result>=0).all())
        np.testing.assert_allclose(result.sum(1),1,atol=1e-14)
        np.testing.assert_allclose(result[:,1:]/result[:,1:].sum(1,keepdims=True),P[:,1:]/P[:,1:].sum(1,keepdims=True))


if __name__=='__main__': unittest.main()
