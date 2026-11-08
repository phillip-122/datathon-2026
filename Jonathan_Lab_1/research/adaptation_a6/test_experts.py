import unittest
from types import SimpleNamespace as Row
import numpy as np
from experts import bank,replace_win,CONFIGS


class ExpertTests(unittest.TestCase):
    def test_cutoff_episode_isolation_and_missing_ticks(self):
        q=[Row(episode_id='e',machine_id='m',machine_type='balanced',signal=0,cutoff=50)]
        h=[Row(episode_id='e',machine_id='m',tick=t,payout=(0,1,5)[t%3]) for t in list(range(12))+list(range(30,49))]
        P=np.array([[.6,.2,.1,.08,.02]])
        B,M=bank(q,h,P)
        extra=[Row(episode_id='e',machine_id='m',tick=50,payout=20),Row(episode_id='e',machine_id='m',tick=51,payout=1),Row(episode_id='other',machine_id='m',tick=49,payout=1)]
        BB,MM=bank(q,list(reversed(h+extra)),P)
        np.testing.assert_array_equal(B,BB); np.testing.assert_array_equal(M,MM)
        self.assertEqual(B.shape,(1,len(CONFIGS)))
        self.assertTrue(np.isfinite(B).all()); self.assertTrue(((B>=0)&(B<=1)).all())
        output=replace_win(P,B[:,0],1.)
        np.testing.assert_allclose(output.sum(1),1.,atol=1e-14)
        np.testing.assert_allclose(output[:,1:]/output[:,1:].sum(1,keepdims=True),P[:,1:]/P[:,1:].sum(1,keepdims=True))


if __name__=='__main__': unittest.main()
