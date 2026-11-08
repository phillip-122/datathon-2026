import unittest
import numpy as np
from slot_forecasting.schema import Query,Play
from .system import nearest_simplex,vertices_for,alternatives


INFO={'high_ratio':[.8,.196,.004],'one_share':[.45,.75]}


class GeometryTests(unittest.TestCase):
    def test_projection_does_not_increase_error_for_compatible_truth(self):
        rng=np.random.default_rng(42); V=vertices_for(INFO,True)
        for _ in range(50):
            point=rng.dirichlet(np.ones(5)); truth=rng.dirichlet(np.ones(3))@V
            projected=nearest_simplex(point,V)
            self.assertLessEqual(np.sum((projected-truth)**2),np.sum((point-truth)**2)+1e-10)
            np.testing.assert_allclose(projected.sum(),1,atol=1e-12)
            np.testing.assert_allclose(nearest_simplex(projected,V),projected,atol=1e-10)

    def test_regime_uses_observed_positive_payouts_and_missing_ticks(self):
        q=Query('q','ep','m','steady',0,20)
        raw={'q':(.5,.3,.15,.049,.001)}
        info={'steady':INFO}
        a=alternatives([q],[],raw,info)
        b=alternatives([q],[Play('ep','m',2,0),Play('ep','m',18,0)],raw,info)
        np.testing.assert_allclose(a['bayes_all']['q'],b['bayes_all']['q'])
        positive=alternatives([q],[Play('ep','m',18,1)],raw,info)
        self.assertGreater(positive['bayes_all']['q'][1]/sum(positive['bayes_all']['q'][1:]),a['bayes_all']['q'][1]/sum(a['bayes_all']['q'][1:]))


if __name__=='__main__': unittest.main()
