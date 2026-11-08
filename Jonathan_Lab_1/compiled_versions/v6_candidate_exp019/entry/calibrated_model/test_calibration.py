import unittest
import numpy as np
from .run import transform,fit


class CalibrationChecks(unittest.TestCase):
    def test_identity_and_extremes(self):
        P=np.array([[.5,.2,.15,.1,.05],[1.,0,0,0,0]])
        np.testing.assert_allclose(transform(P,np.zeros(6)),P,atol=1e-10)
        out=transform(P,np.array([.2,.1,-.1,.2,-.2,0]))
        self.assertTrue(np.isfinite(out).all())
        np.testing.assert_allclose(out.sum(1),1)

    def test_known_bias_generalizes(self):
        rng=np.random.default_rng(42)
        P=rng.dirichlet(np.ones(5),size=100)
        Q=transform(P,np.array([.1,.2,-.2,.1,0,-.1]))
        theta=fit(P[:70],Q[:70],np.ones(70)/70,.001)
        self.assertLess(np.mean((transform(P[70:],theta)-Q[70:])**2),np.mean((P[70:]-Q[70:])**2))


if __name__=='__main__': unittest.main()
