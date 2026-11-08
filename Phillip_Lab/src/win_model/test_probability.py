import unittest
import numpy as np
from .run import factorize


class FactorizationChecks(unittest.TestCase):
    def test_composition(self):
        p=factorize([.4],[[1,2,1,0]])
        np.testing.assert_allclose(p,[[.6,.1,.2,.1,0]],atol=1e-10)
        np.testing.assert_allclose(p.sum(1),1)

    def test_extremes_are_finite(self):
        p=factorize([-1,2],np.zeros((2,4)))
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue((p>=0).all())
        np.testing.assert_allclose(p.sum(1),1)


if __name__=='__main__': unittest.main()
