import unittest
import numpy as np
from .features import structural_features


class StructuralChecks(unittest.TestCase):
    def test_sparse_history_and_future(self):
        seq={t:0 for t in list(range(24))+list(range(160,200))}
        p=structural_features(seq,[seq],200)
        self.assertEqual(p[0],64)
        self.assertEqual(p[3],40)
        self.assertEqual(p[5],1)
        changed={**seq,200:20,201:5}
        np.testing.assert_equal(p,structural_features(changed,[changed],200))


if __name__=='__main__': unittest.main()
