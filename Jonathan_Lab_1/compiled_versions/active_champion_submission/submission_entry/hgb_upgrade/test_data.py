import unittest
import numpy as np
from .run import masked


class MaskChecks(unittest.TestCase):
    def test_budget_and_no_future(self):
        sequence={t:t%5 for t in range(300)}
        a=masked(sequence,200,np.random.default_rng(42))
        b=masked(sequence,200,np.random.default_rng(42))
        self.assertEqual(a,b)
        self.assertEqual(len(a),64)
        self.assertTrue(all(t<200 for t in a))
        self.assertTrue(all(t in a for t in range(160,200)))
        self.assertEqual(sum(t<160 for t in a),24)


if __name__=='__main__': unittest.main()
