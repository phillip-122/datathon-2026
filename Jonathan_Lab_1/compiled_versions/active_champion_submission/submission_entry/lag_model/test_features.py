import unittest
import numpy as np
from .features import changing_relationships


class RelationshipChecks(unittest.TestCase):
    def test_no_future_and_missing(self):
        a={t:0 if t%2 else 1 for t in range(100)}
        first=changing_relationships(a,80)
        b=dict(a); b.update({80:20,81:5,90:2})
        np.testing.assert_equal(first,changing_relationships(b,80))
        self.assertEqual(len(first),128)
        del a[79]
        missing=changing_relationships(a,80)
        self.assertEqual(missing[:2],[0,0])
        self.assertTrue(np.isnan(missing[2:8]).all())

    def test_reversal_changes_signed_lift(self):
        # Older outcomes alternate (negative lag-1 association), recent outcomes
        # persist in blocks (positive association).
        seq={t:t%2 for t in range(64)}
        seq.update({t:((t-64)//4)%2 for t in range(64,96)})
        f=changing_relationships(seq,96)
        self.assertGreater(f[4],0)
        self.assertLess(f[5],0)
        self.assertGreater(f[6],0)
        self.assertLess(f[7],0)


if __name__=='__main__': unittest.main()
