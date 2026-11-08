import unittest
import numpy as np
from slot_forecasting.schema import Query,Play
from .models import examples,fit_softmax,local_predictions


class ModelChecks(unittest.TestCase):
    def test_historical_label_is_not_a_feature(self):
        q = Query('q','ep','m','steady',0,30)
        a = [Play('ep','m',t,0 if t%2 else 1) for t in range(30)]
        b = [Play('ep','m',r.tick,20 if r.tick==20 else r.payout) for r in a]
        first,second = examples([q],a)[0],examples([q],b)[0]
        index = 20-16
        np.testing.assert_equal(first[1][index],second[1][index])
        self.assertNotEqual(first[2][index],second[2][index])

    def test_regularization_and_missing_history(self):
        anchor = np.zeros((53,5))
        np.testing.assert_equal(fit_softmax(np.zeros((0,53)),np.array([],dtype=int),np.array([]),anchor,8),anchor)
        q = Query('q','ep','m','steady',0,30)
        p = local_predictions(examples([q],[]),anchor,8,64,True)['q']
        np.testing.assert_allclose(p,np.ones(5)/5)


if __name__ == '__main__':
    unittest.main()
