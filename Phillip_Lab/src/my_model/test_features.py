"""Checks for temporal leakage, exact lag offsets, and model persistence."""
import tempfile
import unittest
from pathlib import Path
from slot_forecasting.schema import Query, Play
from .features import FEATURE_NAMES, HistoryFeatures
from .model import HouseEdgeForecaster
import numpy as np


class FeaturesTest(unittest.TestCase):
    def test_missing_tick_and_neighbor_future_are_not_used(self):
        queries = [Query("q1", "ep", "a", "steady", 0, 10), Query("q2", "ep", "b", "balanced", 1, 10)]
        history = [Play("ep", "a", 6, 2), Play("ep", "a", 8, 1),
                   Play("ep", "b", 7, 5), Play("ep", "b", 9, 20)]
        row = HistoryFeatures(queries, history).row(queries[0], cutoff=9)
        self.assertEqual(row[FEATURE_NAMES.index("lag_1_is_1")], 1)
        self.assertTrue(np.isnan(row[FEATURE_NAMES.index("lag_2_is_0")]))
        self.assertEqual(row[FEATURE_NAMES.index("neighbor_8_count")], 1)
        self.assertEqual(row[FEATURE_NAMES.index("neighbor_8_p_20")], 0)
        changed = history + [Play("ep", "a", 9, 20)]
        np.testing.assert_equal(row, HistoryFeatures(queries, changed).row(queries[0], cutoff=9))

    def test_transition_counts_and_smoothing(self):
        q = Query("q", "ep", "a", "steady", 0, 6)
        h = [Play("ep", "a", t, p) for t, p in enumerate([0, 1, 0, 2, 0, 5])]
        row = HistoryFeatures([q], h).row(q)
        # At tick 6, lag 1 is category 2 (5x). Prior category-2 tick 3
        # was followed by a zero at tick 4: exactly one matching transition.
        self.assertEqual(row[FEATURE_NAMES.index("transition_lag_1_all_count")], 1)
        probs = [row[FEATURE_NAMES.index(f"transition_lag_1_all_p_{p}")] for p in (0,1,2,5,20)]
        self.assertAlmostEqual(sum(probs), 1)
        self.assertGreater(probs[0], probs[1])
        # At cutoff 5 the observed tick-5 outcome cannot affect any feature.
        altered = h[:-1] + [Play("ep", "a", 5, 20)]
        np.testing.assert_equal(HistoryFeatures([q], h).row(q, 5),
                                HistoryFeatures([q], altered).row(q, 5))

    def test_saved_model_reproduces(self):
        queries = [Query(f"q{i}", "ep", str(i), "steady", 0, 40) for i in range(5)]
        history = [Play("ep", str(i), t, (0, 1, 2, 5, 20)[(t+i) % 5]) for i in range(5) for t in range(40)]
        labels = {f"q{i}": (0, 1, 2, 5, 20)[i] for i in range(5)}
        model = HouseEdgeForecaster(rounds=2).fit(queries, history, labels)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.ubj"
            model.save(path)
            restored = HouseEdgeForecaster.load(path)
            for key, probs in model.predict(queries, history).items():
                np.testing.assert_allclose(probs, restored.predict(queries, history)[key], atol=1e-8)
                self.assertAlmostEqual(sum(probs), 1)


if __name__ == "__main__":
    unittest.main()
