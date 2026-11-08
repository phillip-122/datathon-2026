"""Native XGBoost multiclass model; saved without pickle or scikit-learn."""
from pathlib import Path
import numpy as np
import xgboost as xgb
from .features import FEATURE_NAMES, HistoryFeatures, training_examples


class HouseEdgeForecaster:
    def __init__(self, depth=3, rounds=1500):
        self.settings = dict(objective="multi:softprob", num_class=5, max_depth=depth,
                             eta=0.05, min_child_weight=10, reg_lambda=5,
                             seed=42, nthread=2, tree_method="hist", eval_metric="mlogloss")
        self.rounds = rounds
        self.booster = None

    def fit(self, queries, history, labels):
        X, y, weight = training_examples(queries, history, labels)
        self.training_rows = len(y)
        self.booster = xgb.train(self.settings, xgb.DMatrix(X, label=y, weight=weight,
                                 feature_names=FEATURE_NAMES, nthread=2), num_boost_round=self.rounds)
        return self

    def predict(self, queries, history):
        X = HistoryFeatures(queries, history).matrix(queries)
        values = self.booster.predict(xgb.DMatrix(X, feature_names=FEATURE_NAMES, nthread=2))
        # Convert float32 scores to normalized float64 probabilities for strict CSV checks.
        values = values.astype(np.float64)
        values /= values.sum(axis=1, keepdims=True)
        return {q.query_id: tuple(p) for q, p in zip(queries, values)}

    def save(self, path):
        self.booster.save_model(Path(path))

    @classmethod
    def load(cls, path):
        model = cls()
        model.booster = xgb.Booster(params={"nthread": 2})
        model.booster.load_model(Path(path))
        return model
