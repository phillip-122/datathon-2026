"""Production Model Class for House Edge: HistGradientBoosting with Cutoff-Safe Dynamics."""
import json
import math
from collections import defaultdict
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.schema import PAYOUTS, ValidationError, probabilities

from .features import build_advanced_features, PAYOUT_TO_IDX

class HGBForecaster:
    def __init__(
        self,
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=60,
        min_samples_leaf=25,
        l2_regularization=2.0,
        step_ticks=6,
        random_state=42
    ):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.step_ticks = step_ticks
        self.random_state = random_state
        
        self.base_forecaster = FrequencyForecaster("pooled_recent", prior_strength=10.0, half_life=16.0)
        self.hgb = None

    def fit(self, queries, history, labels=None):
        queries = tuple(queries)
        history = tuple(history)
        
        # 1. Fit baseline forecaster
        self.base_forecaster.fit(queries, history, labels)
        
        # 2. Index history by episode and machine
        ep_hist = defaultdict(lambda: defaultdict(dict))
        for h in history:
            ep_hist[h.episode_id][h.machine_id][h.tick] = h.payout
            
        # 3. Extract multi-cutoff training examples
        X_train = []
        y_train = []
        for q in queries:
            ep_seqs = ep_hist[q.episode_id]
            m_seq = ep_seqs[q.machine_id]
            peers = [ep_seqs[m] for m in ep_seqs if m != q.machine_id]
            
            # Label at query cutoff if available
            if labels is not None and q.query_id in labels:
                f = build_advanced_features(q, m_seq, peers, q.cutoff, self.base_forecaster)
                X_train.append(f)
                y_train.append(PAYOUT_TO_IDX[labels[q.query_id]])
                
            # Sample earlier historical cutoffs
            for t in range(30, q.cutoff, self.step_ticks):
                if t in m_seq:
                    sub_seq = {tick: p for tick, p in m_seq.items() if tick < t}
                    if len(sub_seq) >= 8:
                        sub_peers = [{tick: p for tick, p in p_seq.items() if tick < t} for p_seq in peers]
                        f = build_advanced_features(q, sub_seq, sub_peers, t, self.base_forecaster)
                        X_train.append(f)
                        y_train.append(PAYOUT_TO_IDX[m_seq[t]])
                        
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int32)
        
        # 4. Fit HistGradientBoostingClassifier
        self.hgb = HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            random_state=self.random_state
        )
        self.hgb.fit(X_train, y_train)
        return self

    def predict(self, queries, history):
        if self.hgb is None or self.base_forecaster.global_prior is None:
            raise ValidationError("Model must be fitted or loaded before predicting.")
            
        queries = tuple(queries)
        history = tuple(history)
        
        ep_hist = defaultdict(lambda: defaultdict(dict))
        for h in history:
            ep_hist[h.episode_id][h.machine_id][h.tick] = h.payout
            
        X = []
        for q in queries:
            ep_seqs = ep_hist[q.episode_id]
            m_seq = ep_seqs[q.machine_id]
            peers = [ep_seqs[m] for m in ep_seqs if m != q.machine_id]
            f = build_advanced_features(q, m_seq, peers, q.cutoff, self.base_forecaster)
            X.append(f)
            
        X = np.array(X, dtype=np.float32)
        probs = self.hgb.predict_proba(X)
        
        predictions = {}
        for i, q in enumerate(queries):
            raw = probs[i]
            # Ensure probabilities sum strictly to 1.0 within 1e-6
            tot = math.fsum(raw)
            normalized = tuple(float(v / tot) for v in raw)
            predictions[q.query_id] = probabilities(normalized, q.query_id)
            
        return predictions

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        meta = {
            "model_type": "HGBForecaster",
            "max_iter": self.max_iter,
            "learning_rate": self.learning_rate,
            "max_leaf_nodes": self.max_leaf_nodes,
            "min_samples_leaf": self.min_samples_leaf,
            "l2_regularization": self.l2_regularization,
            "step_ticks": self.step_ticks,
            "random_state": self.random_state,
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self.base_forecaster.save(directory / "base_forecaster.json")
        joblib.dump(self.hgb, directory / "hgb_model.joblib", compress=3)

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        model = cls(
            max_iter=meta["max_iter"],
            learning_rate=meta["learning_rate"],
            max_leaf_nodes=meta["max_leaf_nodes"],
            min_samples_leaf=meta["min_samples_leaf"],
            l2_regularization=meta["l2_regularization"],
            step_ticks=meta["step_ticks"],
            random_state=meta["random_state"]
        )
        model.base_forecaster = FrequencyForecaster.load(directory / "base_forecaster.json")
        model.hgb = joblib.load(directory / "hgb_model.joblib")
        return model
