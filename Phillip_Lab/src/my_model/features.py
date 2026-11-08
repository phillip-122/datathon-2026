"""Features use actual tick offsets and observations strictly before cutoff."""
from collections import defaultdict
import numpy as np
from slot_forecasting.schema import PAYOUTS, index_history

WINDOWS = (8, 24, 64, 256)
FEATURE_NAMES = (["type_steady", "type_balanced", "type_jackpot", "signal", "observations"]
    + [f"lag_{lag}_is_{p}" for lag in range(1, 9) for p in PAYOUTS]
    + [f"window_{w}_{name}" for w in WINDOWS for name in ["count", *[f"p_{p}" for p in PAYOUTS]]]
    + [f"neighbor_{w}_{name}" for w in (8, 24, 64) for name in ["count", *[f"p_{p}" for p in PAYOUTS]]])

TRANSITION_WINDOWS = (32, 128, None)
TRANSITION_STRENGTH = 8.0
FEATURE_NAMES += [f"transition_lag_{lag}_{window or 'all'}_{name}"
                  for lag in range(1, 9) for window in TRANSITION_WINDOWS
                  for name in ("count", "p_0", "p_1", "p_2", "p_5", "p_20")]


def category(value):
    # Match the organizer's categories while predicting all five outcomes.
    return None if value is None else min(value, 2)



class HistoryFeatures:
    def __init__(self, queries, history):
        self.rows = index_history(queries, history)
        self.sequences = {key: {r.tick: r.payout for r in rows} for key, rows in self.rows.items()}
        self.episodes = defaultdict(list)
        for query in queries:
            self.episodes[query.episode_id].append(query.machine_id)

    @staticmethod
    def summary(values):
        return [len(values)] + [values.count(p) / len(values) if values else np.nan for p in PAYOUTS]

    def row(self, query, cutoff=None):
        tick = query.cutoff if cutoff is None else cutoff
        key = (query.episode_id, query.machine_id)
        sequence = self.sequences.get(key, {})
        past = [r for r in self.rows.get(key, []) if r.tick < tick]
        result = [float(query.machine_type == t) for t in ("steady", "balanced", "jackpot")]
        result += [query.signal, len(past)]
        for lag in range(1, 9):
            value = sequence.get(tick - lag)
            result += [float(value == p) if value is not None else np.nan for p in PAYOUTS]
        for window in WINDOWS:
            result += self.summary([r.payout for r in past if r.tick >= tick - window])
        for window in (8, 24, 64):
            values = [r.payout for machine in self.episodes[query.episode_id] if machine != query.machine_id
                      for r in self.rows.get((query.episode_id, machine), [])
                      if tick - window <= r.tick < tick]
            result += self.summary(values)
        # Historical transition outcomes are strictly before this example's tick.
        for lag in range(1, 9):
            wanted = category(sequence.get(tick - lag))
            for window in TRANSITION_WINDOWS:
                eligible = [r for r in past if window is None or r.tick >= tick - window]
                prior_counts = np.array([sum(r.payout == p for r in eligible) for p in PAYOUTS], dtype=float)
                prior = (prior_counts + 0.5) / (len(eligible) + 2.5)
                if wanted is None:
                    result += [0.0] + [np.nan] * 5
                    continue
                matched = [r.payout for r in eligible
                           if category(sequence.get(r.tick - lag)) == wanted]
                counts = np.array([matched.count(p) for p in PAYOUTS], dtype=float)
                probs = (counts + TRANSITION_STRENGTH * prior) / (len(matched) + TRANSITION_STRENGTH)
                result += [len(matched)] + probs.tolist()
        return result

    def matrix(self, queries):
        return np.asarray([self.row(q) for q in queries], dtype=np.float32)


def training_examples(queries, history, labels, stride=4):
    """Historical observed payouts become labels, never inputs at their own tick."""
    features = HistoryFeatures(queries, history)
    rows, answers, weights = [], [], []
    for query in queries:
        machine_rows = features.rows.get((query.episode_id, query.machine_id), [])
        examples = [(r.tick, r.payout) for r in machine_rows if r.tick >= 32][::stride]
        examples.append((query.cutoff, labels[query.query_id]))
        for tick, outcome in examples:
            rows.append(features.row(query, tick))
            answers.append(PAYOUTS.index(outcome))
            # Each machine contributes the same total training weight.
            weights.append(32.0 / len(examples))
    return np.asarray(rows, dtype=np.float32), np.asarray(answers), np.asarray(weights)
