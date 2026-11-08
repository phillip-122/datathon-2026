"""Regularized categorical lag models and a small neural sequence classifier."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax, logsumexp
from slot_forecasting.schema import PAYOUTS, index_history


def sequence_features(sequence, tick, query):
    # Each lag has five payout indicators plus an explicit unknown indicator.
    x = [1.0, query.signal] + [float(query.machine_type == t) for t in ('steady','balanced','jackpot')]
    for lag in range(1, 9):
        value = sequence.get(tick-lag)
        x += [float(value == p) for p in PAYOUTS] + [float(value is None)]
    return np.asarray(x)


def examples(queries, history, labels=None, stride=1):
    indexed = index_history(queries, history)
    result = []
    for q in queries:
        rows = indexed.get((q.episode_id,q.machine_id), [])
        sequence = {r.tick:r.payout for r in rows}
        available = [(r.tick,r.payout) for r in rows if r.tick >= 16][::stride]
        if labels is not None:
            available.append((q.cutoff,labels[q.query_id]))
        X = np.asarray([sequence_features(sequence,t,q) for t,p in available]).reshape(-1,53)
        y = np.asarray([PAYOUTS.index(p) for t,p in available],dtype=int)
        ages = np.asarray([q.cutoff-1-t for t,p in available])
        result.append((q, X, y, ages, sequence_features(sequence,q.cutoff,q)))
    return result


def fit_softmax(X,y,weights,anchor,strength,iterations=80):
    if not len(y):
        return anchor.copy()
    def objective(flat):
        W = flat.reshape(anchor.shape)
        scores = X @ W
        loss = np.sum(weights*(logsumexp(scores,axis=1)-scores[np.arange(len(y)),y]))
        delta = W-anchor
        residual = softmax(scores,axis=1)
        residual[np.arange(len(y)),y] -= 1
        gradient = X.T @ (weights[:,None]*residual) + strength*delta
        return loss+strength/2*np.sum(delta*delta), gradient.ravel()
    fitted = minimize(objective,anchor.ravel(),jac=True,method='L-BFGS-B',
                      options={'maxiter':iterations,'ftol':1e-8})
    if not np.isfinite(fitted.fun):
        raise RuntimeError('Nonfinite local model objective')
    return fitted.x.reshape(anchor.shape)


def local_predictions(examples_,global_weights,strength,half_life,hierarchical):
    result = {}
    for q,X,y,ages,current in examples_:
        if hierarchical:
            anchor = global_weights
        else:
            anchor = np.zeros_like(global_weights)
            counts = np.bincount(y,minlength=5)+2
            anchor[0] = np.log(counts/counts.sum())
        weights = 2.0**(-ages/half_life)
        W = fit_softmax(X,y,weights,anchor,strength)
        result[q.query_id] = softmax(current @ W)
    return result


def train_shared(examples_,epochs=100):
    X = np.concatenate([e[1] for e in examples_])
    y = np.concatenate([e[2] for e in examples_])
    weights = np.concatenate([np.full(len(e[2]),32/len(e[2])) for e in examples_])
    linear = fit_softmax(X,y,weights,np.zeros((53,5)),32,iterations=150)
    rng = np.random.default_rng(42)
    params = [rng.normal(0,0.12,(53,64)), np.zeros(64), rng.normal(0,0.12,(64,5)),np.zeros(5)]
    moments = [np.zeros_like(p) for p in params]
    squares = [np.zeros_like(p) for p in params]
    step = 0
    for epoch in range(epochs):
        for ids in np.array_split(rng.permutation(len(y)),max(1,len(y)//512)):
            a = X[ids]; hidden = np.tanh(a @ params[0]+params[1])
            residual = softmax(hidden @ params[2]+params[3],axis=1)
            residual[np.arange(len(ids)),y[ids]] -= 1
            residual *= weights[ids,None]/weights[ids].sum()
            dh = (residual @ params[2].T)*(1-hidden*hidden)
            grads = [a.T@dh+0.001*params[0],dh.sum(0),hidden.T@residual+0.001*params[2],residual.sum(0)]
            step += 1
            for i,g in enumerate(grads):
                moments[i] = .9*moments[i]+.1*g
                squares[i] = .999*squares[i]+.001*g*g
                params[i] -= .001*(moments[i]/(1-.9**step))/(np.sqrt(squares[i]/(1-.999**step))+1e-8)
        if (epoch+1)%20 == 0:
            print(f'Neural sequence model epoch {epoch+1}/{epochs}',flush=True)
    return linear,params


def neural_predictions(examples_,params):
    X = np.asarray([e[4] for e in examples_])
    probs = softmax(np.tanh(X@params[0]+params[1])@params[2]+params[3],axis=1)
    return {e[0].query_id:p for e,p in zip(examples_,probs)}
