"""Estimate structure from permitted tuning targets, then infer from past payouts."""
from itertools import combinations
import numpy as np
from scipy.special import expit
from slot_forecasting.schema import index_history


def learn_constraints(targets,queries):
    result={}
    for kind in ('steady','balanced','jackpot'):
        P=np.asarray([t.probabilities for t in targets if queries[t.query_id].machine_type==kind])
        if not len(P): raise ValueError('Each training fold needs all machine types')
        high=P[:,2:]; c=np.mean(high/high.sum(1,keepdims=True),axis=0)
        c/=c.sum()
        share=P[:,1]/np.maximum(1-P[:,0],1e-12)
        midpoint=(share.min()+share.max())/2
        low=share[share<=midpoint].mean(); upper=share[share>midpoint].mean()
        if not np.isfinite(upper): raise ValueError('Need both observed payout regimes')
        result[kind]={'high_ratio':c.tolist(),'one_share':[float(low),float(upper)]}
    return result


def nearest_simplex(point,vertices):
    """Exact Euclidean projection onto a triangle, including edges and vertices."""
    vertices=np.asarray(vertices)
    best=None; best_loss=float('inf')
    for size in (1,2,3):
        for support in combinations(range(3),size):
            V=vertices[list(support)]
            gram=V@V.T
            block=np.block([[gram,np.ones((size,1))],[np.ones((1,size)),np.zeros((1,1))]])
            rhs=np.r_[V@point,1.]
            coeff=np.linalg.solve(block,rhs)[:-1]
            if np.min(coeff)<-1e-10: continue
            coeff=np.maximum(coeff,0); coeff/=coeff.sum()
            projected=coeff@V; loss=np.sum((projected-point)**2)
            if loss<best_loss: best_loss=loss; best=projected
    return best/best.sum()


def vertices_for(info,restricted):
    c=np.asarray(info['high_ratio'])
    vertices=[np.array([1.,0,0,0,0])]
    for r in info['one_share'] if restricted else (0.,1.):
        vertices.append(np.r_[0.,r,(1-r)*c])
    return np.asarray(vertices)


def alternatives(queries,history,raw,constraints):
    history=index_history(queries,history)
    output={name:{} for name in ('high_projection','regime_projection','bayes_24','bayes_64','bayes_all')}
    for q in queries:
        p=np.asarray(raw[q.query_id]); info=constraints[q.machine_type]
        output['high_projection'][q.query_id]=nearest_simplex(p,vertices_for(info,False))
        output['regime_projection'][q.query_id]=nearest_simplex(p,vertices_for(info,True))
        low,high=info['one_share']; c=np.asarray(info['high_ratio'])
        for half in (24,64,None):
            # Conditional positive-payout likelihood. Zero payouts do not count
            # as evidence about the share of 1x among positive payouts.
            log_odds=0.
            for row in history.get((q.episode_id,q.machine_id),[]):
                if row.payout==0: continue
                weight=1. if half is None else 2.**(-(q.cutoff-1-row.tick)/half)
                log_odds+=weight*(np.log(high/low) if row.payout==1 else np.log((1-high)/(1-low)))
            posterior=expit(log_odds)
            r=(1-posterior)*low+posterior*high
            # Preserve uncertainty by mixing regimes, not choosing a hard state.
            win=np.r_[0.,r,(1-r)*c]
            zero=np.array([1.,0,0,0,0]); direction=win-zero
            w=np.clip(np.dot(p-zero,direction)/np.dot(direction,direction),0,1)
            output[f'bayes_{half or "all"}'][q.query_id]=zero+w*direction
    return output


def blend(left,right,weight):
    return {k:tuple(weight*np.asarray(left[k])+(1-weight)*np.asarray(right[k])) for k in left}
