"""Exact per-S1 F0.5 plus explicitly defined micro precision and recall."""
import numpy as np
from core import macro_f05

MAX_THRESHOLD = 1.000001

def validate_threshold(value):
    try: threshold=float(value)
    except (TypeError,ValueError) as exc: raise ValueError('Threshold must be numeric') from exc
    if not np.isfinite(threshold) or threshold<0 or threshold>MAX_THRESHOLD:
        raise ValueError(f'Threshold must be finite and between 0 and {MAX_THRESHOLD}')
    return threshold

def evaluate(queries,predictions,candidates):
    truth={q['entity_id']:set(q['matches']) for q in queries}
    if not queries:return {'s1_count':0,'macro_f05':None}
    tp=sum(len(truth[k]&predictions[k]) for k in truth);npred=sum(len(predictions[k]) for k in truth);npos=sum(map(len,truth.values()))
    singles=[k for k,t in truth.items() if not t]
    return {'s1_count':len(queries),'macro_f05':macro_f05(truth,{k:predictions[k] for k in truth}),'micro_precision':tp/npred if npred else 0.,'micro_recall':tp/npos if npos else None,'singleton_accuracy':sum(not predictions[k] for k in singles)/len(singles) if singles else None,'candidate_micro_recall':sum(len(truth[k]&candidates[k]) for k in truth)/npos if npos else None,'average_predicted_matches':npred/len(queries),'false_positives':npred-tp,'false_negatives':npos-tp}

def slices(q):
    script=q['script'];n=len(q['matches'])
    return ['all','country:'+q['country'],'script:'+script,'writing:'+('Latin' if script=='LATIN' else 'non-Latin' if script!='NONE' else 'unknown'),'cardinality:'+('singleton' if n==0 else 'one-match' if n==1 else 'multi-match')]

def breakdown(queries,predictions,candidates):
    groups={}
    for q in queries:
        for key in slices(q):groups.setdefault(key,[]).append(q)
    return {key:evaluate(qs,predictions,candidates) for key,qs in groups.items()}

def threshold_search(queries,qi,y,scores,grid):
    grid=np.asarray(grid,dtype=float)
    if grid.ndim!=1 or not len(grid):raise ValueError('Threshold grid must be a nonempty vector')
    for value in grid:validate_threshold(value)
    counts=np.array([len(q['matches']) for q in queries]);best=(-1.,0.);trace=[]
    for t in grid:
        mask=scores>=t;pred=np.bincount(qi[mask],minlength=len(counts));tp=np.bincount(qi[mask&(y==1)],minlength=len(counts))
        vals=np.zeros(len(counts));single=counts==0;vals[single]=pred[single]==0;non=~single
        vals[non]=1.25*tp[non]/(.25*counts[non]+pred[non]);score=float(vals.mean())
        trace.append({'threshold':float(t),'macro_f05':score});best=max(best,(score,float(t)))
    return best[1],trace
