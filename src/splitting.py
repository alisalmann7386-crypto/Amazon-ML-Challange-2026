import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from io_utils import connect,sample_queries,linked_groups,dump

def split(index,cfg):
    db=connect(index)
    if json.loads((Path(index)/'catalog.json').read_text())['split']!='train':raise ValueError('Labels are required for train splitting')
    qs=sample_queries(db,cfg['sample_size'],cfg['seed']);groups=np.asarray(linked_groups(db,qs));db.close()
    if len(set(groups))<5:raise ValueError('Need at least five independent entity groups')
    ti,rest=next(GroupShuffleSplit(n_splits=1,test_size=.4,random_state=cfg['seed']).split(qs,groups=groups))
    ci,hi=next(GroupShuffleSplit(n_splits=1,test_size=.5,random_state=cfg['seed']+1).split(rest,groups=groups[rest]))
    return {'train':[qs[i] for i in ti],'calibration':[qs[rest[i]] for i in ci],'holdout':[qs[rest[i]] for i in hi]}
