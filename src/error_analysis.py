"""Export observed holdout mistakes; no guessed test labels."""
import csv
from pathlib import Path
from io_utils import get_rows,batches
from similarity import numbers,jaro_winkler,token_set

def export(retriever,queries,candidate_lists,predictions,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    names=['false_positives','false_negatives','retrieval_misses','singleton_false_merges','numeric_conflicts','high_name_low_address','low_name_high_address','cross_script_failures','transliteration_successes','transliteration_failures']
    files={n:(out/(n+'.tsv')).open('w',encoding='utf-8',newline='') for n in names}
    writers={n:csv.writer(f,delimiter='\t') for n,f in files.items()}
    for w in writers.values():w.writerow(['source1_entity_id','target_entity_id','true_match','predicted','retrieved','source_script','target_script'])
    try:
        for q,cs in zip(queries,candidate_lists):
            truth=set(q['matches']);pred=predictions[q['entity_id']];by_id={c['target']['entity_id']:c for c in cs};targets={k:c['target'] for k,c in by_id.items()}
            missing=truth-set(targets)
            for part in batches(missing,500):
                import json
                for row in retriever.db.execute(f'SELECT entity_id,data FROM targets WHERE entity_id IN ({",".join("?" for _ in part)})',part):targets[row[0]]=json.loads(row[1])
            for tid in sorted(truth|pred):
                t=targets[tid];c=by_id.get(tid);yes=tid in truth;selected=tid in pred
                row=[q['entity_id'],tid,int(yes),int(selected),int(c is not None),q['script'],t['script']]
                groups=[]
                if selected and not yes:groups.append('false_positives')
                if yes and not selected:groups.append('false_negatives')
                if yes and c is None:groups.append('retrieval_misses')
                if selected and not truth:groups.append('singleton_false_merges')
                a,b=numbers(q['business_address_norm']),numbers(t['business_address_norm'])
                if a and b and not a&b:groups.append('numeric_conflicts')
                ns=jaro_winkler(q['business_name_norm'],t['business_name_norm']);ads=token_set(q['business_address_norm'],t['business_address_norm'])
                if ns>=.85 and ads<.4:groups.append('high_name_low_address')
                if ns<.4 and ads>=.8:groups.append('low_name_high_address')
                if q['script']!=t['script'] and yes!=selected:groups.append('cross_script_failures')
                if c:
                    trans=any(ch.endswith(('tname','taddress')) for ch in c['ranks'])
                    original=any(not ch.endswith(('tname','taddress')) for ch in c['ranks'])
                    if trans and not original:groups.append('transliteration_successes' if yes and selected else 'transliteration_failures')
                for g in groups:writers[g].writerow(row)
    finally:
        for f in files.values():f.close()
