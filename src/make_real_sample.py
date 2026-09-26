"""Create a deterministic, label-complete real-data sample for bounded experiments."""
import argparse,csv,hashlib,heapq,json
from pathlib import Path
SOURCE_FIELDS=['entity_id','business_name','business_address','country']
def key(seed,value):return int.from_bytes(hashlib.blake2b(f'{seed}:{value}'.encode(),digest_size=8).digest(),'big')
def select_source1(path,count,seed):
    heap=[];rows=0
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f,delimiter='\t',strict=True)
        if reader.fieldnames!=SOURCE_FIELDS:raise ValueError(f'{path}: unexpected columns')
        for row in reader:
            rows+=1;k=key(seed,row['entity_id']);item=(-k,row['entity_id'],row)
            if len(heap)<count:heapq.heappush(heap,item)
            elif k < -heap[0][0]:heapq.heapreplace(heap,item)
    return sorted((x[2] for x in heap),key=lambda r:r['entity_id']),rows
def select_truth(path,qids):
    selected=[];targets=set();rows=0
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f,delimiter='\t',strict=True)
        if reader.fieldnames!=['source1_entity_id','matched_entity_ids']:raise ValueError(f'{path}: unexpected columns')
        for row in reader:
            rows+=1
            if row['source1_entity_id'] in qids:selected.append(row);targets.update(x for x in row['matched_entity_ids'].split(',') if x)
    if {r['source1_entity_id'] for r in selected}!=qids:raise ValueError('Selected Source 1 IDs are missing labels')
    return sorted(selected,key=lambda r:r['source1_entity_id']),targets,rows
def select_targets(path,positives,distractors,seed):
    heap=[];positive_rows={};rows=0
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f,delimiter='\t',strict=True)
        if reader.fieldnames!=SOURCE_FIELDS:raise ValueError(f'{path}: unexpected columns')
        for row in reader:
            rows+=1;eid=row['entity_id']
            if eid in positives:positive_rows[eid]=row;continue
            k=key(seed,eid);item=(-k,eid,row)
            if len(heap)<distractors:heapq.heappush(heap,item)
            elif k < -heap[0][0]:heapq.heapreplace(heap,item)
    missing=positives-set(positive_rows)
    if missing:raise ValueError(f'{path}: {len(missing)} labeled targets missing')
    return sorted(list(positive_rows.values())+[x[2] for x in heap],key=lambda r:r['entity_id']),rows
def write(path,fields,rows):
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
def run(data,output,queries,distractors,seed):
    data=Path(data);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    s1,s1_total=select_source1(data/'train_source1.tsv',queries,seed);qids={r['entity_id'] for r in s1}
    truth,positive_ids,truth_total=select_truth(data/'train_ground_truth.tsv',qids)
    by_source={2:{x for x in positive_ids if x.startswith('S2-')},3:{x for x in positive_ids if x.startswith('S3-')}};selected={};totals={1:s1_total}
    for source in (2,3):selected[source],totals[source]=select_targets(data/f'train_source{source}.tsv',by_source[source],distractors,seed+source)
    write(output/'train_source1.tsv',SOURCE_FIELDS,s1)
    for source in (2,3):write(output/f'train_source{source}.tsv',SOURCE_FIELDS,selected[source])
    write(output/'train_ground_truth.tsv',['source1_entity_id','matched_entity_ids'],truth)
    manifest={'scope':'deterministic sampled-real experiment; every selected query label retained; target catalog contains all labeled targets plus hashed real distractors','seed':seed,'full_rows':{'source1':s1_total,'source2':totals[2],'source3':totals[3],'ground_truth':truth_total},'sample_rows':{'source1':len(s1),'source2':len(selected[2]),'source3':len(selected[3]),'ground_truth':len(truth)},'positive_targets':{'source2':len(by_source[2]),'source3':len(by_source[3])},'requested_distractors_per_target_source':distractors}
    (output/'sample_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps(manifest,indent=2));return manifest
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--queries',type=int,default=2000);p.add_argument('--distractors-per-source',type=int,default=100000);p.add_argument('--seed',type=int,default=42);a=p.parse_args();run(a.data,a.output,a.queries,a.distractors_per_source,a.seed)
