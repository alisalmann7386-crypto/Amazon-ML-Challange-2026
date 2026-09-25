"""Copy challenge TSVs from a folder or ZIPs to canonical train/test folders.

Only expected filenames are copied. Archive directory paths are never extracted.
Supports Windows duplicate names such as train_source3(1).tsv.
"""
import argparse
import re
import shutil
import zipfile
from pathlib import Path

PATTERN=re.compile(r'^(train|test)_(source[123]|ground_truth)(?:\(\d+\))?\.tsv$',re.I)

def canonical(name):
    match=PATTERN.fullmatch(Path(name.replace('\\','/')).name)
    if not match or (match[1].lower()=='test' and match[2].lower()=='ground_truth'):
        return None
    return f'{match[1].lower()}_{match[2].lower()}.tsv'

def prepare(source,destination,split):
    source,destination=Path(source),Path(destination)
    expected={f'{split}_source{i}.tsv' for i in (1,2,3)}
    if split=='train': expected.add('train_ground_truth.tsv')
    candidates={name:[] for name in expected}
    files=sorted(source.rglob('*')) if source.is_dir() else [source]
    for path in files:
        if not path.is_file(): continue
        name=canonical(path.name)
        if name in expected: candidates[name].append((path,None))
        elif path.suffix.lower()=='.zip':
            with zipfile.ZipFile(path) as z:
                for info in z.infolist():
                    name=canonical(info.filename)
                    if name in expected and not info.is_dir(): candidates[name].append((path,info.filename))
    for name,found in candidates.items():
        if len(found)!=1:
            raise ValueError(f'{name}: found {len(found)} copies. Put exactly one copy of each required file in the input folder (TSV or ZIP).')
    destination.mkdir(parents=True,exist_ok=True)
    for name,found in sorted(candidates.items()):
        path,member=found[0];out=destination/name
        if member is None and path.resolve()==out.resolve(): continue
        if out.exists(): raise ValueError(f'{out} already exists; use a new destination or reuse existing prepared data')
        temp=out.with_suffix('.tsv.copying')
        try:
            with temp.open('wb') as target:
                if member:
                    with zipfile.ZipFile(path) as z,z.open(member) as stream:
                        shutil.copyfileobj(stream,target,8*1024*1024)
                else:
                    with path.open('rb') as stream: shutil.copyfileobj(stream,target,8*1024*1024)
            temp.replace(out)
        finally:
            if temp.exists():temp.unlink()
        print(f'Prepared {out}: {out.stat().st_size/1e6:.1f} MB',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--split',choices=['train','test'],required=True)
    a=p.parse_args();prepare(a.input,a.output,a.split)
