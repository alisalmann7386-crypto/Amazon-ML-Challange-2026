"""Copy challenge TSVs from a folder or ZIPs to canonical train/test folders.

Only expected filenames are copied. Archive directory paths are never extracted.
Supports Windows duplicate names such as train_source3(1).tsv.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import zipfile
from pathlib import Path

PATTERN=re.compile(r'^(train|test)_(source[123]|ground_truth)(?:\(\d+\))?\.tsv$',re.I)

def file_stats(path):
    digest=hashlib.sha256();size=0;newlines=0;last=b''
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):
            digest.update(block);size+=len(block);newlines+=block.count(b'\n');last=block[-1:]
    return {'sha256':digest.hexdigest(),'bytes':size,'physical_lines':newlines+bool(last and last!=b'\n')}

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
    manifest={'split':split,'files':{}}
    for name,found in sorted(candidates.items()):
        path,member=found[0];out=destination/name
        if member is None and path.resolve()==out.resolve():
            manifest['files'][name]=file_stats(out);continue
        if out.exists():
            if member is None:
                original = file_stats(path)
            else:
                digest = hashlib.sha256(); size = 0
                with zipfile.ZipFile(path) as z, z.open(member) as stream:
                    for block in iter(lambda: stream.read(8*1024*1024), b''):
                        digest.update(block); size += len(block)
                original = {'sha256': digest.hexdigest(), 'bytes': size}
            existing = file_stats(out)
            if any(existing[key] != original[key] for key in ('sha256', 'bytes')):
                raise ValueError(f'{out}: prepared data differs from input. Use a new RUN_TAG/output directory; nothing was overwritten.')
            manifest['files'][name] = existing
            print(f'Verified existing prepared file: {name}', flush=True)
            continue
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
        if member is None and file_stats(path)!=file_stats(out):
            out.unlink(missing_ok=True)
            raise IOError(f'Copy verification failed for {name}')
        manifest['files'][name]=file_stats(out)
        print(f'Prepared {out}: {out.stat().st_size/1e6:.1f} MB',flush=True)
    manifest_path=destination/'prepare_manifest.json';temporary=manifest_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(manifest,indent=2),encoding='utf-8');os.replace(temporary,manifest_path)
    print(f'Verified byte-for-byte preparation manifest: {manifest_path}',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--split',choices=['train','test'],required=True)
    a=p.parse_args();prepare(a.input,a.output,a.split)
