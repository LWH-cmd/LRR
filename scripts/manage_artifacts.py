"""List, verify or restore the existing source-bound paper artifacts. No training."""
import argparse, hashlib, json, os, shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix',default='',help='Original workspace-relative source path prefix')
    parser.add_argument('--category',choices=['sources','records','checkpoints','inputs'])
    actions=parser.add_mutually_exclusive_group()
    actions.add_argument('--verify',action='store_true',help='Check extracted files against the manifest')
    actions.add_argument('--restore',action='store_true',help='Restore original relative paths into a separate directory')
    parser.add_argument('--destination',type=Path,default=ROOT/'outputs/source_layout')
    args=parser.parse_args()
    rows=json.loads((ROOT/'manifests/artifact_catalog.json').read_text(encoding='utf-8'))
    rows=[r for r in rows if r['source'].startswith(args.prefix) and (not args.category or r['category']==args.category)]
    verified={}
    for row in rows:
        if not (args.verify or args.restore):
            print(f"{row['source']}\t{row['archive']}\t{row['bytes']}")
            continue
        source=ROOT/row['release_path']
        if not source.is_file():
            raise FileNotFoundError(f"Download and extract {row['archive']} first: {source}")
        if source not in verified:verified[source]=sha(source)
        if verified[source]!=row['sha256']:raise ValueError(f'checksum mismatch: {source}')
        if args.restore:
            base=args.destination.resolve();target=(base/row['source']).resolve()
            if not target.is_relative_to(base):raise ValueError('unsafe original path')
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if sha(target)!=row['sha256']:raise FileExistsError(f'refusing to overwrite: {target}')
                continue
            try:os.link(source,target)
            except OSError:shutil.copyfile(source,target)
    if args.verify or args.restore:print(f'{len(rows)} source paths verified')

if __name__=='__main__':main()
