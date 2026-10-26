"""Download hash-pinned paper assets; default: Original checkpoints only."""
import argparse, hashlib, json, shutil, tarfile, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository',default='LWH-cmd/LRR');p.add_argument('--tag',default='paper-v1.0.0')
    p.add_argument('--asset',action='append',help='Download a named archive; repeat to select several')
    p.add_argument('--all',action='store_true',help='Download every declared archive (large)')
    p.add_argument('--list',action='store_true',help='List names and byte sizes without downloading')
    a=p.parse_args();cache=ROOT/'.cache/downloads';cache.mkdir(parents=True,exist_ok=True)
    assets=json.loads((ROOT/'manifests/release_assets.json').read_text())
    if a.list:
        for asset in assets:print(f"{asset['name']}\t{asset['bytes']}\t{asset['role']}")
        return
    selected=set(a.asset or ['lrr-pretrained-models.tar.gz'])
    if a.all:selected={asset['name'] for asset in assets}
    unknown=selected-{asset['name'] for asset in assets}
    if unknown:p.error('unknown asset: '+', '.join(sorted(unknown)))
    for asset in assets:
        if asset['name'] not in selected:continue
        path=cache/asset['name']
        if not path.exists():
            url=f"https://github.com/{a.repository}/releases/download/{a.tag}/{asset['name']}"
            with urllib.request.urlopen(url) as response,path.open('wb') as stream:shutil.copyfileobj(response,stream)
        if sha(path)!=asset['sha256']:raise ValueError(f'asset checksum mismatch: {path}')
        with tarfile.open(path) as archive:
            for member in archive.getmembers():
                target=(ROOT/member.name).resolve()
                if not target.is_relative_to(ROOT.resolve()) or not member.isfile():raise ValueError('unsafe asset member')
                if target.exists():continue
                target.parent.mkdir(parents=True,exist_ok=True)
                with archive.extractfile(member) as stream,target.open('wb') as out:shutil.copyfileobj(stream,out)
    if 'lrr-pretrained-models.tar.gz' in selected:
        for cell in json.loads((ROOT/'manifests/experiments.json').read_text()):
            if sha(ROOT/cell['checkpoint'])!=cell['checkpoint_sha256']:raise ValueError('checkpoint mismatch')
    print('selected archives downloaded and verified')


if __name__=='__main__':main()
