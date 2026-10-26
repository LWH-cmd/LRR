"""Run one or all paper LRR cells; explicitly choose frozen or newly pretrained originals."""
from _common import ROOT
import argparse,json,subprocess,sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cell')
    p.add_argument('--all',action='store_true');p.add_argument('--list',action='store_true')
    p.add_argument('--original',choices=['published','retrained'],default='published')
    a=p.parse_args();cells=json.loads((ROOT/'manifests/experiments.json').read_text())
    if a.list:
        print('\n'.join(c['cell_id'] for c in cells));return
    if not (a.cell or a.all):p.error('select --cell, --all, or --list')
    selected=[c for c in cells if a.all or c['cell_id']==a.cell]
    if not selected:raise ValueError('unknown cell')
    for c in selected:
        config=ROOT/'configs/main'/(c['cell_id']+'.json');cfg=json.loads(config.read_text())
        output=(config.parent/cfg['output_path']).resolve()
        if output.exists():raise FileExistsError(f'existing run output: {output}')
        family=cfg['backbone'].lower();bundle=(config.parent/cfg['bundle_path']).resolve()
        original=(ROOT/c['checkpoint'] if a.original=='published' else ROOT/f"outputs/pretrain/{family}_{cfg['dataset'].lower()}_seed{cfg['seed']}.pt")
        if not bundle.exists():subprocess.run([sys.executable,str(ROOT/'scripts/build_bundle.py'),'--config',str(config),'--checkpoint',str(original)],check=True)
        subprocess.run([sys.executable,str(ROOT/'scripts'/f'run_{family}.py'),'--config',str(config),'--checkpoint',str(original)],check=True)


if __name__=='__main__':main()
