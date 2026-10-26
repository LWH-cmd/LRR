"""Evaluate a published main LRR selected state on a prepared paper bundle."""
import argparse,json,subprocess,sys
from pathlib import Path
import torch
from load_published import ROOT,load_model
from lrr.evaluation.evaluator import GCNValidationBundle,RGCNValidationBundle,evaluate_gcn,evaluate_gat,evaluate_rgcn

@torch.no_grad()
def evaluate(model,bundle,family,device='cpu'):
    model=model.to(device).eval()
    if family=='rgcn':
        values={k:v.to(device) for k,v in bundle['test'].items()}
        bank=RGCNValidationBundle(**values)
        z=model(bank.x,bank.edge_index,bank.edge_type,bank.gate1,bank.gate2)
        return evaluate_rgcn(model,z,bank)
    training=bundle['training'];x=training['x'].to(device);dr=training['dr_mp_edge_index'].to(device)
    z=model(x,dr,training['gate1'].to(device),training['gate2'].to(device))
    bank=GCNValidationBundle(**bundle['test'])
    if family=='gat':return evaluate_gat(model,z,model.backbone(x,dr),bank)
    return evaluate_gcn(model,z,bank)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cell',required=True);p.add_argument('--bundle',type=Path)
    p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path)
    a=p.parse_args();rows=json.loads((ROOT/'manifests/checkpoints.json').read_text())
    entry=next(r for r in rows if r['group']=='main' and a.cell in r['bindings'])
    config=json.loads((ROOT/'configs/main'/f'{a.cell}.json').read_text())
    cell=next(c for c in json.loads((ROOT/'manifests/experiments.json').read_text()) if c['cell_id']==a.cell)
    path=a.bundle or ROOT/f'data/bundles/{a.cell}.pt'
    if not path.exists():
        subprocess.run([sys.executable,str(ROOT/'scripts/build_bundle.py'),'--config',str(ROOT/'configs/main'/f'{a.cell}.json'),
                        '--checkpoint',str(ROOT/cell['checkpoint']),'--output',str(path)],check=True)
    bundle=torch.load(path,map_location='cpu',weights_only=False)
    if bundle.get('source_cell_id')!=a.cell:raise ValueError('bundle belongs to a different paper cell')
    if bundle.get('checkpoint_sha256')!=cell['checkpoint_sha256']:raise ValueError('bundle must use the paper Original checkpoint')
    model,_=load_model(entry);result=evaluate(model,bundle,config['backbone'].lower(),a.device)
    text=json.dumps(dict(cell_id=a.cell,checkpoint_sha256=entry['sha256'],metrics=result),indent=2)
    if a.output:
        with a.output.open('x',encoding='utf-8') as f:f.write(text)
    print(text)

if __name__=='__main__':main()
