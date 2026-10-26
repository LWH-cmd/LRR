"""Load the unchanged selected LRR checkpoint into the released model classes."""
import argparse,hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lrr.models.gcn import GCNBackbone,FullRankGCN
from lrr.models.gat import GATBackbone,FullRankGAT
from lrr.models.rgcn import RGCNBackbone,FullRankRGCN

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def load_model(entry,path=None,original_path=None):
    path=Path(path) if path else ROOT/entry['release_path']
    if sha(path)!=entry['sha256']:raise ValueError('selected checkpoint checksum mismatch')
    payload=torch.load(path,map_location='cpu',weights_only=False)
    state=payload.get('model_state',payload.get('operator_state',payload.get('state_dict',payload)))
    if 'operator_state' in payload:
        cell=next(c for c in json.loads((ROOT/'manifests/experiments.json').read_text()) if c['cell_id'] in entry['bindings'])
        original_path=Path(original_path) if original_path else ROOT/cell['checkpoint']
        if sha(original_path)!=cell['checkpoint_sha256']:raise ValueError('Original checkpoint checksum mismatch')
        original=torch.load(original_path,map_location='cpu',weights_only=False)
        original=original.get('model_state',original.get('state_dict',original))
        model=FullRankRGCN(RGCNBackbone(40943,11));model.backbone.load_state_dict(original,strict=True)
        model.repair1.load_state_dict({'W':state['repair1.W']},strict=True)
        model.repair2.load_state_dict({'W':state['repair2.W']},strict=True)
    else:
        weight=next(v for k,v in state.items() if k.endswith('conv1.lin.weight'))
        if not any(k.startswith('repair') for k in state):
            if not any(b.startswith('formulation:C:') for b in entry['bindings']):raise ValueError('not a registered formulation control')
            from types import SimpleNamespace
            from control_models.gnndelete_matched import GCNDeleteMatched
            args=SimpleNamespace(in_dim=weight.size(1),hidden_dim=128,out_dim=64,input_mode='feature_matrix')
            model=GCNDeleteMatched(args,input_mode='feature_matrix');model.load_state_dict(state,strict=True)
            model.eval();return model,payload
        gat=any('att_src' in k for k in state)
        backbone=GATBackbone(weight.size(1)) if gat else GCNBackbone(weight.size(1))
        model=FullRankGAT(backbone) if gat else FullRankGCN(backbone)
        normalized={('backbone.'+k if k.startswith(('conv1.','conv2.','node_emb.')) else k):v for k,v in state.items()}
        model.load_state_dict(normalized,strict=True)
    model.eval();return model,payload

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--id');p.add_argument('--list',action='store_true')
    a=p.parse_args();rows=json.loads((ROOT/'manifests/checkpoints.json').read_text())
    if a.list:
        for r in rows:print(';'.join(r['bindings'])+'\t'+r['archive'])
        return
    selected=[r for r in rows if a.id in r['bindings']]
    if len(selected)!=1:p.error('select one exact binding ID from --list')
    model,_=load_model(selected[0]);print(f'strict loading passed: {type(model).__name__}')

if __name__=='__main__':main()
