"""Run the source-bound gate/layer controls without changing frozen results."""
from _common import ROOT
import argparse,json
from pathlib import Path
import torch
from torch_geometric.seed import seed_everything
from lrr.utils import load_bundle,save_result
from training.train_gcn import prepare_gcn,train_prepared_gcn,evaluate_gcn_test

VARIANTS=['MAIN_GRADED','LOCAL_BINARY','UNGATED','LAST_LAYER_ONLY','UNIFORM_MASSMATCH','TIER2_MASSMATCH','DECAY_LAMBDA_0P5','DECAY_LAMBDA_1P25']


def apply(prepared,variant):
    t=prepared.training;g1=t['gate1'].clone();g2=t['gate2'].clone()
    b0=g1[:,0]==1.;b1=(g1[:,0]>0)&~b0;b2=(g2[:,0]>0)&~b0&~b1
    if variant=='LOCAL_BINARY':g1[t['support1']]=1.;g2[t['support2']]=1.
    elif variant=='UNGATED':g1.fill_(1.);g2.fill_(1.)
    elif variant=='UNIFORM_MASSMATCH':
        g1[t['support1']]=g1.sum()/t['support1'].sum();g2[t['support2']]=g2.sum()/t['support2'].sum()
    elif variant=='TIER2_MASSMATCH':
        g2[b1|b2]=(0.7*b1.sum()+0.3*b2.sum())/(b1.sum()+b2.sum())
    elif variant.startswith('DECAY_LAMBDA_'):
        rho=.5 if variant.endswith('0P5') else 1.25
        g1[b1]=1.-.3*rho;g2[b1]=1.-.3*rho;g2[b2]=1.-.7*rho
    elif variant=='LAST_LAYER_ONLY':
        prepared.model.repair1.W.requires_grad_(False)
        prepared.optimizer=torch.optim.Adam([prepared.model.repair2.W],lr=float(prepared.config.get('lr',.0003)),weight_decay=0.)
    t['gate1'],t['gate2']=g1,g2


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path)
    p.add_argument('--variant',choices=VARIANTS,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args();config=json.loads(a.config.read_text());seed_everything(config['seed'])
    if config['dataset']!='Cora' or config['backbone']!='GCN' or config['deletion_mode']!='Ed_IN' or float(config['deletion_ratio'])!=2.5:
        raise ValueError('controls are source-bound to Cora/GCN/Ed_IN/2.5%')
    if a.output.exists():raise FileExistsError(a.output)
    bundle=load_bundle((a.config.resolve().parent/config['bundle_path']).resolve())
    prepared=prepare_gcn(config,bundle,a.device);apply(prepared,a.variant)
    result=train_prepared_gcn(prepared);result['test_metrics']=evaluate_gcn_test(prepared,bundle['test']);result['variant']=a.variant
    result['selected_repair_state']={k:v.detach().cpu() for k,v in result['selected_repair_state'].items()}
    save_result(a.output,result)


if __name__=='__main__':main()
