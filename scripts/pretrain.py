"""Pretrain original backbones with the published training contracts."""
from _common import ROOT
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.seed import seed_everything
from torch_geometric.utils import negative_sampling, to_undirected
from lrr.models.gcn import GCNBackbone
from lrr.models.gat import GATBackbone
from lrr.models.rgcn import RGCNBackbone
from lrr.sampling.graphsaint import WindowsSafeGraphSAINTRandomWalkSampler
from lrr.evaluation.metrics import dt_metrics
from contracts.gat_rng import child_seed, isolated_rng
from contracts import rgcn
from contracts.sampling import isolated_pyg_negative_sampling


def build_model(config,data,device):
    if config["backbone"]=="gat":
        with isolated_rng(int(config["model_seed"])):
            return GATBackbone(data.x.size(1)).to(device)
    if config["backbone"]=="gcn":return GCNBackbone(data.x.size(1)).to(device)
    return RGCNBackbone(data.num_nodes,11).to(device)


def train(config,data,device,epochs=None):
    seed=int(config["seed"]);seed_everything(seed)
    family=config["backbone"]
    count=int(epochs or config["epochs"]);interval=int(config["validation_interval"])
    history=[];best_state=None;best_score=float("-inf")
    if family=="rgcn":
        settings=config["training"]
        loader=WindowsSafeGraphSAINTRandomWalkSampler(data.cpu(),batch_size=settings["graphsaint_batch_size"],
             walk_length=settings["graphsaint_walk_length"],num_steps=settings["graphsaint_num_steps"],num_workers=0)
        rgcn.set_seed(int(config.get("evaluation_seed",42)))
        bank={}
        for split in ["valid","test"]:
            pos=getattr(data,split+"_pos_edge_index");typ=getattr(data,split+"_edge_type")
            bank[split]=(pos,rgcn.negative_sampling_kg(pos,typ),typ)
    else:
        graph=(to_undirected(data.train_pos_edge_index) if family=="gcn" else data.train_pos_edge_index).to(device)
        x=data.x.to(device)
    model=build_model(config,data,device)
    optimizer=torch.optim.Adam(model.parameters(),lr=float(config["lr"]),weight_decay=float(config["weight_decay"]))
    for epoch in range(count):
        model.train()
        if family=="rgcn":
            for batch in loader:
                batch=batch.to(device);positive=batch.edge_index[:,batch.edge_type<11];typ=batch.edge_type[batch.edge_type<11]
                negative=rgcn.negative_sampling_kg(positive,typ)
                z=model(batch.x,batch.edge_index,batch.edge_type)
                logits=torch.cat((model.decode(z,positive,typ),model.decode(z,negative,typ)))
                labels=torch.cat((torch.ones(positive.size(1),device=device),torch.zeros(negative.size(1),device=device)))
                loss=F.binary_cross_entropy_with_logits(logits,labels)
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        else:
            if family=="gat":
                derived=child_seed(config["dataset"],int(config["training_negative_seed"]),"training_negative",epoch)
                negative=isolated_pyg_negative_sampling(edge_index=graph.cpu(),num_nodes=data.num_nodes,
                         num_neg_samples=graph.size(1),derived_seed=derived).to(device)
            else:
                negative=negative_sampling(graph,num_nodes=data.num_nodes,num_neg_samples=graph.size(1),method="sparse",force_undirected=False)
            z=model(x,graph);pos=model.decode(z,graph);neg=model.decode(z,negative)
            loss=F.binary_cross_entropy_with_logits(torch.cat((pos,neg)),torch.cat((torch.ones_like(pos),torch.zeros_like(neg))))
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if (epoch+1)%interval==0 or epoch+1==count:
            model.eval()
            with torch.no_grad():
                if family=="rgcn":
                    z=model(data.x.to(device),data.edge_index.to(device),data.edge_type.to(device));p,n,t=bank["valid"]
                    pos=model.decode(z,p.to(device),t.to(device));neg=model.decode(z,n.to(device),t.to(device))
                else:
                    z=model(x,graph);pos=model.decode(z,data.val_pos_edge_index.to(device));neg=model.decode(z,data.val_neg_edge_index.to(device))
                labels=torch.cat((torch.ones_like(pos),torch.zeros_like(neg)))
                bce=float(F.binary_cross_entropy_with_logits(torch.cat((pos,neg)),labels))
                scores=(pos.sigmoid(),neg.sigmoid()) if family=="gat" else (pos,neg)
                auc,ap=dt_metrics(*(s.cpu().numpy() for s in scores))
            score=-bce if config["selector"]=="min_validation_bce" else (ap if family=="rgcn" else auc)
            history.append(dict(epoch=epoch+1,DT_AUC=auc,DT_AUP=ap,bce=bce))
            tol=1e-12 if family=="gat" else 0.
            if score>best_score+tol:
                best_score=score;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            print(f"epoch={epoch+1} val_auc={auc:.6f} val_ap={ap:.6f}",flush=True)
    return dict(model_state=best_state,config=config,history=history,training_role="Original",test_used_for_selection=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",type=Path,required=True)
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output",type=Path);p.add_argument("--smoke-epochs",type=int)
    a=p.parse_args();config=json.loads(a.config.read_text());output=a.output or ROOT/config["output_path"]
    if output.exists():raise FileExistsError(output)
    data=torch.load(ROOT/config["data_path"],map_location="cpu",weights_only=False)
    result=train(config,data,torch.device(a.device),a.smoke_epochs)
    if a.smoke_epochs:result["formal_result"]=False
    output.parent.mkdir(parents=True,exist_ok=True);torch.save(result,output)


if __name__=="__main__":main()
