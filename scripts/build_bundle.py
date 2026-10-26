"""Construct formal request/support/sampling inputs from prepared raw data."""
from _common import ROOT
import argparse, hashlib, json
from pathlib import Path
import torch
import numpy as np
from torch_geometric.seed import seed_everything
from torch_geometric.utils import k_hop_subgraph
from contracts.lrr_runtime import reproduce_delete_gnn_df_indices, build_official_gnndelete_support, build_message_passing_domains, build_gate_buckets
from contracts.sampling import SamplingIdentity, SharedSampler
from contracts.gat_support import build_support_gates
from contracts import rgcn
from lrr.models.gat import GATBackbone


def procedural_schedule(data,full,mask,identity,updates):
    official=build_official_gnndelete_support(full,mask,data.num_nodes)
    domains=build_message_passing_domains(full,mask,official.official_sdf_mask)
    return dict(format="role_seeded_retain_v1",shape=[updates,2,min(domains.dr_mp_edge_index.size(1),50000)],
                identity=dict(dataset=identity.dataset,backbone="gcn",deletion_mode=identity.deletion_mode,
                              deletion_ratio=identity.deletion_ratio,run_seed=identity.run_seed),
                full=domains.full_train_mp_edge_index,dr=domains.dr_mp_edge_index,
                local_mask=domains.local_loss_edge_mask,df=full[:,mask],num_nodes=data.num_nodes)


def homogeneous(config,data,state):
    family=config["backbone"].lower();identity=SamplingIdentity(**config["sampling_identity"])
    full=data.train_pos_edge_index.cpu().long()
    _,_,_,inside=k_hop_subgraph(data.test_pos_edge_index.flatten().unique(),2,full,num_nodes=data.num_nodes)
    seed_everything(config["seed"])
    frozen=ROOT/'data/deletion_requests'/f"{config.get('source_cell_id','')}.npz"
    if frozen.is_file():
        metadata=next(r for r in json.loads((ROOT/'manifests/deletion_requests.json').read_text()) if r['cell_id']==config['source_cell_id'])
        if hashlib.sha256(frozen.read_bytes()).hexdigest()!=metadata['sha256']:raise ValueError('frozen request checksum mismatch')
        with np.load(frozen,allow_pickle=False) as arrays:expected=torch.from_numpy(arrays['df'].copy()).long()
        ids=full[0]*data.num_nodes+full[1];requested_ids=expected[0]*data.num_nodes+expected[1]
        order=torch.argsort(ids);positions=torch.searchsorted(ids[order],requested_ids)
        if bool((positions>=ids.numel()).any()):raise ValueError('request edge absent from the prepared graph')
        indices=order[positions]
        if not torch.equal(full[:,indices],expected):raise ValueError('frozen request does not match the prepared training graph')
    else:
        indices=reproduce_delete_gnn_df_indices(inside if identity.deletion_mode=="in" else ~inside,
                                               full.size(1),float(config["deletion_ratio"]))
    mask=torch.zeros(full.size(1),dtype=torch.bool);mask[indices]=True
    df=expected if frozen.is_file() else full[:,mask]
    if family=="gcn":
        official=build_official_gnndelete_support(full,mask,data.num_nodes)
        domains=build_message_passing_domains(full,mask,official.official_sdf_mask)
        gates=build_gate_buckets(official.endpoint_mask,official.support1,official.support2,dtype=data.x.dtype)
        support1,support2=official.support1,official.support2
        full_mp,dr,local_mask=domains.full_train_mp_edge_index,domains.dr_mp_edge_index,domains.local_loss_edge_mask
    else:
        gates=build_support_gates(num_nodes=data.num_nodes,full_d_edge_index=full,df_edge_index=df)
        support1,support2=gates.support1,gates.support2
        full_mp,dr=full,full[:,~mask]
        local_mask=support2[full[0]] & support2[full[1]] & ~mask
    sampler=SharedSampler(identity,full_train_mp_edge_index=full_mp,dr_mp_edge_index=dr,
                           local_mp_mask=local_mask,directed_df_edge_index=df,num_nodes=data.num_nodes)
    static=sampler.build_static_bundle()
    evaluation=sampler.build_evaluation_bundle(validation_pos=data.val_pos_edge_index,validation_neg=data.val_neg_edge_index,
                   test_pos=data.test_pos_edge_index,test_neg=data.test_neg_edge_index)
    controls=dr[:,evaluation.df_retained_group_positions].permute(1,0,2).contiguous()
    train=dict(x=data.x,full_train_mp_edge_index=full_mp,dr_mp_edge_index=dr,df=df,
               support1=support1,support2=support2,gate1=gates.gate1,gate2=gates.gate2,
               local=static.local_static,retain_positive=static.retain_positive_static,anchor=static.anchor_static,
               retain_negative_schedule=procedural_schedule(data,full,mask,identity,int(config["updates"])))
    if family=="gat":
        teacher=GATBackbone(data.x.size(1));teacher.load_state_dict(state,strict=True);teacher.eval()
        with torch.no_grad():
            z=teacher(data.x,dr)
            train["teacher_local_probability"]=teacher.decode(z,static.local_static).sigmoid()
            train["teacher_anchor_probability"]=teacher.decode(z,static.anchor_static).sigmoid()
    def val(split):return dict(positive=getattr(evaluation,split+"_pos"),negative=getattr(evaluation,split+"_neg"),requested=df,retained_control_groups=controls)
    return dict(model=dict(input_dim=data.x.size(1),input_mode="feature_matrix"),training=train,
                validation=val("validation"),test=val("test"),backbone_state_dict=state,sampling_identity=config["sampling_identity"])


def relational(config,data,state,checkpoint):
    rgcn.SEED=int(config["seed"]);rgcn.DELETION_TYPE=config["deletion_mode"];rgcn.DELETION_RATIO=float(config["deletion_ratio"])
    rgcn.IDENTITY=SamplingIdentity(**config["sampling_identity"])
    rgcn.ORIGINAL_CONFIG=ROOT/f"configs/pretraining/rgcn_wn18rr_seed{config['seed']}.json"
    rgcn.ORIGINAL_CHECKPOINT=checkpoint
    a=rgcn.build_assets(data,11,data.train_pos_edge_index)
    request=rgcn.build_request(data,11,data.train_pos_edge_index)
    graph=rgcn.build_data(data.train_pos_edge_index,data.train_edge_type,request["selected_train_positions"])
    train={k:a[k] for k in ["df","df_type","local","local_type","retain_positive","retain_type","anchor","anchor_type","support1","support2","gate1","gate2"]}
    edges=a["dr"][:,a["df_retained_group_positions"]].permute(1,0,2).contiguous()
    relations=a["dr_type"][a["df_retained_group_positions"]]
    def bank(split):
        positive=data.valid_pos_edge_index if split=="validation" else data.test_pos_edge_index
        typ=data.valid_edge_type if split=="validation" else data.test_edge_type
        rgcn.set_seed(config["seed"]);negative=rgcn.negative_sampling_kg(positive,typ)
        return dict(x=data.x,edge_index=a["dr_mp"],edge_type=a["dr_mp_type"],gate1=a["gate1"],gate2=a["gate2"],
                    val_positive=positive,val_negative=negative,val_relation=typ,requested=a["df"],requested_relation=a["df_type"],
                    retained_control_groups=edges,retained_control_relations=relations)
    return dict(model=dict(num_entities=data.num_nodes,base_relations=11),graphsaint_data=graph,training=train,
                validation=bank("validation"),test=bank("test"),backbone_state_dict=state)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path);p.add_argument("--output",type=Path)
    a=p.parse_args();config=json.loads(a.config.read_text());family=config["backbone"].lower()
    checkpoint=a.checkpoint or ROOT/f"outputs/pretrain/{family}_{config['dataset'].lower()}_seed{config['seed']}.pt"
    output=a.output or (a.config.resolve().parent/config["bundle_path"]).resolve()
    if output.exists():raise FileExistsError(output)
    data=torch.load(ROOT/f"data/processed/{config['dataset']}/seed{config['seed']}.pt",map_location="cpu",weights_only=False)
    payload=torch.load(checkpoint,map_location="cpu",weights_only=False);state=payload.get("model_state",payload.get("state_dict",payload))
    bundle=relational(config,data,state,checkpoint) if family=="rgcn" else homogeneous(config,data,state)
    bundle["source_cell_id"]=config["source_cell_id"]
    digest=hashlib.sha256()
    with checkpoint.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    bundle['checkpoint_sha256']=digest.hexdigest()
    output.parent.mkdir(parents=True,exist_ok=True);torch.save(bundle,output);print(f"bundle={output}")


if __name__=="__main__":main()
