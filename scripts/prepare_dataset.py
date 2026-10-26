"""Prepare the paper's CitationFull or WN18RR inputs from upstream raw data."""
from _common import ROOT
import argparse
import hashlib,json
import numpy as np
from pathlib import Path
import torch
from torch_geometric.datasets import CitationFull
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.seed import seed_everything
from contracts.split import split_edges_no_neg_adj_mask
from contracts import rgcn

def verify_raw(dataset,folder):
    manifest=json.loads((ROOT/'manifests/raw_datasets.json').read_text())
    for row in manifest:
        if row['dataset']!=dataset:continue
        path=folder/row['filename'];digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=row['sha256']:raise ValueError(f'raw dataset checksum mismatch: {path}')

def prepare(dataset, seed, raw_root):
    seed_everything(seed)
    if dataset in {"Cora", "DBLP"}:
        raw = CitationFull(str(raw_root / dataset), dataset, transform=NormalizeFeatures())
        verify_raw(dataset,Path(raw.raw_dir))
        data = raw[0]
        data.train_mask = data.val_mask = data.test_mask = None
        pinned=ROOT/f"data/splits/{dataset.lower()}_seed_{seed}.npz"
        if pinned.is_file():
            metadata=next(r for r in json.loads((ROOT/'manifests/data_splits.json').read_text()) if r['dataset']==dataset and r['seed']==seed)
            if hashlib.sha256(pinned.read_bytes()).hexdigest()!=metadata['sha256']:raise ValueError('frozen split checksum mismatch')
            if hashlib.sha256(data.x.contiguous().numpy().tobytes()).hexdigest()!=metadata['feature_sha256']:raise ValueError('downloaded dataset features differ from the final-paper input')
            with np.load(pinned,allow_pickle=False) as arrays:
                for key in arrays.files:setattr(data,key,torch.from_numpy(arrays[key].copy()))
            data.edge_index=None
            return data
        raise ValueError('No published frozen split for this dataset/seed')
    rgcn.DATA_ROOT = raw_root / "WordNet18RR"
    graph=rgcn.load_graph()[0]
    verify_raw('WN18RR',rgcn.DATA_ROOT/'raw')
    return graph


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",choices=["Cora","DBLP","WN18RR"],required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--raw-root",type=Path,default=ROOT/"data/raw")
    p.add_argument("--output",type=Path)
    a=p.parse_args();output=a.output or ROOT/f"data/processed/{a.dataset}/seed{a.seed}.pt"
    if output.exists(): raise FileExistsError(output)
    data=prepare(a.dataset,a.seed,a.raw_root)
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(data,output)
    print(f"prepared={output} nodes={data.num_nodes}")


if __name__=="__main__":main()
