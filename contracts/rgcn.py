from __future__ import annotations
import hashlib, json, random
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import WordNet18RR
from torch_geometric.seed import seed_everything
from torch_geometric.utils import k_hop_subgraph
from contracts.lrr_runtime import reproduce_delete_gnn_df_indices
from contracts.sampling import SamplingIdentity, anchor_child_seed, canonical_sampling_key, deterministic_index_selection, seed_from_canonical_key
from contracts.rgcn_support import SupportGateBundle, build_support_gates
from lrr.models.rgcn import RGCNBackbone
SEED=42
BASE_RELATIONS=11
NUM_NODES=40943
DELETION_TYPE='Ed_IN'
DELETION_RATIO=2.5
DATA_ROOT=Path('data/raw/WordNet18RR')
ORIGINAL_CONFIG=Path('configs/pretraining/rgcn_wn18rr_seed42.json')
ORIGINAL_CHECKPOINT=Path('outputs/pretrain/rgcn_wn18rr_seed42.pt')
IDENTITY=SamplingIdentity('WordNet18RR','rgcn','in','2.5',42)
def build_original_model(num_entities,base_relations,config,device):
    return RGCNBackbone(num_entities,base_relations).to(device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    seed_everything(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def canonical_key(identity: SamplingIdentity, role: str, group: int) -> str:
    # This is the accepted relational evaluation construction, with only the
    # dataset identity parameterized from WordNet18 to WordNet18RR.
    return json.dumps({
        "evaluation_candidate_version": "WORDNET18-DF-EVAL-v1",
        "dataset": identity.dataset,
        "backbone": identity.backbone,
        "deletion_mode": identity.deletion_mode,
        "deletion_ratio_canonical": identity.deletion_ratio,
        "run_seed": identity.run_seed,
        "candidate_role": role,
        "group_index": group,
    }, separators=(",", ":"))

def key_seed(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") % (2**63 - 1)

def evaluation_positions(dr_count: int, count: int, identity: SamplingIdentity) -> torch.Tensor:
    rows = []
    for group in range(500):
        seed = key_seed(canonical_key(identity, "df_retained_positive_group", group))
        generator = torch.Generator(device="cpu").manual_seed(seed)
        rows.append(torch.randperm(dr_count, generator=generator)[:count])
    return torch.stack(rows)

def load_graph() -> tuple[Data, int, dict[str, int], torch.Tensor]:
    raw = WordNet18RR(str(DATA_ROOT))[0]
    base_relations = int(raw.edge_type.unique(sorted=True).numel())
    if base_relations != BASE_RELATIONS:
        raise RuntimeError(f"unexpected WN18RR base relation count: {base_relations}")
    if not torch.equal(raw.edge_type.unique(sorted=True).cpu(), torch.arange(base_relations)):
        raise RuntimeError("WN18RR relation IDs are not contiguous")
    train = raw.edge_index[:, raw.train_mask].contiguous().long()
    train_type = raw.edge_type[raw.train_mask].contiguous().long()
    valid = raw.edge_index[:, raw.val_mask].contiguous().long()
    valid_type = raw.edge_type[raw.val_mask].contiguous().long()
    test = raw.edge_index[:, raw.test_mask].contiguous().long()
    test_type = raw.edge_type[raw.test_mask].contiguous().long()
    graph = Data(
        x=torch.arange(raw.num_nodes, dtype=torch.long),
        edge_index=torch.cat((train, train.flip(0)), dim=1),
        edge_type=torch.cat((train_type, train_type + base_relations)),
        num_nodes=int(raw.num_nodes),
        train_pos_edge_index=train,
        train_edge_type=train_type,
        valid_pos_edge_index=valid,
        valid_edge_type=valid_type,
        test_pos_edge_index=test,
        test_edge_type=test_type,
    )
    counts = {
        "num_entities": int(raw.num_nodes),
        "base_num_relations": base_relations,
        "train": int(train.size(1)),
        "valid": int(valid.size(1)),
        "test": int(test.size(1)),
        "reverse_relation_ids": f"{base_relations}..{2 * base_relations - 1}",
    }
    return graph, base_relations, counts, raw.edge_index[:, raw.train_mask].contiguous().long()

def negative_sampling_kg(edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
    result = edge_index.clone()
    for relation in edge_type.unique(sorted=True):
        mask = edge_type == relation
        heads = result[0, mask]
        result[0, mask] = heads[torch.randperm(heads.numel())]
    return result

def build_request(
    graph: Data,
    base_relations: int,
    train: torch.Tensor,
    *,
    deletion_type: str | None = None,
    deletion_ratio: float | None = None,
) -> dict[str, torch.Tensor | int | str]:
    deletion_type = DELETION_TYPE if deletion_type is None else deletion_type
    deletion_ratio = DELETION_RATIO if deletion_ratio is None else float(deletion_ratio)
    if deletion_type not in {"Ed_IN", "Ed_OUT"}:
        raise RuntimeError(f"BLOCKED: no accepted WN18RR construction for {deletion_type}")
    test = graph.test_pos_edge_index
    _, _, _, in_mask = k_hop_subgraph(test.flatten().unique(), 2, train, num_nodes=graph.num_nodes)
    candidate_mask = in_mask if deletion_type == "Ed_IN" else ~in_mask
    candidate_count = int(candidate_mask.sum())
    requested = int((deletion_ratio / 100.0) * train.size(1))
    set_seed(SEED)
    selected = reproduce_delete_gnn_df_indices(candidate_mask, train.size(1), deletion_ratio).long().contiguous()
    if selected.unique().numel() != requested:
        raise RuntimeError(f"{deletion_type} selection count mismatch: {selected.unique().numel()} != {requested}")
    mask = torch.zeros(train.size(1), dtype=torch.bool)
    mask[selected] = True
    train_type = graph.train_edge_type
    df, df_type = train[:, mask].contiguous(), train_type[mask].contiguous()
    dr, dr_type = train[:, ~mask].contiguous(), train_type[~mask].contiguous()
    dr_mp = torch.cat((dr, dr.flip(0)), dim=1).contiguous()
    dr_mp_type = torch.cat((dr_type, dr_type + base_relations)).contiguous()
    return {
        "candidate_count": candidate_count,
        "requested_count": requested,
        "candidate_mask": candidate_mask,
        "df_mask": mask,
        "dr_mask": ~mask,
        "df": df,
        "df_type": df_type,
        "dr": dr,
        "dr_type": dr_type,
        "dr_mp": dr_mp,
        "dr_mp_type": dr_mp_type,
        "selected_train_positions": selected,
    }


def unique_triple_positions(edges: torch.Tensor, relations: torch.Tensor, positions: torch.Tensor,
                            excluded: set[tuple[int, int, int]] | None = None):
    seen = set() if excluded is None else set(excluded)
    kept: list[int] = []
    identities: list[tuple[int, int, int]] = []
    for position in positions.tolist():
        identity = (int(edges[0, position]), int(relations[position]), int(edges[1, position]))
        if identity in seen:
            continue
        seen.add(identity)
        kept.append(position)
        identities.append(identity)
    return torch.tensor(kept, dtype=torch.long), identities


def select_positions(pool: torch.Tensor, count: int, role: str) -> torch.Tensor:
    key = canonical_sampling_key(IDENTITY, sample_role=role)
    seed, _ = seed_from_canonical_key(key)
    return pool[deterministic_index_selection(pool.numel(), min(count, pool.numel()), seed)]


def build_static(dr: torch.Tensor, dr_type: torch.Tensor, support2: torch.Tensor):
    all_positions = torch.arange(dr.size(1), dtype=torch.long)
    local_pool = all_positions[support2[dr[0]] & support2[dr[1]]]
    local = select_positions(local_pool, 50_000, "local_static")
    retain = select_positions(all_positions, 50_000, "retain_positive_static")
    parent = canonical_sampling_key(IDENTITY, sample_role="anchor_static")
    local_seed, _ = anchor_child_seed(parent, "local_select")
    global_seed, _ = anchor_child_seed(parent, "global_fill")
    local_unique, local_ids = unique_triple_positions(dr, dr_type, local_pool)
    local_pick = deterministic_index_selection(local_unique.numel(), min(50_000, local_unique.numel()), local_seed)
    anchor_local = local_unique[local_pick]
    selected_ids = {local_ids[int(i)] for i in local_pick.tolist()}
    global_unique, _ = unique_triple_positions(dr, dr_type, all_positions, selected_ids)
    remaining = 50_000 - anchor_local.numel()
    global_pick = deterministic_index_selection(global_unique.numel(), min(remaining, global_unique.numel()), global_seed)
    anchor = torch.cat([anchor_local, global_unique[global_pick]])
    return local, retain, anchor


def build_data(train: torch.Tensor, train_type: torch.Tensor, df_positions: torch.Tensor) -> Data:
    df_base = torch.zeros(train.size(1), dtype=torch.bool)
    df_base[df_positions] = True
    edge_index = torch.cat([train, train.flip(0)], dim=1)
    edge_type = torch.cat([train_type, train_type + BASE_RELATIONS])
    df_mask = torch.cat([df_base, df_base])
    return Data(
        x=torch.arange(NUM_NODES, dtype=torch.long), num_nodes=NUM_NODES,
        edge_index=edge_index, edge_type=edge_type, df_mask=df_mask, dr_mask=~df_mask,
    )


def build_assets(graph: Data, base_relations: int, train: torch.Tensor) -> dict[str, Any]:
    request = build_request(graph, base_relations, train)
    df, df_type, dr, dr_type = (request[name] for name in ("df", "df_type", "dr", "dr_type"))
    gates = build_support_gates(full_d=train, df=df, num_nodes=NUM_NODES)
    local_pos, retain_pos, anchor_pos = build_static(dr, dr_type, gates.support2)
    local, local_type = dr[:, local_pos], dr_type[local_pos]
    retain, retain_type = dr[:, retain_pos], dr_type[retain_pos]
    anchor, anchor_type = dr[:, anchor_pos], dr_type[anchor_pos]
    identity = IDENTITY
    eval_pos = evaluation_positions(dr.size(1), int(request["requested_count"]), identity)

    config = json.loads(ORIGINAL_CONFIG.read_text(encoding="utf-8"))
    # See the native torch_sparse/GraphSAINT Windows isolation in the DECNI
    # asset builder; only precomputed teacher probabilities are CPU-side.
    device = torch.device("cpu")
    teacher = build_original_model(NUM_NODES, base_relations, config, device)
    payload = torch.load(ORIGINAL_CHECKPOINT, map_location="cpu", weights_only=False)
    teacher.load_state_dict(payload["model_state"], strict=True)
    teacher.eval()
    x = graph.x.to(device)
    dr_mp, dr_mp_type = request["dr_mp"].to(device), request["dr_mp_type"].to(device)
    with torch.no_grad():
        h2 = teacher(x, dr_mp, dr_mp_type)
    teacher_h2 = h2.cpu()
    with torch.no_grad():
        teacher_local = torch.sigmoid(teacher.decode(h2, local.to(device), local_type.to(device))).cpu()
        teacher_retain = torch.sigmoid(teacher.decode(h2, retain.to(device), retain_type.to(device))).cpu()
        teacher_anchor = torch.sigmoid(teacher.decode(h2, anchor.to(device), anchor_type.to(device))).cpu()
    del teacher, h2
    if device.type == "cuda":
        torch.cuda.empty_cache()

    set_seed(SEED)
    val_pos, val_type = graph.valid_pos_edge_index, graph.valid_edge_type
    val_neg = val_pos.clone()
    for relation in val_type.unique(sorted=True):
        mask = val_type == relation
        heads = val_neg[0, mask]
        val_neg[0, mask] = heads[torch.randperm(heads.numel())]
    return {
        "df": df, "df_type": df_type, "dr": dr, "dr_type": dr_type,
        "dr_mp": request["dr_mp"], "dr_mp_type": request["dr_mp_type"],
        "support1": gates.support1, "support2": gates.support2, "gate1": gates.gate1, "gate2": gates.gate2,
        "distance": gates.distance, "teacher_h2": teacher_h2,
        "teacher_local_probability": teacher_local, "teacher_retain_probability": teacher_retain,
        "teacher_anchor_probability": teacher_anchor,
        "local": local, "local_type": local_type, "retain_positive": retain, "retain_type": retain_type,
        "anchor": anchor, "anchor_type": anchor_type, "df_retained_group_positions": eval_pos,
        "val_pos": val_pos, "val_neg": val_neg, "val_type": val_type,
        "df_position_lookup": {
            (int(df[0, i]), int(df_type[i]), int(df[1, i])): i for i in range(df.size(1))
        },
        "metadata": {"deleted": int(df.size(1)), "retained": int(dr.size(1)), "base_relations": base_relations},
    }
