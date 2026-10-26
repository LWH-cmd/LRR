from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch_geometric.data import Data

from lrr.evaluation import RGCNValidationBundle, evaluate_rgcn
from lrr.gates import validate_gates
from lrr.models.rgcn import FullRankRGCN, RGCNBackbone
from lrr.objectives.p2 import P2_MARGIN, P2_WEIGHTS, p2_objective
from lrr.sampling.graphsaint import WindowsSafeGraphSAINTRandomWalkSampler
from lrr.sampling.relational import (RelationalSampler, RelationalSamplingIdentity,
                                     relation_aware_negative)
from lrr.selectors import RGCNSelector


RGCN_BUDGET = 1600
RGCN_VALIDATION_UPDATES = (
    0, 8, 16, 24, 32, 40, 48, 56, 64, 128, 192, 256, 320, 384, 448,
    512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216,
    1280, 1344, 1408, 1472, 1536, 1600,
)


@dataclass
class PreparedRGCN:
    config: dict[str, Any]
    bundle: dict[str, Any]
    device: torch.device
    model: FullRankRGCN
    optimizer: torch.optim.Optimizer
    graphsaint_data: Data
    loader: WindowsSafeGraphSAINTRandomWalkSampler
    assets: dict[str, torch.Tensor]
    domain_keys: dict[str, torch.Tensor]
    validation: RGCNValidationBundle
    sampler: RelationalSampler
    selector: RGCNSelector
    base_relations: int
    num_nodes: int


def _require(mapping: dict[str, Any], keys: tuple[str, ...], label: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise KeyError(f"{label} is missing fields: {', '.join(missing)}")


def _tensors_to(mapping: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    result = {}
    for key, value in mapping.items():
        if not torch.is_tensor(value):
            raise TypeError(f"bundle field {key!r} must be a tensor")
        result[key] = value.to(device)
    return result


def _domain_keys(edges: torch.Tensor, relations: torch.Tensor, base_relations: int,
                 num_nodes: int) -> torch.Tensor:
    keys = ((edges[0].long() * base_relations + relations.long()) * num_nodes + edges[1].long())
    return keys.sort().values


def prepare_rgcn(config: dict[str, Any], bundle: dict[str, Any],
                 device: torch.device | str = "cpu") -> PreparedRGCN:
    """Load frozen assets and construct GraphSAINT/validation state before timing."""
    device = torch.device(device)
    model_cfg = bundle["model"]
    train_cpu = bundle["training"]
    val_cpu = bundle["validation"]
    data_cpu = bundle["graphsaint_data"]
    base_relations, num_nodes = int(model_cfg["base_relations"]), int(model_cfg["num_entities"])
    asset_names = ("df", "df_type", "local", "local_type", "retain_positive", "retain_type",
                   "anchor", "anchor_type", "support1", "support2", "gate1", "gate2")
    val_names = ("x", "edge_index", "edge_type", "gate1", "gate2", "val_positive",
                 "val_negative", "val_relation", "requested", "requested_relation",
                 "retained_control_groups", "retained_control_relations")
    _require(train_cpu, asset_names, "training")
    _require(val_cpu, val_names, "validation")
    if not isinstance(data_cpu, Data):
        raise TypeError("graphsaint_data must be a torch_geometric.data.Data instance")
    if data_cpu.edge_index.size(1) != data_cpu.edge_type.numel():
        raise ValueError("GraphSAINT edge_index and edge_type length mismatch")
    if data_cpu.edge_type.numel() != data_cpu.df_mask.numel() or data_cpu.df_mask.numel() != data_cpu.dr_mask.numel():
        raise ValueError("GraphSAINT edge masks must align with edge columns")

    backbone = RGCNBackbone(num_nodes, base_relations,
                            embedding_dim=int(model_cfg.get("embedding_dim", 128)),
                            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
                            output_dim=int(model_cfg.get("output_dim", 64)))
    state = bundle.get("backbone_state_dict")
    if not isinstance(state, dict):
        raise KeyError("bundle must contain backbone_state_dict")
    backbone.load_state_dict(state, strict=True)
    model = FullRankRGCN(backbone).to(device)
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    actual = [(name, int(p.numel())) for name, p in trainable]
    if actual != [("repair1.W", 16_384), ("repair2.W", 4_096)]:
        raise RuntimeError(f"unexpected FullRank trainable set: {actual}")
    if any(torch.count_nonzero(p).item() for _, p in trainable):
        raise RuntimeError("FullRank repairs must start from zero")
    optimizer = torch.optim.Adam([p for _, p in trainable], lr=float(config.get("lr", 1e-3)),
                                 weight_decay=float(config.get("weight_decay", 0.0)))

    assets = _tensors_to(train_cpu, device)
    validate_gates(assets["df"].flatten(), assets["support1"], assets["support2"],
                   assets["gate1"], assets["gate2"])
    domain_keys = {
        name: _domain_keys(assets[edge_key], assets[type_key], base_relations, num_nodes)
        for name, edge_key, type_key in (("local", "local", "local_type"),
                                        ("retain", "retain_positive", "retain_type"),
                                        ("anchor", "anchor", "anchor_type"))
    }
    validation = RGCNValidationBundle(**_tensors_to(val_cpu, device))
    if validation.retained_control_groups.size(0) != 500:
        raise ValueError("formal RGCN bundle must contain 500 retained-control groups")

    graphsaint_data = data_cpu.cpu()
    loader = WindowsSafeGraphSAINTRandomWalkSampler(
        graphsaint_data,
        batch_size=int(config.get("start_nodes", 1024)),
        walk_length=int(config.get("walk_length", 2)),
        num_steps=int(config.get("batches_per_round", 32)),
        num_workers=0,
    )
    identity_cfg = config.get("sampling_identity")
    if not isinstance(identity_cfg, dict):
        raise KeyError("sampling_identity must contain the exact formal sampler identity")
    _require(identity_cfg, ("dataset", "backbone", "deletion_mode", "deletion_ratio", "run_seed"),
             "sampling_identity")
    if int(identity_cfg["run_seed"]) != int(config["seed"]):
        raise ValueError("sampling_identity.run_seed must match config.seed")
    identity = RelationalSamplingIdentity(
        dataset=str(identity_cfg["dataset"]), backbone=str(identity_cfg["backbone"]),
        deletion_mode=str(identity_cfg["deletion_mode"]),
        deletion_ratio=str(identity_cfg["deletion_ratio"]), run_seed=int(identity_cfg["run_seed"]),
    )
    sampler = RelationalSampler(identity)
    selector = RGCNSelector(epsilon=float(config.get("selector_epsilon", 0.010)))
    return PreparedRGCN(config, bundle, device, model, optimizer, graphsaint_data, loader,
                        assets, domain_keys, validation, sampler, selector,
                        base_relations, num_nodes)


def _actual_domain(prepared: PreparedRGCN, batch: Data, domain: str,
                   edge_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    mask = batch.dr_mask if edge_mask is None else edge_mask
    mask = mask & (batch.edge_type < prepared.base_relations)
    edges, types = batch.edge_index[:, mask], batch.edge_type[mask]
    if domain == "df":
        return edges, types
    global_edges = batch.x[edges]
    keys = ((global_edges[0].long() * prepared.base_relations + types.long()) *
            prepared.num_nodes + global_edges[1].long())
    candidates = prepared.domain_keys[domain]
    positions = torch.searchsorted(candidates, keys)
    keep = positions < candidates.numel()
    safe = positions.clamp_max(max(candidates.numel() - 1, 0))
    keep &= candidates[safe].eq(keys)
    return edges[:, keep], types[keep]


def _map_global_edges(edges: torch.Tensor, global_to_local: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    local = global_to_local[edges.to(global_to_local.device)]
    return local, (local >= 0).all(dim=0)


def _step(prepared: PreparedRGCN, batch: Data, update: int, *,
          capture: bool = False) -> dict[str, Any]:
    model, device = prepared.model, prepared.device
    raw_df, raw_df_type = _actual_domain(prepared, batch, "df", edge_mask=batch.df_mask)
    if raw_df.size(1) == 0:
        return {"stepped": False}

    model.train()
    optimizer = prepared.optimizer
    optimizer.zero_grad(set_to_none=True)
    edge = batch.edge_index[:, batch.dr_mask]
    edge_type = batch.edge_type[batch.dr_mask]
    nodes = batch.x.to(device)
    gate1 = prepared.assets["gate1"][nodes]
    gate2 = prepared.assets["gate2"][nodes]
    student_z, teacher_z = model.shared_first_layer(nodes, edge, edge_type, gate1, gate2)
    global_to_local = torch.full((prepared.num_nodes,), -1, dtype=torch.long, device=device)
    global_to_local[nodes] = torch.arange(nodes.numel(), device=device)

    raw_df_global = nodes[raw_df].detach().cpu()
    forget_global = relation_aware_negative(
        raw_df_global, raw_df_type.detach().cpu(),
        prepared.sampler.role_seed("forget_negative", update),
    )
    forget_local, forget_valid = _map_global_edges(forget_global, global_to_local)
    if not bool(forget_valid.all()):
        raise RuntimeError("formal Forget negatives escaped current GraphSAINT batch")

    local, local_type = _actual_domain(prepared, batch, "local")
    retain_pos, retain_type = _actual_domain(prepared, batch, "retain")
    anchor, anchor_type = _actual_domain(prepared, batch, "anchor")
    if local.size(1) == 0 or retain_pos.size(1) == 0 or anchor.size(1) == 0:
        raise RuntimeError(f"GraphSAINT batch lacks a P2 domain at update {update}")
    retain_global = nodes[retain_pos].detach().cpu()
    retain_negative_global = relation_aware_negative(
        retain_global, retain_type.detach().cpu(),
        prepared.sampler.role_seed("retain_negative", update),
    )
    retain_neg, retain_valid = _map_global_edges(retain_negative_global, global_to_local)
    if not bool(retain_valid.all()):
        raise RuntimeError("formal Retain negatives escaped current GraphSAINT batch")

    df_type = raw_df_type.to(device)
    local_type, retain_type, anchor_type = (local_type.to(device), retain_type.to(device), anchor_type.to(device))
    with torch.no_grad():
        teacher_forget = model.decode(teacher_z, forget_local, df_type).detach()
        teacher_local = torch.sigmoid(model.decode(teacher_z, local, local_type))
        teacher_anchor = torch.sigmoid(model.decode(teacher_z, anchor, anchor_type))

    rp = model.decode(student_z, retain_pos, retain_type)
    rn = model.decode(student_z, retain_neg, retain_type)
    loss, parts = p2_objective(
        forget_positive_logits=model.decode(student_z, raw_df, df_type),
        forget_reference_logits=teacher_forget,
        local_logits=model.decode(student_z, local, local_type),
        teacher_local_probability=teacher_local,
        retain_positive_logits=rp, retain_negative_logits=rn,
        anchor_logits=model.decode(student_z, anchor, anchor_type),
        teacher_anchor_probability=teacher_anchor,
        weights=P2_WEIGHTS, margin=P2_MARGIN,
    )
    loss.backward()
    values = {name: value.detach() for name, value in parts.items()} if capture else None
    optimizer.step()
    return {"stepped": True, "losses": values}


@torch.no_grad()
def _validate(prepared: PreparedRGCN) -> dict[str, float]:
    bundle = prepared.validation
    z = prepared.model(bundle.x, bundle.edge_index, bundle.edge_type,
                       bundle.gate1, bundle.gate2)
    return evaluate_rgcn(prepared.model, z, bundle)


def train_prepared_rgcn(prepared: PreparedRGCN) -> dict[str, Any]:
    rounds = int(prepared.config.get("rounds", 50))
    steps_per_round = int(prepared.config.get("batches_per_round", 32))
    expected_updates = int(prepared.config.get("updates", RGCN_BUDGET))
    if rounds * steps_per_round != expected_updates:
        raise ValueError("rounds * batches_per_round must equal updates")
    if expected_updates != RGCN_BUDGET:
        schedule = tuple(int(u) for u in prepared.config.get("validation_updates", ()))
    else:
        schedule = RGCN_VALIDATION_UPDATES
    if not schedule or schedule[0] != 0 or schedule[-1] != expected_updates:
        raise ValueError("RGCN validation schedule must include update 0 and final update")
    torch.manual_seed(int(prepared.config["seed"]))
    prepared.model.eval()
    prepared.selector.consider(0, _validate(prepared), prepared.model)
    update = 0
    optimizer_steps = 0
    skipped_batches = 0
    for _round in range(rounds):
        for cpu_batch in prepared.loader:
            batch = cpu_batch.to(prepared.device)
            step_result = _step(prepared, batch, update)
            if step_result["stepped"]:
                optimizer_steps += 1
            else:
                skipped_batches += 1
            update += 1
            if update in schedule:
                prepared.model.eval()
                prepared.selector.consider(update, _validate(prepared), prepared.model)
    if update != expected_updates or optimizer_steps != expected_updates or skipped_batches:
        raise RuntimeError(f"RGCN budget mismatch: updates={update}, optimizer={optimizer_steps}, skipped={skipped_batches}")
    if tuple(int(row["update"]) for row in prepared.selector.trajectory) != schedule:
        raise RuntimeError("RGCN selector trajectory does not match configured schedule")
    prepared.selector.restore(prepared.model)
    return {
        "selected_update": prepared.selector.best_update,
        "selected_metrics": prepared.selector.best_metrics,
        "selected_repair_state": prepared.selector.best_state,
        "validation_trajectory": prepared.selector.trajectory,
        "validation_updates": list(schedule),
        "updates": update,
        "optimizer_steps": optimizer_steps,
        "skipped_batches": skipped_batches,
        "forget_reference": "Original_teacher_negative_score / FROZEN_TEACHER",
        "trainable_parameters": sum(p.numel() for p in prepared.model.parameters() if p.requires_grad),
        "optimizer": {"name": "Adam", "lr": float(prepared.optimizer.defaults["lr"]),
                      "weight_decay": float(prepared.optimizer.defaults["weight_decay"])},
    }


def train_rgcn(config: dict[str, Any], bundle: dict[str, Any], device: torch.device | str = "cpu") -> dict[str, Any]:
    return train_prepared_rgcn(prepare_rgcn(config, bundle, device))


@torch.no_grad()
def evaluate_rgcn_test(prepared: PreparedRGCN, test_bundle: dict[str, Any]) -> dict[str, float]:
    values = _tensors_to(test_bundle, prepared.device)
    bundle = RGCNValidationBundle(**values)
    z = prepared.model(bundle.x, bundle.edge_index, bundle.edge_type,
                       bundle.gate1, bundle.gate2)
    return evaluate_rgcn(prepared.model, z, bundle)
