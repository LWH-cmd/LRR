from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from lrr.evaluation import GCNValidationBundle, evaluate_gcn
from lrr.gates import validate_gates
from lrr.models.gcn import FullRankGCN, GCNBackbone
from lrr.objectives.p2 import P2_MARGIN, P2_WEIGHTS, p2_objective
from lrr.sampling.homogeneous import ForgetNegativeSampler, SamplingIdentity, RetainNegativeSchedule
from lrr.selectors import GCNSelector


@dataclass
class PreparedGCN:
    config: dict[str, Any]
    bundle: dict[str, Any]
    device: torch.device
    model: FullRankGCN
    optimizer: torch.optim.Optimizer
    base_h1: torch.Tensor
    training: dict[str, torch.Tensor]
    validation: GCNValidationBundle
    sampler: ForgetNegativeSampler
    schedule: RetainNegativeSchedule
    edge_buffer: torch.Tensor
    slices: dict[str, slice]
    counts: dict[str, int]
    selector: GCNSelector


def _to_device(mapping: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    result = {}
    for key, value in mapping.items():
        if not torch.is_tensor(value):
            raise TypeError(f"bundle field {key!r} must be a tensor")
        result[key] = value.to(device)
    return result


def _required(mapping: dict[str, Any], keys: tuple[str, ...], label: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise KeyError(f"{label} is missing fields: {', '.join(missing)}")


def prepare_gcn(config: dict[str, Any], bundle: dict[str, Any],
                device: torch.device | str = "cpu") -> PreparedGCN:
    """Load model/static tensors and prepare caches outside a runtime timer."""
    device = torch.device(device)
    if "sampling_identity" in bundle and bundle["sampling_identity"] != config["sampling_identity"]:
        raise ValueError("bundle and config sampling identities differ")
    model_cfg = bundle["model"]
    training_cpu = bundle["training"]
    validation_cpu = bundle["validation"]
    train_keys = ("x", "full_train_mp_edge_index", "dr_mp_edge_index", "support1", "gate1",
                  "support2", "gate2", "df", "local", "retain_positive", "anchor",
                  "retain_negative_schedule")
    val_keys = ("positive", "negative", "requested", "retained_control_groups")
    _required(training_cpu, train_keys, "training")
    _required(validation_cpu, val_keys, "validation")

    backbone = GCNBackbone(
        int(model_cfg["input_dim"]), int(model_cfg.get("hidden_dim", 128)),
        int(model_cfg.get("output_dim", 64)),
        input_mode=str(model_cfg.get("input_mode", "feature_matrix")),
        num_nodes=model_cfg.get("num_nodes"), cached=True,
    )
    state = bundle.get("backbone_state_dict")
    if not isinstance(state, dict):
        raise KeyError("bundle must contain backbone_state_dict")
    backbone.load_state_dict(state, strict=True)
    model = FullRankGCN(backbone).to(device)
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    expected = [("repair1.W", 128 * 128), ("repair2.W", 64 * 64)]
    actual = [(name, int(p.numel())) for name, p in trainable]
    if actual != expected:
        raise RuntimeError(f"unexpected FullRank trainable set: {actual}")
    if any(torch.count_nonzero(p).item() for _, p in trainable):
        raise RuntimeError("FullRank repairs must start from zero")
    optimizer = torch.optim.Adam([p for _, p in trainable], lr=float(config.get("lr", 3e-4)),
                                 weight_decay=float(config.get("weight_decay", 0.0)))

    schedule_source = training_cpu["retain_negative_schedule"]
    schedule = (schedule_source if isinstance(schedule_source, RetainNegativeSchedule)
                else RetainNegativeSchedule(schedule_source))
    training = _to_device({k: v for k, v in training_cpu.items()
                           if k != "retain_negative_schedule"}, device)
    validation = GCNValidationBundle(**_to_device(validation_cpu, device))
    validate_gates(training["df"].flatten(), training["support1"], training["support2"],
                   training["gate1"], training["gate2"])
    if validation.retained_control_groups.size(0) != 500:
        raise ValueError("formal GCN bundle must contain 500 retained-control groups")
    x = training["x"]
    graph = training["dr_mp_edge_index"]
    backbone.eval()
    with torch.no_grad():
        h0 = backbone.common_input_preparation(x)
        base_h1 = model.conv1(h0, graph).detach()
        base_z = model.conv2(backbone.official_intermediate_processing(base_h1), graph).detach()
        training["teacher_local_probability"] = torch.sigmoid(
            model.decode(base_z, training["local"])
        ).detach()
        training["teacher_anchor_probability"] = torch.sigmoid(
            model.decode(base_z, training["anchor"])
        ).detach()
    if schedule.size(0) < int(config.get("updates", 3000)):
        raise ValueError("retain_negative_schedule is shorter than the configured update budget")
    schedule.validate()
    identity_cfg = config.get("sampling_identity")
    if not isinstance(identity_cfg, dict):
        raise KeyError("sampling_identity must contain the exact formal sampler identity")
    _required(identity_cfg, ("dataset", "backbone", "deletion_mode", "deletion_ratio", "run_seed"),
              "sampling_identity")
    if int(identity_cfg["run_seed"]) != int(config["seed"]):
        raise ValueError("sampling_identity.run_seed must match config.seed")
    identity = SamplingIdentity(
        dataset=str(identity_cfg["dataset"]), backbone=str(identity_cfg["backbone"]),
        deletion_mode=str(identity_cfg["deletion_mode"]),
        deletion_ratio=str(identity_cfg["deletion_ratio"]), run_seed=int(identity_cfg["run_seed"]),
    )
    sampler = ForgetNegativeSampler(
        training["full_train_mp_edge_index"], int(x.size(0)), int(training["df"].size(1)), identity,
    )
    group_names = ("df", "forget_negative", "local", "retain_positive", "retain_negative", "anchor")
    sizes = {
        "df": int(training["df"].size(1)), "forget_negative": int(training["df"].size(1)),
        "local": int(training["local"].size(1)),
        "retain_positive": int(training["retain_positive"].size(1)),
        "retain_negative": int(schedule.size(2)),
        "anchor": int(training["anchor"].size(1)),
    }
    slices, cursor = {}, 0
    for name in group_names:
        slices[name] = slice(cursor, cursor + sizes[name])
        cursor += sizes[name]
    edge_buffer = torch.empty((2, cursor), dtype=torch.long, device=device)
    for name in ("df", "local", "retain_positive", "anchor"):
        edge_buffer[:, slices[name]].copy_(training[name])
    selector = GCNSelector()
    return PreparedGCN(config, bundle, device, model, optimizer, base_h1,
                       training, validation, sampler, schedule,
                       edge_buffer, slices, sizes, selector)


def _update(prepared: PreparedGCN, update: int, *,
            forget_negative: torch.Tensor | None = None,
            capture: bool = False) -> dict[str, torch.Tensor] | None:
    model, train, device = prepared.model, prepared.training, prepared.device
    if forget_negative is None:
        forget_negative = prepared.sampler.sample(update)
    forget_negative = forget_negative.to(device)
    retain_negative = prepared.schedule[update].to(device)
    prepared.edge_buffer[:, prepared.slices["forget_negative"]].copy_(forget_negative)
    prepared.edge_buffer[:, prepared.slices["retain_negative"]].copy_(retain_negative)
    optimizer = prepared.optimizer
    optimizer.zero_grad(set_to_none=True)
    h1 = prepared.base_h1 + train["gate1"] * prepared.base_h1.matmul(model.repair1.W)
    h2_base = model.conv2(model.backbone.official_intermediate_processing(h1), train["dr_mp_edge_index"])
    z = h2_base + train["gate2"] * h2_base.matmul(model.repair2.W)
    logits = model.decode(z, prepared.edge_buffer)
    groups = prepared.slices
    rp, rn = logits[groups["retain_positive"]], logits[groups["retain_negative"]]
    loss, parts = p2_objective(
        forget_positive_logits=logits[groups["df"]],
        forget_reference_logits=logits[groups["forget_negative"]],
        local_logits=logits[groups["local"]],
        teacher_local_probability=train["teacher_local_probability"],
        retain_positive_logits=rp, retain_negative_logits=rn,
        anchor_logits=logits[groups["anchor"]],
        teacher_anchor_probability=train["teacher_anchor_probability"],
        weights=P2_WEIGHTS, margin=P2_MARGIN,
    )
    loss.backward()
    components = {name: value.detach() for name, value in parts.items()} if capture else None
    optimizer.step()
    return components


@torch.no_grad()
def _validate(prepared: PreparedGCN) -> dict[str, float]:
    t = prepared.training
    h1 = prepared.base_h1 + t["gate1"] * prepared.base_h1.matmul(prepared.model.repair1.W)
    h2 = prepared.model.conv2(prepared.model.backbone.official_intermediate_processing(h1),
                              t["dr_mp_edge_index"])
    z = h2 + t["gate2"] * h2.matmul(prepared.model.repair2.W)
    return evaluate_gcn(prepared.model, z, prepared.validation)


def train_prepared_gcn(prepared: PreparedGCN) -> dict[str, Any]:
    updates = int(prepared.config.get("updates", 3000))
    interval = int(prepared.config.get("validation_interval", 100))
    if updates <= 0 or interval <= 0 or updates % interval:
        raise ValueError("updates must be positive and divisible by validation_interval")
    trajectory = []
    for update in range(updates):
        prepared.model.train()
        _update(prepared, update)
        if (update + 1) % interval == 0:
            prepared.model.eval()
            metrics = _validate(prepared)
            prepared.selector.consider(update + 1, metrics, prepared.model)
    prepared.selector.restore(prepared.model)
    return {
        "selected_update": prepared.selector.best_update,
        "selected_metrics": prepared.selector.best_metrics,
        "selected_repair_state": prepared.selector.best_state,
        "validation_trajectory": prepared.selector.trajectory,
        "updates": updates,
        "validation_count": len(prepared.selector.trajectory),
        "optimizer": {"name": "Adam", "lr": float(prepared.optimizer.defaults["lr"]),
                      "weight_decay": float(prepared.optimizer.defaults["weight_decay"])},
        "trainable_parameters": sum(p.numel() for p in prepared.model.parameters() if p.requires_grad),
    }


def train_gcn(config: dict[str, Any], bundle: dict[str, Any], device: torch.device | str = "cpu") -> dict[str, Any]:
    return train_prepared_gcn(prepare_gcn(config, bundle, device))


@torch.no_grad()
def evaluate_gcn_test(prepared: PreparedGCN, test_bundle: dict[str, torch.Tensor]) -> dict[str, float]:
    values = _to_device(test_bundle, prepared.device)
    bundle = GCNValidationBundle(**values)
    t = prepared.training
    h1 = prepared.base_h1 + t["gate1"] * prepared.base_h1.matmul(prepared.model.repair1.W)
    h2 = prepared.model.conv2(prepared.model.backbone.official_intermediate_processing(h1),
                              t["dr_mp_edge_index"])
    z = h2 + t["gate2"] * h2.matmul(prepared.model.repair2.W)
    return evaluate_gcn(prepared.model, z, bundle)
