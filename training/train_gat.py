from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from lrr.evaluation.evaluator import GCNValidationBundle, evaluate_gat
from lrr.gates import validate_gates
from lrr.models.gat import GATBackbone, FullRankGAT
from lrr.objectives.p2 import p2_objective
from lrr.sampling.homogeneous import ForgetNegativeSampler, SamplingIdentity, RetainNegativeSchedule
from lrr.selectors import GATSelector
from training.train_gcn import _required, _to_device


@dataclass
class PreparedGAT:
    config: dict[str, Any]
    model: FullRankGAT
    training: dict[str, torch.Tensor]
    validation: GCNValidationBundle
    optimizer: torch.optim.Optimizer
    sampler: ForgetNegativeSampler
    schedule: RetainNegativeSchedule
    selector: GATSelector
    base_h1: torch.Tensor
    teacher_z: torch.Tensor
    device: torch.device


def prepare_gat(config, bundle, device="cpu") -> PreparedGAT:
    device = torch.device(device)
    if "sampling_identity" in bundle and bundle["sampling_identity"] != config["sampling_identity"]:
        raise ValueError("bundle and config sampling identities differ")
    if bundle["model"].get("architecture_version", "COMMON-GAT-v1") != "COMMON-GAT-v1":
        raise ValueError("GAT requires COMMON-GAT-v1")
    backbone = GATBackbone(int(bundle["model"]["input_dim"]))
    backbone.load_state_dict(bundle["backbone_state_dict"], strict=True)
    model = FullRankGAT(backbone).to(device)
    raw = bundle["training"]
    _required(raw, ("x", "full_train_mp_edge_index", "dr_mp_edge_index", "df", "local",
                    "retain_positive", "anchor", "support1", "support2", "gate1", "gate2",
                    "teacher_local_probability", "teacher_anchor_probability",
                    "retain_negative_schedule"), "GAT training")
    training = _to_device({k: v for k, v in raw.items() if k != "retain_negative_schedule"}, device)
    if training["x"].ndim != 2 or not training["x"].is_floating_point():
        raise TypeError("Common-GAT requires floating feature_matrix input")
    validate_gates(training["df"].flatten(), training["support1"], training["support2"],
                   training["gate1"], training["gate2"])
    schedule = raw["retain_negative_schedule"]
    if not isinstance(schedule, RetainNegativeSchedule):
        schedule = RetainNegativeSchedule(schedule)
    if schedule.size(0) < int(config.get("updates", 3000)):
        raise ValueError("retained-negative schedule is shorter than the update budget")
    schedule.validate()
    identity = SamplingIdentity(**config["sampling_identity"])
    if identity.backbone != "gat" or identity.run_seed != int(config["seed"]):
        raise ValueError("GAT online Forget sampler identity must match gat/config seed")
    sampler = ForgetNegativeSampler(training["full_train_mp_edge_index"], training["x"].size(0),
                                    training["df"].size(1), identity)
    validation = GCNValidationBundle(**_to_device(bundle["validation"], device))
    with torch.no_grad():
        h1 = backbone.conv1(training["x"], training["dr_mp_edge_index"]).detach()
        teacher_z = backbone(training["x"], training["dr_mp_edge_index"]).detach()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=float(config.get("lr", 3e-4)),
                                 weight_decay=float(config.get("weight_decay", 0.0)))
    return PreparedGAT(config, model, training, validation, optimizer, sampler, schedule,
                       GATSelector(), h1, teacher_z, device)


def _embedding(prepared):
    t = prepared.training
    return prepared.model.from_layer1(prepared.base_h1, t["dr_mp_edge_index"], t["gate1"], t["gate2"])


def _update(prepared, update, *, forget_negative=None, capture=False):
    model, t = prepared.model, prepared.training
    if forget_negative is None:
        forget_negative = prepared.sampler.sample(update)
    prepared.optimizer.zero_grad(set_to_none=True)
    z = _embedding(prepared)
    score = lambda edges: model.decode(z, edges.to(prepared.device))
    loss, parts = p2_objective(
        forget_positive_logits=score(t["df"]), forget_reference_logits=score(forget_negative),
        local_logits=score(t["local"]), teacher_local_probability=t["teacher_local_probability"],
        retain_positive_logits=score(t["retain_positive"]),
        retain_negative_logits=score(prepared.schedule[update]),
        anchor_logits=score(t["anchor"]), teacher_anchor_probability=t["teacher_anchor_probability"])
    loss.backward()
    if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all()
                                      for p in (model.repair1.W, model.repair2.W)):
        raise RuntimeError("nonfinite GAT loss or missing/nonfinite repair gradient")
    if any(p.grad is not None for p in model.backbone.parameters()):
        raise RuntimeError("frozen GAT backbone received gradients")
    result = {key: value.detach().clone() for key, value in parts.items()}
    prepared.optimizer.step()
    return result if capture else float(loss.detach())


def train_prepared_gat(prepared) -> dict[str, Any]:
    updates = int(prepared.config.get("updates", 3000))
    interval = int(prepared.config.get("validation_interval", 100))
    if updates <= 0 or interval <= 0 or updates % interval:
        raise ValueError("updates must be positive and divisible by validation_interval")
    for update in range(updates):
        prepared.model.train()
        loss = _update(prepared, update)
        if (update + 1) % interval == 0:
            prepared.model.eval()
            metrics = evaluate_gat(prepared.model, _embedding(prepared),
                                   prepared.teacher_z, prepared.validation)
            # The source's validation_total_loss is the current pre-update training loss.
            prepared.selector.consider(update + 1, metrics, prepared.model, validation_total_loss=loss)
    prepared.selector.restore(prepared.model)
    return {"selected_update": prepared.selector.best_update,
            "selected_epoch": prepared.selector.best_update - 1,
            "selected_metrics": prepared.selector.best_metrics,
            "selected_repair_state": prepared.selector.best_state,
            "validation_trajectory": prepared.selector.trajectory, "updates": updates,
            "validation_count": len(prepared.selector.trajectory), "trainable_parameters": 20480}


def train_gat(config, bundle, device="cpu"):
    return train_prepared_gat(prepare_gat(config, bundle, device))


@torch.no_grad()
def evaluate_gat_test(prepared, test_bundle):
    return evaluate_gat(prepared.model, _embedding(prepared), prepared.teacher_z,
                        GCNValidationBundle(**_to_device(test_bundle, prepared.device)))
