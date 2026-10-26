from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .metrics import dt_metrics, grouped_df_metrics, grouped_df_metrics_vectorized


@dataclass
class GCNValidationBundle:
    positive: torch.Tensor
    negative: torch.Tensor
    requested: torch.Tensor
    retained_control_groups: torch.Tensor  # [500, 2, M]

    def __post_init__(self) -> None:
        if self.retained_control_groups.ndim != 3 or self.retained_control_groups.size(0) != 500:
            raise ValueError("retained_control_groups must have shape [500,2,M]")


@dataclass
class RGCNValidationBundle:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    gate1: torch.Tensor
    gate2: torch.Tensor
    val_positive: torch.Tensor
    val_negative: torch.Tensor
    val_relation: torch.Tensor
    requested: torch.Tensor
    requested_relation: torch.Tensor
    retained_control_groups: torch.Tensor  # [500, 2, M]
    retained_control_relations: torch.Tensor  # [500, M]

    def __post_init__(self) -> None:
        if self.retained_control_groups.ndim != 3 or self.retained_control_groups.size(0) != 500:
            raise ValueError("retained_control_groups must have shape [500,2,M]")


@torch.no_grad()
def evaluate_gcn(model, z: torch.Tensor, bundle: GCNValidationBundle) -> dict[str, float]:
    device = z.device
    positive = bundle.positive.to(device)
    negative = bundle.negative.to(device)
    requested = bundle.requested.to(device)
    controls = bundle.retained_control_groups.to(device)
    dt_auc, dt_aup = dt_metrics(model.decode(z, positive).cpu().numpy(),
                                model.decode(z, negative).cpu().numpy())
    requested_scores = model.decode(z, requested).cpu().numpy()
    retained = model.decode(z, controls.permute(1, 0, 2).reshape(2, -1)).reshape(500, -1).cpu().numpy()
    df_auc, df_aup = grouped_df_metrics(requested_scores, retained)
    return {"DT_AUC": dt_auc, "DT_AUP": dt_aup, "Df_AUC": df_auc, "Df_AUP": df_aup,
            "J_AUC": (dt_auc + df_auc) / 2, "J_AUP": (dt_aup + df_aup) / 2}


@torch.no_grad()
def evaluate_rgcn(model, z: torch.Tensor, bundle: RGCNValidationBundle) -> dict[str, float]:
    device = z.device
    positive, negative = bundle.val_positive.to(device), bundle.val_negative.to(device)
    requested, requested_type = bundle.requested.to(device), bundle.requested_relation.to(device)
    val_type = bundle.val_relation.to(device)
    dt_auc, dt_aup = dt_metrics(
        model.decode(z, positive, val_type).cpu().numpy(),
        model.decode(z, negative, val_type).cpu().numpy(),
    )
    requested_scores = model.decode(z, requested, requested_type).cpu().numpy()
    groups = bundle.retained_control_groups.to(device)
    group_types = bundle.retained_control_relations.to(device)
    retained = model.decode(z, groups.permute(1, 0, 2).reshape(2, -1), group_types.reshape(-1))
    retained = retained.reshape(500, -1).cpu().numpy()
    df_auc, df_aup = grouped_df_metrics_vectorized(requested_scores, retained)
    return {"DT_AUC": dt_auc, "DT_AUP": dt_aup, "Df_AUC": df_auc, "Df_AUP": df_aup,
            "J_AUC": (dt_auc + df_auc) / 2, "J_AUP": (dt_aup + df_aup) / 2}


@torch.no_grad()
def evaluate_gat(model, z: torch.Tensor, teacher_z: torch.Tensor,
                 bundle: GCNValidationBundle) -> dict[str, float]:
    """Phase 3C/3D metrics, including the retained-MSE selector diagnostic."""
    metrics = evaluate_gcn(model, z, bundle)
    dt_auc, dt_aup = dt_metrics(
        model.decode(z, bundle.positive.to(z.device)).sigmoid().cpu().numpy(),
        model.decode(z, bundle.negative.to(z.device)).sigmoid().cpu().numpy(),
    )
    edges = bundle.retained_control_groups.to(z.device).permute(1, 0, 2).reshape(2, -1)
    student = model.decode(z, edges).sigmoid()
    teacher = model.backbone.decode(teacher_z, edges).sigmoid()
    metrics.update(DT_AUC=dt_auc, DT_AUP=dt_aup,
                   J_AUC=(dt_auc + metrics["Df_AUC"]) / 2,
                   J_AUP=(dt_aup + metrics["Df_AUP"]) / 2,
                   retain_mse=float(torch.nn.functional.mse_loss(student, teacher)))
    return metrics
