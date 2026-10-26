from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch


def _selected_state(model) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()
            if name in {"repair1.W", "repair2.W"}}


@dataclass
class GCNSelector:
    """Maximum validation J-AUC; chronological evaluation gives earlier ties."""

    best_score: float = float("-inf")
    best_update: int | None = None
    best_metrics: dict[str, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    trajectory: list[dict[str, float | int]] = field(default_factory=list)

    def consider(self, update: int, metrics: dict[str, float], model) -> bool:
        row = {"update": int(update), **{key: float(metrics[key]) for key in
                ("DT_AUC", "DT_AUP", "Df_AUC", "Df_AUP", "J_AUC", "J_AUP")}}
        self.trajectory.append(row)
        score = row["J_AUC"]
        if score > self.best_score:
            self.best_score, self.best_update = score, int(update)
            self.best_metrics, self.best_state = dict(row), _selected_state(model)
            return True
        return False

    def restore(self, model) -> None:
        if self.best_state is None:
            raise RuntimeError("GCN selector has no selected state")
        with torch.no_grad():
            model.repair1.W.copy_(self.best_state["repair1.W"])
            model.repair2.W.copy_(self.best_state["repair2.W"])


@dataclass
class GATSelector(GCNSelector):
    """Exact Phase 3C S6 lexicographic selector, tolerance 1e-12."""

    best_key: tuple[float, ...] | None = None

    def consider(self, update: int, metrics: dict[str, float], model, *,
                 validation_total_loss: float) -> bool:
        key = (metrics["DT_AUC"] + metrics["Df_AUC"], metrics["DT_AUC"],
               metrics["Df_AUC"], -metrics["retain_mse"],
               -float(validation_total_loss), -int(update))
        if not all(math.isfinite(value) for value in key):
            raise ValueError("GAT selector metrics and loss must be finite")
        row = {"update": int(update), "epoch": int(update) - 1, **metrics,
               "validation_total_loss": float(validation_total_loss)}
        self.trajectory.append(row)
        replaces = self.best_key is None
        if self.best_key is not None:
            for candidate, incumbent in zip(key, self.best_key):
                if candidate > incumbent + 1e-12:
                    replaces = True
                    break
                if candidate < incumbent - 1e-12:
                    break
        if replaces:
            self.best_key, self.best_score = key, key[0]
            self.best_update, self.best_metrics = int(update), dict(row)
            self.best_state = _selected_state(model)
        return replaces


@dataclass
class RGCNSelector:
    """Update-0-relative constrained selector from the paper-main protocol."""

    epsilon: float = 0.010
    dt0: float | None = None
    threshold: float | None = None
    best_df: float = float("-inf")
    best_update: int | None = None
    best_metrics: dict[str, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    trajectory: list[dict[str, float | int]] = field(default_factory=list)

    def consider(self, update: int, metrics: dict[str, float], model) -> bool:
        update = int(update)
        if update == 0:
            if self.dt0 is not None:
                raise RuntimeError("update-0 selector reference already set")
            self.dt0 = float(metrics["DT_AUC"])
            self.threshold = self.dt0 - self.epsilon
        elif self.dt0 is None:
            raise RuntimeError("RGCN selector requires update 0 first")
        row = {"update": update, **{key: float(metrics[key]) for key in
                ("DT_AUC", "DT_AUP", "Df_AUC", "Df_AUP", "J_AUC", "J_AUP")}}
        self.trajectory.append(row)
        if row["DT_AUC"] < float(self.threshold) or row["Df_AUC"] <= self.best_df:
            return False
        self.best_df = row["Df_AUC"]
        self.best_update = update
        self.best_metrics = dict(row)
        self.best_state = _selected_state(model)
        return True

    def restore(self, model) -> None:
        if self.best_state is None:
            raise RuntimeError("RGCN selector has no feasible selected state")
        with torch.no_grad():
            model.repair1.W.copy_(self.best_state["repair1.W"])
            model.repair2.W.copy_(self.best_state["repair2.W"])
