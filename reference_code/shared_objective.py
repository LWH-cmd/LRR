"""Shared objective and training core for matched GNNDelete/LRR."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from ..lrr_runtime import tensor_hash_v1
from ..precomputed_schedule import (
    FrozenRetainNegativeSchedule,
    sample_negative_role_online,
)
from ..precomputed_schedule_identity import (
    file_sha256,
    profile_contract_hash,
    runtime_schedule_identity,
    support_identity_hash,
    validate_source_configuration,
)
from ..sampling import (
    EvaluationCandidateBundle,
    MAX_RETAIN_EDGES,
    SharedSampler,
    StaticSampleBundle,
)
from ..teacher_target_bundle import (
    load_authoritative_teacher_target_bundle,
)


OBJECTIVE_VERSION = "shared_lrr_objective_v1"
TRAINING_BUDGET_VERSION = "matched_training_budget_v1"
BUDGET_PROFILE_ID = "gnndelete_hg_gcn_non_ogb_matched_v1"
VALIDATION_SELECTION_VERSION = "val_dt_auc_plus_val_df_auc_v1"
VALIDATION_RETRAIN_DISTANCE_VERSION = "retrain_validation_distance_v1"
OBJECTIVE_WEIGHTS = {
    "forget": 0.5,
    "local": 0.5,
    "retain": 0.2,
    "anchor": 0.05,
    "reg": 0.0,
}
FORGET_MARGIN = 0.05


def legacy_validation_selection_score(metrics: Mapping[str, float]) -> float:
    score = float(metrics["dt_auc"]) + float(metrics["df_auc"])
    if not math.isfinite(score):
        raise RuntimeError("invalid validation selection score")
    return score


def strict_maximum_replaces(candidate: float, incumbent: float) -> bool:
    return math.isfinite(float(candidate)) and float(candidate) > float(incumbent)


def strict_minimum_replaces(candidate: float, incumbent: float) -> bool:
    return math.isfinite(float(candidate)) and float(candidate) < float(incumbent)


@dataclass(frozen=True)
class SharedRuntimeContext:
    x: torch.Tensor
    full_train_mp_edge_index: torch.Tensor
    dr_mp_edge_index: torch.Tensor
    directed_df_edge_index: torch.Tensor
    support1: torch.Tensor
    support2: torch.Tensor
    gate1: torch.Tensor
    gate2: torch.Tensor
    local_loss_edge_mask: torch.Tensor
    graph_identity: Dict[str, object]


@dataclass(frozen=True)
class TeacherTargetBundle:
    local_probability: torch.Tensor
    anchor_probability: torch.Tensor
    teacher_embedding_hash: str
    local_probability_hash: str
    anchor_probability_hash: str


def _state_hash(
    state: Mapping[str, torch.Tensor], *, prefixes: Optional[Tuple[str, ...]] = None
) -> str:
    entries = []
    for name in sorted(state):
        if prefixes is not None and not name.startswith(prefixes):
            continue
        entries.append((name, tensor_hash_v1(state[name])))
    payload = json.dumps(entries, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def shared_lrr_objective(
    *,
    model,
    student_z: torch.Tensor,
    directed_df_edge_index: torch.Tensor,
    forget_negative_edge_index: torch.Tensor,
    local_edges: torch.Tensor,
    teacher_local_probability: torch.Tensor,
    retain_positive_edges: torch.Tensor,
    retain_negative_edges: torch.Tensor,
    anchor_edges: torch.Tensor,
    teacher_anchor_probability: torch.Tensor,
    objective_weights: Optional[Mapping[str, float]] = None,
    margin: float = FORGET_MARGIN,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    weights = (
        dict(OBJECTIVE_WEIGHTS)
        if objective_weights is None
        else {name: float(value) for name, value in objective_weights.items()}
    )
    if set(weights) != set(OBJECTIVE_WEIGHTS):
        raise ValueError("objective weight keys do not match Shared-LRR-Objective")
    if weights["reg"] != 0.0:
        raise ValueError("Phase 1B Shared-LRR-Objective requires reg_weight=0")
    if not math.isfinite(float(margin)):
        raise ValueError("forget margin must be finite")
    required_edges = {
        "directed_df_edge_index": directed_df_edge_index,
        "forget_negative_edge_index": forget_negative_edge_index,
        "local_edges": local_edges,
        "retain_positive_edges": retain_positive_edges,
        "retain_negative_edges": retain_negative_edges,
        "anchor_edges": anchor_edges,
    }
    for name, edges in required_edges.items():
        if edges.dim() != 2 or edges.size(0) != 2 or edges.size(1) == 0:
            raise RuntimeError(f"non-zero objective requires non-empty {name}")

    forget_positive = model.decode(student_z, directed_df_edge_index)
    forget_negative = model.decode(student_z, forget_negative_edge_index)
    if forget_positive.numel() != forget_negative.numel():
        raise RuntimeError("forget positive/negative counts must be equal")
    forget_margin = forget_positive - forget_negative.detach() + float(margin)
    forget_loss = F.softplus(forget_margin).mean()

    student_local_probability = torch.sigmoid(
        model.decode(student_z, local_edges)
    )
    if student_local_probability.shape != teacher_local_probability.shape:
        raise RuntimeError("student/teacher local target shapes differ")
    local_loss = F.mse_loss(
        student_local_probability, teacher_local_probability
    )

    retain_positive_logits = model.decode(student_z, retain_positive_edges)
    retain_negative_logits = model.decode(student_z, retain_negative_edges)
    raw_retain_logits = torch.cat(
        [retain_positive_logits, retain_negative_logits], dim=0
    )
    retain_labels = torch.cat(
        [
            torch.ones_like(retain_positive_logits),
            torch.zeros_like(retain_negative_logits),
        ],
        dim=0,
    )
    retain_loss = F.binary_cross_entropy_with_logits(
        raw_retain_logits, retain_labels
    )

    student_anchor_probability = torch.sigmoid(
        model.decode(student_z, anchor_edges)
    )
    if student_anchor_probability.shape != teacher_anchor_probability.shape:
        raise RuntimeError("student/teacher anchor target shapes differ")
    anchor_loss = F.mse_loss(
        student_anchor_probability, teacher_anchor_probability
    )

    total_loss = (
        weights["forget"] * forget_loss
        + weights["local"] * local_loss
        + weights["retain"] * retain_loss
        + weights["anchor"] * anchor_loss
    )
    all_losses = [forget_loss, local_loss, retain_loss, anchor_loss, total_loss]
    if not all(bool(torch.isfinite(value).item()) for value in all_losses):
        raise RuntimeError("Shared-LRR-Objective produced NaN/Inf")

    values = {
        "total_loss": float(total_loss.detach()),
        "forget_loss": float(forget_loss.detach()),
        "local_loss": float(local_loss.detach()),
        "retain_loss": float(retain_loss.detach()),
        "anchor_loss": float(anchor_loss.detach()),
        "reg_loss": 0.0,
        "forget_positive_score_mean": float(forget_positive.detach().mean()),
        "forget_negative_score_mean": float(forget_negative.detach().mean()),
        "forget_margin_violation_rate": float(
            (forget_margin.detach() > 0).float().mean()
        ),
        "local_edge_count": int(local_edges.size(1)),
        "student_local_prob_mean": float(
            student_local_probability.detach().mean()
        ),
        "teacher_local_prob_mean": float(
            teacher_local_probability.detach().mean()
        ),
        "retain_positive_count": int(retain_positive_edges.size(1)),
        "retain_negative_count": int(retain_negative_edges.size(1)),
        "retain_positive_logit_mean": float(
            retain_positive_logits.detach().mean()
        ),
        "retain_negative_logit_mean": float(
            retain_negative_logits.detach().mean()
        ),
        "anchor_total_count": int(anchor_edges.size(1)),
    }
    return total_loss, values


class SharedObjectiveTrainer:
    """Common full-batch training loop for matched GNNDelete and LRR."""

    method_identity = "shared_matched_method"
    checkpoint_schema_version = "shared_matched_checkpoint_v1"

    def __init__(self, args) -> None:
        self.args = args

    def student_forward(self, model, runtime: SharedRuntimeContext) -> torch.Tensor:
        raise NotImplementedError

    def trainable_parameters(self, model):
        raise NotImplementedError

    def expected_trainable_parameter_names(self, model):
        raise NotImplementedError

    def validate_epoch_gradients(self, epoch, gradient_norms) -> None:
        if any(value <= 0 for value in gradient_norms.values()):
            raise RuntimeError(f"operator gradient is inactive at epoch {epoch}")

    def validate_method_model(self, model) -> None:
        raise NotImplementedError

    @torch.no_grad()
    def build_teacher_targets(
        self,
        teacher,
        runtime: SharedRuntimeContext,
        static_bundle: StaticSampleBundle,
    ) -> TeacherTargetBundle:
        teacher.eval()
        teacher_z = teacher(runtime.x, runtime.dr_mp_edge_index)
        local_edges = static_bundle.local_static.to(runtime.x.device)
        anchor_edges = static_bundle.anchor_static.to(runtime.x.device)
        local_probability = torch.sigmoid(
            teacher.decode(teacher_z, local_edges)
        ).detach()
        anchor_probability = torch.sigmoid(
            teacher.decode(teacher_z, anchor_edges)
        ).detach()
        return TeacherTargetBundle(
            local_probability=local_probability,
            anchor_probability=anchor_probability,
            teacher_embedding_hash=tensor_hash_v1(teacher_z),
            local_probability_hash=tensor_hash_v1(local_probability),
            anchor_probability_hash=tensor_hash_v1(anchor_probability),
        )

    @torch.no_grad()
    def evaluate(
        self,
        model,
        runtime: SharedRuntimeContext,
        evaluation_bundle: EvaluationCandidateBundle,
        dr_mp_edge_index_cpu: torch.Tensor,
        *,
        split: str,
    ) -> Dict[str, float]:
        model.eval()
        student_z = self.student_forward(model, runtime)
        if split == "validation":
            positive = evaluation_bundle.validation_pos
            negative = evaluation_bundle.validation_neg
        elif split == "test":
            positive = evaluation_bundle.test_pos
            negative = evaluation_bundle.test_neg
        else:
            raise ValueError("split must be validation or test")
        positive = positive.to(runtime.x.device)
        negative = negative.to(runtime.x.device)
        positive_logits = model.decode(student_z, positive)
        negative_logits = model.decode(student_z, negative)
        raw_dt_logits = torch.cat([positive_logits, negative_logits], dim=0)
        dt_labels = torch.cat(
            [torch.ones_like(positive_logits), torch.zeros_like(negative_logits)]
        )
        dt_loss = F.binary_cross_entropy_with_logits(
            raw_dt_logits, dt_labels
        )
        dt_probabilities = torch.sigmoid(raw_dt_logits).cpu().numpy()
        dt_labels_numpy = dt_labels.cpu().numpy()
        dt_auc = float(roc_auc_score(dt_labels_numpy, dt_probabilities))
        dt_aup = float(
            average_precision_score(dt_labels_numpy, dt_probabilities)
        )

        df_edges = runtime.directed_df_edge_index
        df_logits = model.decode(student_z, df_edges).detach().cpu()
        group_positions = evaluation_bundle.df_retained_group_positions
        flat_positions = group_positions.reshape(-1)
        group_edges = dr_mp_edge_index_cpu[:, flat_positions].to(runtime.x.device)
        group_logits = (
            model.decode(student_z, group_edges)
            .detach()
            .cpu()
            .reshape(group_positions.shape)
        )
        df_values = df_logits.numpy()
        df_auc_values = []
        df_aup_values = []
        labels = np.concatenate(
            [
                np.zeros(df_logits.numel(), dtype=np.int64),
                np.ones(df_logits.numel(), dtype=np.int64),
            ]
        )
        for group in group_logits:
            values = np.concatenate([df_values, group.numpy()])
            df_auc_values.append(roc_auc_score(labels, values))
            df_aup_values.append(average_precision_score(labels, values))
        result = {
            f"{split}_loss": float(dt_loss),
            "dt_auc": dt_auc,
            "dt_aup": dt_aup,
            "df_auc": float(np.mean(df_auc_values)),
            "df_aup": float(np.mean(df_aup_values)),
            "df_logit_mean": float(df_logits.mean()),
            "df_logit_std": float(df_logits.std(unbiased=False)),
            "dr_control_logit_mean": float(group_logits.mean()),
            "dr_control_logit_std": float(
                group_logits.std(unbiased=False)
            ),
        }
        if not all(math.isfinite(value) for value in result.values()):
            raise RuntimeError(f"{split} evaluation produced NaN/Inf")
        return result

    @staticmethod
    def validation_retrain_distance(
        metrics: Mapping[str, float],
        target: Mapping[str, float],
    ) -> float:
        fields = (
            "dt_auc",
            "dt_aup",
            "df_auc",
            "df_aup",
            "df_logit_mean",
            "df_logit_std",
            "dr_control_logit_mean",
            "dr_control_logit_std",
        )
        terms = []
        for field in fields:
            if field not in target:
                raise RuntimeError(
                    f"Retrain validation target is missing {field}"
                )
            scale = (
                1.0
                if field.endswith(("auc", "aup"))
                else max(1.0, abs(float(target[field])))
            )
            terms.append(
                ((float(metrics[field]) - float(target[field])) / scale) ** 2
            )
        return math.sqrt(sum(terms) / len(terms))

    def _checkpoint(
        self,
        *,
        model,
        optimizer,
        epoch: int,
        role: str,
        resolved_profile: Dict[str, object],
        sampler: SharedSampler,
        static_bundle: StaticSampleBundle,
        evaluation_bundle: EvaluationCandidateBundle,
        runtime: SharedRuntimeContext,
        teacher_hash: str,
        backbone_hash: str,
        validation_metrics: Optional[Dict[str, float]],
    ) -> Dict[str, object]:
        return {
            "checkpoint_schema_version": self.checkpoint_schema_version,
            "method_identity": self.method_identity,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": None,
            "epoch": int(epoch),
            "checkpoint_role": role,
            "resolved_profile": resolved_profile,
            "sampler_deterministic_state": {
                "identity": sampler.identity.__dict__,
                "next_epoch": int(epoch) + 1,
                "step": 0,
                "matched_sampler_version": static_bundle.metadata[
                    "matched_sampler_version"
                ],
            },
            "input_identities": runtime.graph_identity,
            "static_bundle_sha256": static_bundle.metadata[
                "static_bundle_sha256"
            ],
            "evaluation_candidate_bundle_sha256": evaluation_bundle.metadata[
                "total_bundle_sha256"
            ],
            "teacher_state_sha256": teacher_hash,
            "backbone_state_sha256": backbone_hash,
            "validation_metrics": validation_metrics,
        }

    def run(
        self,
        *,
        model,
        teacher,
        runtime: SharedRuntimeContext,
        sampler: SharedSampler,
        static_bundle: StaticSampleBundle,
        evaluation_bundle: EvaluationCandidateBundle,
        resolved_profile: Dict[str, object],
        output_dir: Path,
        checkpoint_dir: Path,
        smoke_only: bool,
        retrain_validation_target: Optional[Mapping[str, float]] = None,
    ) -> Dict[str, object]:
        end_to_end_started = time.perf_counter()
        runtime_accounting = {
            "setup_seconds": 0.0,
            "schedule_identity_validation_seconds": 0.0,
            "teacher_bundle_load_seconds": 0.0,
            "core_training_seconds": 0.0,
            "retain_negative_seconds": 0.0,
            "forget_negative_seconds": 0.0,
            "validation_seconds": 0.0,
            "checkpoint_serialization_seconds": 0.0,
            "final_evaluation_seconds": 0.0,
            "reporting_seconds": 0.0,
            "end_to_end_seconds": 0.0,
            "peak_gpu_memory_mb": 0.0,
            "peak_cpu_rss_mb": 0.0,
        }
        setup_started = time.perf_counter()
        self.validate_method_model(model)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        metrics_path = output_dir / "metrics.jsonl"

        profile_values = resolved_profile.get(
            "training_profile", resolved_profile
        )
        objective_weights = profile_values["objective_weights"]
        margin = float(profile_values["forget_margin"])
        learning_rate = float(profile_values["learning_rate"])
        profile_epochs = int(profile_values["epochs"])
        validation_frequency = int(
            profile_values["validation_frequency"]
        )
        diagnostic_candidates = bool(
            profile_values.get("diagnostic_checkpoint_candidates", False)
        )
        if diagnostic_candidates and retrain_validation_target is None:
            raise RuntimeError(
                "diagnostic checkpoint selectors require Retrain validation target"
            )
        epochs = 2 if smoke_only else profile_epochs
        validation_frequency = 1 if smoke_only else validation_frequency
        retain_source = str(
            getattr(self.args, "retain_negative_source", "online")
        )
        schedule_path = getattr(
            self.args, "retain_negative_schedule_path", None
        )
        teacher_source = str(
            getattr(self.args, "teacher_target_source", "runtime")
        )
        teacher_bundle_path = getattr(
            self.args, "teacher_target_bundle_path", None
        )
        validate_source_configuration(
            retain_negative_source=retain_source,
            retain_negative_schedule_path=schedule_path,
            teacher_target_source=teacher_source,
            teacher_target_bundle_path=teacher_bundle_path,
        )

        expected_names = tuple(
            self.expected_trainable_parameter_names(model)
        )
        named_parameters = dict(model.named_parameters())
        parameters = list(self.trainable_parameters(model))
        if tuple(
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ) != expected_names:
            raise RuntimeError("operator trainable name allowlist mismatch")
        if {id(named_parameters[name]) for name in expected_names} != {
            id(parameter) for parameter in parameters
        }:
            raise RuntimeError("operator trainable Parameter identity mismatch")
        optimizer = torch.optim.Adam(
            parameters, lr=learning_rate, weight_decay=0.0
        )
        optimizer_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        if optimizer_ids != {id(parameter) for parameter in parameters}:
            raise RuntimeError("optimizer Parameter identity allowlist mismatch")
        if len(parameters) != len(expected_names):
            raise RuntimeError("optimizer parameter count mismatch")

        schedule = None
        if retain_source == "precomputed":
            schedule_started = time.perf_counter()
            expected_schedule_identity = runtime_schedule_identity(
                dataset=sampler.identity.dataset,
                seed=sampler.identity.run_seed,
                graph_identity=runtime.graph_identity,
                node_count=sampler.num_nodes,
                epochs=epochs,
                retain_negative_count=min(
                    int(sampler.dr_mp_edge_index.size(1)),
                    int(MAX_RETAIN_EDGES),
                ),
                resolved_profile=resolved_profile,
            )
            schedule = FrozenRetainNegativeSchedule(
                Path(str(schedule_path)).resolve(),
                expected_schedule_identity,
            )
            runtime_accounting[
                "schedule_identity_validation_seconds"
            ] = time.perf_counter() - schedule_started

        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad = False
        teacher_hash_before = _state_hash(teacher.state_dict())
        backbone_hash_before = _state_hash(
            model.state_dict(), prefixes=("node_emb.", "conv1.", "conv2.")
        )
        if teacher_source == "runtime":
            teacher_targets = self.build_teacher_targets(
                teacher, runtime, static_bundle
            )
        else:
            bundle_started = time.perf_counter()
            expected_teacher_identity = {
                "dataset": sampler.identity.dataset,
                "seed": int(sampler.identity.run_seed),
                "df_hash": runtime.graph_identity["directed_df_sha256"],
                "dr_hash": runtime.graph_identity["dr_mp_sha256"],
                "support_hash": support_identity_hash(runtime.graph_identity),
                "original_checkpoint_sha256": file_sha256(
                    Path(str(self.args.original_checkpoint)).resolve()
                ),
                "static_bundle_sha256": static_bundle.metadata[
                    "static_bundle_sha256"
                ],
                "profile_contract_hash": profile_contract_hash(
                    resolved_profile
                ),
            }
            teacher_targets = load_authoritative_teacher_target_bundle(
                Path(str(teacher_bundle_path)).resolve(),
                expected_identity=expected_teacher_identity,
                device=runtime.x.device,
                expected_local_count=static_bundle.local_static.size(1),
                expected_anchor_count=static_bundle.anchor_static.size(1),
            )
            runtime_accounting[
                "teacher_bundle_load_seconds"
            ] = time.perf_counter() - bundle_started
        if any(parameter.grad is not None for parameter in teacher.parameters()):
            raise RuntimeError("teacher target creation must not build gradients")

        local_edges = static_bundle.local_static.to(runtime.x.device)
        retain_positive = static_bundle.retain_positive_static.to(
            runtime.x.device
        )
        anchor_edges = static_bundle.anchor_static.to(runtime.x.device)
        validation_history = []
        best_score = -math.inf
        best_epoch = None
        best_metrics = None
        best_dt_auc = -math.inf
        best_dt_epoch = None
        closest_retrain_distance = math.inf
        closest_retrain_epoch = None
        minimum_total_loss = math.inf
        minimum_total_epoch = None
        minimum_total_checkpoint = None
        start_time = time.time()
        runtime_accounting["setup_seconds"] = (
            time.perf_counter() - setup_started
        )
        if runtime.x.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(runtime.x.device)

        def timed_save(value, path: Path) -> None:
            save_started = time.perf_counter()
            torch.save(value, path)
            runtime_accounting["checkpoint_serialization_seconds"] += (
                time.perf_counter() - save_started
            )

        core_started = time.perf_counter()
        with metrics_path.open("x", encoding="utf-8") as metrics_file:
            for epoch in range(epochs):
                epoch_start = time.time()
                model.train()
                negative_started = time.perf_counter()
                forget_negative = sample_negative_role_online(
                    sampler, epoch, role="forget_negative"
                )
                runtime_accounting["forget_negative_seconds"] += (
                    time.perf_counter() - negative_started
                )
                retain_started = time.perf_counter()
                if retain_source == "online":
                    retain_negative = sample_negative_role_online(
                        sampler, epoch, role="retain_negative"
                    )
                else:
                    if schedule is None:
                        raise AssertionError("precomputed schedule was not loaded")
                    retain_negative = schedule.get(epoch)
                runtime_accounting["retain_negative_seconds"] += (
                    time.perf_counter() - retain_started
                )
                optimizer.zero_grad(set_to_none=True)
                student_z = self.student_forward(model, runtime)
                total_loss, values = shared_lrr_objective(
                    model=model,
                    student_z=student_z,
                    directed_df_edge_index=runtime.directed_df_edge_index,
                    forget_negative_edge_index=forget_negative.to(
                        runtime.x.device
                    ),
                    local_edges=local_edges,
                    teacher_local_probability=teacher_targets.local_probability,
                    retain_positive_edges=retain_positive,
                    retain_negative_edges=retain_negative.to(
                        runtime.x.device
                    ),
                    anchor_edges=anchor_edges,
                    teacher_anchor_probability=teacher_targets.anchor_probability,
                    objective_weights=objective_weights,
                    margin=margin,
                )
                total_loss.backward()
                gradient_norms = {
                    name: float(
                        torch.linalg.vector_norm(parameter.grad).detach()
                    )
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                }
                if set(gradient_norms) != set(expected_names):
                    raise RuntimeError("all operator gradients are required")
                if not all(math.isfinite(value) for value in gradient_norms.values()):
                    raise RuntimeError("operator gradient is non-finite")
                self.validate_epoch_gradients(epoch, gradient_norms)
                if any(
                    parameter.grad is not None
                    for name, parameter in model.named_parameters()
                    if name.startswith(("node_emb.", "conv1.", "conv2."))
                ):
                    raise RuntimeError("backbone gradient must remain absent")
                optimizer.step()

                record: Dict[str, object] = {
                    "epoch": epoch,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    **values,
                    "gradient_norms": gradient_norms,
                    "elapsed_time": time.time() - epoch_start,
                }
                for name, value in gradient_norms.items():
                    record[f"gradient_norm_{name.replace('.', '_')}"] = value
                if diagnostic_candidates and values["total_loss"] < minimum_total_loss:
                    minimum_total_loss = values["total_loss"]
                    minimum_total_epoch = epoch
                    minimum_total_checkpoint = copy.deepcopy(
                        self._checkpoint(
                            model=model,
                            optimizer=optimizer,
                            epoch=epoch,
                            role="minimum_total_loss",
                            resolved_profile=resolved_profile,
                            sampler=sampler,
                            static_bundle=static_bundle,
                            evaluation_bundle=evaluation_bundle,
                            runtime=runtime,
                            teacher_hash=teacher_hash_before,
                            backbone_hash=backbone_hash_before,
                            validation_metrics=None,
                        )
                    )
                if (epoch + 1) % validation_frequency == 0:
                    validation_started = time.perf_counter()
                    validation = self.evaluate(
                        model,
                        runtime,
                        evaluation_bundle,
                        sampler.dr_mp_edge_index,
                        split="validation",
                    )
                    runtime_accounting["validation_seconds"] += (
                        time.perf_counter() - validation_started
                    )
                    score = legacy_validation_selection_score(validation)
                    retrain_distance = None
                    if retrain_validation_target is not None:
                        retrain_distance = self.validation_retrain_distance(
                            validation, retrain_validation_target
                        )
                    record.update(
                        {
                            **validation,
                            "validation_metric": score,
                            "selection_score": score,
                            "retrain_validation_distance": retrain_distance,
                        }
                    )
                    validation_record = {
                        "epoch": epoch,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "total_loss": values["total_loss"],
                        "forget_loss": values["forget_loss"],
                        "local_loss": values["local_loss"],
                        "retain_loss": values["retain_loss"],
                        "anchor_loss": values["anchor_loss"],
                        "forget_margin_violation_rate": values[
                            "forget_margin_violation_rate"
                        ],
                        "gradient_norms": gradient_norms,
                        "elapsed_time": time.time() - epoch_start,
                        **validation,
                        "selection_score": score,
                        "retrain_validation_distance": retrain_distance,
                    }
                    validation_history.append(validation_record)
                    if strict_maximum_replaces(score, best_score):
                        best_score = score
                        best_epoch = epoch
                        best_metrics = validation_record
                        if not smoke_only:
                            checkpoint = self._checkpoint(
                                model=model,
                                optimizer=optimizer,
                                epoch=epoch,
                                role="best",
                                resolved_profile=resolved_profile,
                                sampler=sampler,
                                static_bundle=static_bundle,
                                evaluation_bundle=evaluation_bundle,
                                runtime=runtime,
                                teacher_hash=teacher_hash_before,
                                backbone_hash=backbone_hash_before,
                                validation_metrics=validation_record,
                            )
                            timed_save(
                                checkpoint, checkpoint_dir / "model_best.pt"
                            )
                            if diagnostic_candidates:
                                diagnostic_checkpoint = self._checkpoint(
                                    model=model,
                                    optimizer=optimizer,
                                    epoch=epoch,
                                    role="best_by_legacy_sum",
                                    resolved_profile=resolved_profile,
                                    sampler=sampler,
                                    static_bundle=static_bundle,
                                    evaluation_bundle=evaluation_bundle,
                                    runtime=runtime,
                                    teacher_hash=teacher_hash_before,
                                    backbone_hash=backbone_hash_before,
                                    validation_metrics=validation_record,
                                )
                                timed_save(
                                    diagnostic_checkpoint,
                                    checkpoint_dir
                                    / "best_by_legacy_sum.pt",
                                )
                    if diagnostic_candidates and strict_maximum_replaces(
                        validation["dt_auc"], best_dt_auc
                    ):
                        best_dt_auc = validation["dt_auc"]
                        best_dt_epoch = epoch
                        timed_save(
                            self._checkpoint(
                                model=model,
                                optimizer=optimizer,
                                epoch=epoch,
                                role="best_by_dt_auc",
                                resolved_profile=resolved_profile,
                                sampler=sampler,
                                static_bundle=static_bundle,
                                evaluation_bundle=evaluation_bundle,
                                runtime=runtime,
                                teacher_hash=teacher_hash_before,
                                backbone_hash=backbone_hash_before,
                                validation_metrics=validation_record,
                            ),
                            checkpoint_dir / "best_by_dt_auc.pt",
                        )
                    if (
                        diagnostic_candidates
                        and retrain_distance is not None
                        and strict_minimum_replaces(
                            retrain_distance, closest_retrain_distance
                        )
                    ):
                        closest_retrain_distance = retrain_distance
                        closest_retrain_epoch = epoch
                        timed_save(
                            self._checkpoint(
                                model=model,
                                optimizer=optimizer,
                                epoch=epoch,
                                role="closest_to_retrain_validation",
                                resolved_profile=resolved_profile,
                                sampler=sampler,
                                static_bundle=static_bundle,
                                evaluation_bundle=evaluation_bundle,
                                runtime=runtime,
                                teacher_hash=teacher_hash_before,
                                backbone_hash=backbone_hash_before,
                                validation_metrics=validation_record,
                            ),
                            checkpoint_dir
                            / "closest_to_retrain_validation.pt",
                        )
                    print(
                        f"epoch={epoch} total={values['total_loss']:.6f} "
                        f"val_dt_auc={validation['dt_auc']:.6f} "
                        f"val_df_auc={validation['df_auc']:.6f} "
                        f"selection={score:.6f}",
                        flush=True,
                    )
                metrics_file.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
                metrics_file.flush()

        runtime_accounting["core_training_seconds"] = (
            time.perf_counter() - core_started
        )
        teacher_hash_after = _state_hash(teacher.state_dict())
        backbone_hash_after = _state_hash(
            model.state_dict(), prefixes=("node_emb.", "conv1.", "conv2.")
        )
        if teacher_hash_after != teacher_hash_before:
            raise RuntimeError("teacher state changed during training")
        if backbone_hash_after != backbone_hash_before:
            raise RuntimeError("student backbone changed during training")
        if best_epoch is None or best_metrics is None:
            raise RuntimeError("no valid validation checkpoint was produced")

        final_epoch = epochs - 1
        if smoke_only:
            smoke_checkpoint = self._checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=final_epoch,
                role="smoke",
                resolved_profile=resolved_profile,
                sampler=sampler,
                static_bundle=static_bundle,
                evaluation_bundle=evaluation_bundle,
                runtime=runtime,
                teacher_hash=teacher_hash_before,
                backbone_hash=backbone_hash_before,
                validation_metrics=validation_history[-1],
            )
            timed_save(
                smoke_checkpoint, checkpoint_dir / "smoke_checkpoint.pt"
            )
        else:
            final_checkpoint = self._checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=final_epoch,
                role="final",
                resolved_profile=resolved_profile,
                sampler=sampler,
                static_bundle=static_bundle,
                evaluation_bundle=evaluation_bundle,
                runtime=runtime,
                teacher_hash=teacher_hash_before,
                backbone_hash=backbone_hash_before,
                validation_metrics=validation_history[-1],
            )
            timed_save(final_checkpoint, checkpoint_dir / "model_final.pt")
            if diagnostic_candidates:
                if minimum_total_checkpoint is None:
                    raise RuntimeError("minimum-total checkpoint was not captured")
                timed_save(
                    minimum_total_checkpoint,
                    checkpoint_dir / "minimum_total_loss.pt",
                )

        if runtime.x.device.type == "cuda":
            runtime_accounting["peak_gpu_memory_mb"] = (
                float(torch.cuda.max_memory_allocated(runtime.x.device))
                / (1024.0**2)
            )
        try:
            import psutil

            memory_info = psutil.Process().memory_info()
            runtime_accounting["peak_cpu_rss_mb"] = (
                float(getattr(memory_info, "peak_wset", memory_info.rss))
                / (1024.0**2)
            )
        except ImportError:
            runtime_accounting["peak_cpu_rss_mb"] = 0.0
        runtime_accounting["end_to_end_seconds"] = (
            time.perf_counter() - end_to_end_started
        )

        return {
            "status": "PASS",
            "smoke_only": smoke_only,
            "epochs_completed": epochs,
            "best_epoch": best_epoch,
            "best_selection_metric": best_score,
            "best_validation_metrics": best_metrics,
            "final_validation_metrics": validation_history[-1],
            "validation_history": validation_history,
            "training_time_seconds": time.time() - start_time,
            "runtime_accounting": runtime_accounting,
            "retain_negative_source": retain_source,
            "forget_negative_source": "online",
            "teacher_target_source": teacher_source,
            "teacher_state_sha256_before": teacher_hash_before,
            "teacher_state_sha256_after": teacher_hash_after,
            "backbone_state_sha256_before": backbone_hash_before,
            "backbone_state_sha256_after": backbone_hash_after,
            "teacher_target_bundle": {
                "teacher_embedding_hash": (
                    teacher_targets.teacher_embedding_hash
                ),
                "local_probability_hash": (
                    teacher_targets.local_probability_hash
                ),
                "anchor_probability_hash": (
                    teacher_targets.anchor_probability_hash
                ),
            },
            "optimizer_parameter_identity_pass": True,
            "diagnostic_checkpoint_candidates": {
                "enabled": diagnostic_candidates,
                "best_by_legacy_sum_epoch": best_epoch,
                "best_by_dt_auc_epoch": best_dt_epoch,
                "closest_to_retrain_validation_epoch": (
                    closest_retrain_epoch
                ),
                "closest_to_retrain_validation_distance": (
                    None
                    if closest_retrain_epoch is None
                    else closest_retrain_distance
                ),
                "minimum_total_loss_epoch": minimum_total_epoch,
                "minimum_total_loss": (
                    None
                    if minimum_total_epoch is None
                    else minimum_total_loss
                ),
                "distance_version": VALIDATION_RETRAIN_DISTANCE_VERSION,
                "test_metrics_used": False,
            },
        }
