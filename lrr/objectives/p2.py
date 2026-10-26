from __future__ import annotations

import torch
import torch.nn.functional as F

P2_WEIGHTS = {"forget": 0.05, "local": 0.50, "retain": 1.00, "anchor": 0.20}
P2_MARGIN = 0.05


def p2_objective(*, forget_positive_logits: torch.Tensor,
                 forget_reference_logits: torch.Tensor,
                 local_logits: torch.Tensor, teacher_local_probability: torch.Tensor,
                 retain_positive_logits: torch.Tensor, retain_negative_logits: torch.Tensor,
                 anchor_logits: torch.Tensor, teacher_anchor_probability: torch.Tensor,
                 weights: dict[str, float] | None = None, margin: float = P2_MARGIN):
    """Shared P2 algebra; callers bind the method-specific Forget reference explicitly."""
    if forget_positive_logits.numel() == 0:
        forget_loss = forget_positive_logits.sum() * 0.0
    else:
        forget_loss = F.softplus(
            forget_positive_logits - forget_reference_logits.detach() + float(margin)
        ).mean()
    local_loss = F.mse_loss(torch.sigmoid(local_logits), teacher_local_probability)
    retain_logits = torch.cat((retain_positive_logits, retain_negative_logits))
    retain_labels = torch.cat((torch.ones_like(retain_positive_logits),
                               torch.zeros_like(retain_negative_logits)))
    retain_loss = F.binary_cross_entropy_with_logits(retain_logits, retain_labels)
    anchor_loss = F.mse_loss(torch.sigmoid(anchor_logits), teacher_anchor_probability)
    w = P2_WEIGHTS if weights is None else weights
    total = (w["forget"] * forget_loss + w["local"] * local_loss +
             w["retain"] * retain_loss + w["anchor"] * anchor_loss)
    return total, {"LF": forget_loss, "LL": local_loss, "LR": retain_loss,
                   "LA": anchor_loss, "total": total}
