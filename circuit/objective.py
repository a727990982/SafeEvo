"""L_ext = alpha L_ref(M) + beta L_cmp(I-M) + sum lambda mean(sigmoid(q))."""
from dataclasses import dataclass

import torch

from matrix_masks import partition


@dataclass
class ExtractionLoss:
    total: torch.Tensor
    refusal: torch.Tensor
    compliance: torch.Tensor
    mlp_penalty: torch.Tensor
    refusal_output: object

    def log_values(self):
        terms = (self.total, self.refusal, self.compliance,
                 self.mlp_penalty)
        names = ("total_loss", "loss_accept", "loss_refuse",
                 "sparsity_loss_mlp")
        return {name: term.detach().float().mean().item() for name, term in zip(names, terms)}


def extraction_loss(model, inputs, mlp_logits,
                    alpha=1.0, beta=1.0, lambda_mlp=1.0):
    device = next(model.parameters()).device
    batches = {branch: {key: value.to(device) for key, value in inputs[branch].items()}
               for branch in ("accept", "refuse")}
    with partition(model):
        refusal = model(**batches["accept"])
    with partition(model, complement=True):
        compliance = model(**batches["refuse"])
    mlp_penalty = sum(torch.sigmoid(q).mean() for q in mlp_logits)
    total = (alpha * refusal.loss + beta * compliance.loss
             + lambda_mlp * mlp_penalty)
    return ExtractionLoss(total, refusal.loss, compliance.loss,
                          mlp_penalty, refusal)
