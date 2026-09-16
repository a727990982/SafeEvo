"""Matrix operators for circuit extraction.

For a projection W, the two branches are M(Wx+b) and (I-M)(Wx+b),
where M = Diag(STE(sigmoid(q))). A diagonal matrix is stored as its
diagonal: applying M scales output coordinates without allocating a dense matrix.
The saved mask dictionary continues to map projection paths to raw logits q.
"""
from contextlib import contextmanager

import torch
from torch import nn


class DiagonalMask(nn.Module):
    """Learnable M, with an optional fixed support for restricted extraction."""

    def __init__(self, size, reference, initial_logit):
        super().__init__()
        self.q = nn.Parameter(torch.ones(size, device=reference.device,
                                        dtype=reference.dtype) * initial_logit)
        self.register_buffer("support", None)
        self.complement = False

    def diagonal(self, temperature=1.0, hard=True):
        soft = torch.sigmoid(self.q / temperature)
        diagonal = (soft > 0.5).to(soft.dtype) - soft.detach() + soft if hard else soft
        if self.support is not None:
            diagonal = diagonal * self.support.to(diagonal.dtype)
        return 1.0 - diagonal if self.complement else diagonal

    def forward(self, coordinates, temperature=1.0):
        diagonal = self.diagonal(temperature)
        return coordinates * diagonal


class RowProjection(nn.Module):
    """z_C = M(Wx+b); z_complement = (I-M)(Wx+b).

    Keeping the original Linear call preserves its kernel and bias handling.
    For the bias-free projections in the paper this is exactly M W x.
    """

    def __init__(self, projection, initial_logit=1.0):
        super().__init__()
        self.W = projection.requires_grad_(False)
        self.M = DiagonalMask(projection.out_features, projection.weight, initial_logit)

    def forward(self, x):
        return self.M(self.W(x))


def mask_sites(model):
    for path, module in model.named_modules():
        if isinstance(module, RowProjection):
            yield path, module


def install_masks(model, init_value=1.0):
    """Freeze base weights and install output-row masks on MLP projections."""
    model.requires_grad_(False)
    mlp = []
    for path, layer in list(model.named_modules()):
        if path.endswith(("gate_proj", "up_proj", "down_proj")):
            parent, _, name = path.rpartition(".")
            replacement = RowProjection(layer, init_value)
            setattr(model.get_submodule(parent), name, replacement)
            mlp.append(replacement.M.q)
    return mlp


def set_partition(model, complement=False):
    for _, module in mask_sites(model):
        module.M.complement = complement


@contextmanager
def partition(model, complement=False):
    """Select M or I-M for this model and restore its previous state afterward."""
    gates = [module.M for _, module in mask_sites(model)]
    previous = [gate.complement for gate in gates]
    try:
        for gate in gates:
            gate.complement = complement
        yield
    finally:
        for gate, state in zip(gates, previous):
            gate.complement = state


def mask_state(model):
    return {path: module.M.q.detach().cpu() for path, module in mask_sites(model)}


def save_masks(model, path):
    torch.save(mask_state(model), path)


def assign_mask_logits(model, state):
    counts = {"mlp": 0}
    with torch.no_grad():
        for path, module in mask_sites(model):
            if path in state:
                module.M.q.copy_(state[path].to(module.M.q.device))
                counts["mlp"] += 1
    return counts


def load_masks(model, path, map_location="cpu"):
    return assign_mask_logits(model, torch.load(path, map_location=map_location, weights_only=True))
