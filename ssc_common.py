"""Prototype and feature utilities, adapted from the public NSF code."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def proto(labels_s: torch.Tensor, f_s: torch.Tensor, mask, n_classes: int):
    """Compute class prototypes, collecting excluded samples in the final row."""
    labels_s = labels_s.clone()
    if mask is not None:
        labels_s[mask <= 0] = n_classes
    label_one_hot = F.one_hot(labels_s, n_classes + 1).permute(1, 0)
    npps_sum = label_one_hot.float() @ f_s
    label_masks = label_one_hot.sum(dim=-1) + 1e-5
    labelw = 1 / (label_masks.unsqueeze(dim=-1).expand(npps_sum.size()))
    npps = labelw * npps_sum
    label_masks[-1] = 0
    label_masks[label_masks < 1] = 0
    return npps, label_masks


def nearest_prototype(centers: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    return (features.unsqueeze(1) - centers[None]).norm(dim=-1, p=2).argmin(dim=-1)


def identify(labels: torch.Tensor, features: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Find samples whose nearest class prototype disagrees with their label."""
    global_prototypes = proto(labels, features, None, n_classes)[0][:-1]
    return nearest_prototype(global_prototypes, features) != labels


class Transformation(nn.Module):
    def __init__(self, w, b):
        super().__init__()
        self.w = w
        self.b = b

    def forward(self, x):
        return (x + self.b) * self.w - self.b


def group_metrics(predictions, labels, spurious):
    """Compute overall accuracy and worst-group accuracy from predictions."""
    predictions = np.asarray(predictions).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    spurious = np.asarray(spurious).reshape(-1)
    per_group = {}
    for c in np.unique(labels):
        for s in np.unique(spurious):
            m = (labels == c) & (spurious == s)
            if m.sum():
                per_group[f"y{c}_p{s}"] = float((predictions[m] == labels[m]).mean())
    return {
        "mean_accuracy": float((predictions == labels).mean()),
        "wga": float(min(per_group.values())),
        "groups": per_group,
    }


def within_class_whiten(labels, features, n_classes: int, shrink: float):
    """Identify in a within-class whitened metric; use class labels only."""
    x = features.double()
    centred = x.clone()
    for c in range(n_classes):
        selected = labels == c
        if selected.any():
            centred[selected] -= x[selected].mean(0, keepdim=True)
    within = (centred.T @ centred) / max(len(x) - n_classes, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(within)
    scale = (eigenvalues.clamp_min(0) + shrink * eigenvalues.mean()).rsqrt()
    return (((x - x.mean(0, keepdim=True)) @ eigenvectors) * scale).float()
