"""CHORD: neutralize each view, collect reversals, and update the classifier."""
import torch
from scipy.stats import beta
from torch import nn
from torch.nn import functional as F, PairwiseDistance

from ssc_common import proto, nearest_prototype, Transformation


def neutralize_prototypes(labels, features, outliers, n_classes):
    """Give the two within-class partitions equal weight in target-feature space."""
    inlier_centers, _ = proto(labels, features, ~outliers, n_classes)
    outlier_centers, outlier_counts = proto(labels, features, outliers, n_classes)
    balanced_centers = (0.5 * (inlier_centers + outlier_centers))[:-1]
    enough = (outlier_counts[:-1] > 1).float()
    return enough[:, None] * balanced_centers + (1 - enough)[:, None] * inlier_centers[:-1]


def fit_feature_transform(features, labels, centers, steps, device):
    """Fit diagonal feature suppression toward the supplied class centers."""
    feat = features.detach().clone().to(device)
    y = labels.flatten().long().detach().clone().to(device)
    centers = centers.to(device)
    beta = nn.Parameter(torch.ones((1, feat.size(-1)), device=device))
    bias = nn.Parameter(torch.zeros((1, feat.size(-1)), device=device))
    optimizer = torch.optim.AdamW([beta, bias], 1e-3, weight_decay=0)
    transformation = Transformation(beta, bias)
    dist = PairwiseDistance(keepdim=True)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = dist(transformation(feat), centers[y]).mean() + 10 * beta.mean()
        loss.backward()
        optimizer.step()
    return Transformation(beta.detach(), bias.detach())


def validate_view_partition(features, labels, outliers, original_errors,
                            n_classes, steps, device):
    """Return the proposed outliers that reverse to their class in target space."""
    centers = neutralize_prototypes(labels, features, outliers, n_classes)
    transformation = fit_feature_transform(features, labels, centers, steps, device)
    transformed = transformation(features.to(device)).detach()
    recovered = nearest_prototype(centers.to(device), transformed).cpu() == labels
    return original_errors & outliers & recovered


def evidence_support(positive_count, negative_count):
    """Conservative support score with a uniform Beta prior."""
    return float(beta.ppf(0.05, positive_count + 1, negative_count + 1))


def select_geometry(geo_outliers, geo_reversals, anchor_reversals,
                    pretrained_reversals, original_errors):
    """Admit geo only with supported reversals and dominant new coverage."""
    existing = anchor_reversals.clone()
    for mask in pretrained_reversals:
        existing |= mask
    geo_new = int((geo_reversals & ~existing).sum())
    pretrained_new = int((existing & ~anchor_reversals).sum())
    recovered = int(geo_reversals.sum())
    target_correct = int((geo_outliers & ~original_errors).sum())
    # Compute a lower posterior bound with a uniform Beta(1, 1) prior.
    support = evidence_support(recovered, target_correct)
    enabled = support > 0.5 and geo_new > pretrained_new
    return enabled, {'geo_new':geo_new, 'pretrained_new':pretrained_new,
                     'recovered':recovered, 'target_correct_outliers':target_correct,
                     'support_lower':support}


def train_balanced_classifier(features, labels, reversal_weights, stable_mask,
                              n_classes, steps, seed, device):
    """Balance weighted reversals against examples correct under target prototypes."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    torch.manual_seed(seed)
    corrected_mask = (reversal_weights > 0).to(device)
    corrected_features = features[corrected_mask].to(device)
    corrected_labels = labels[corrected_mask].flatten().long().to(device)
    sampling_weights = reversal_weights.to(device)[corrected_mask].double().clamp_min(0).cpu()
    sampling_weights = sampling_weights / sampling_weights.sum()
    stable_features = features[stable_mask].to(device)
    stable_labels = labels[stable_mask].flatten().long().to(device)
    fc = nn.Linear(features.size(-1), n_classes).to(device)
    optimizer = torch.optim.AdamW(fc.parameters(), 1e-3, weight_decay=0)
    batch = min(len(corrected_labels), len(stable_labels))
    for _ in range(steps):
        optimizer.zero_grad()
        corrected_indices = torch.multinomial(sampling_weights, batch, replacement=True,
                                              generator=generator)
        stable_indices = (torch.randint(0, len(stable_labels), (batch,), generator=generator)
                          if batch < len(stable_labels) else torch.arange(0, len(stable_labels)))
        loss = (F.cross_entropy(fc(corrected_features[corrected_indices]), corrected_labels[corrected_indices])
                + F.cross_entropy(fc(stable_features[stable_indices]), stable_labels[stable_indices]))
        loss.backward()
        optimizer.step()
    return fc


def aggregate_reversal_weights(view_reversals, labels, n_classes):
    """Normalize each view over its supported classes, then average the weights."""
    weights = []
    for mask in view_reversals.values():
        if not mask.any():
            continue
        weight = torch.zeros(len(labels), dtype=torch.float64)
        supported = [c for c in range(n_classes) if (mask & (labels == c)).any()]
        for c in supported:
            selected = mask & (labels == c)
            weight[selected] = 1.0 / (len(supported) * int(selected.sum()))
        weights.append(weight)
    return torch.stack(weights).mean(0)
