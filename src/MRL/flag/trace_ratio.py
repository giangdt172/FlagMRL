"""Pairwise nested trace ratio on the existing aggregated flag projector."""

import math

import torch


@torch.no_grad()
def trace_ratio_loss_and_grad(U, a_col, near, far, eps):
    """Return -tr(P_U C_neg)/(tr(P_U C_pos)+eps) and its Euclidean gradient.

    Inputs follow the spherical objective's detached, normalized embeddings and
    pair weights. With one positive per query and uniform negatives these give
    C_pos = sum_i delta_ii delta_ii.T/B and
    C_neg = sum_{i != j} delta_ij delta_ij.T/(B*(B-1)).
    The existing a_col holds W's diagonal, including the full-dimensional level.
    Aggregate traces BEFORE division; do not normalize projected coordinates.
    """
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("The trace-ratio denominator regularizer eps must be positive and finite.")
    with torch.autocast(device_type=U.device.type, enabled=False):
        basis, weights = U.detach().float(), a_col.detach().float()

        # Positive weights have one nonzero per valid query. Pair differences
        # avoid cancellation when matched embeddings are nearly identical.
        query, target, pair_weights = (t.detach().float() for t in near)
        rows, columns = pair_weights.nonzero(as_tuple=True)
        delta = query[rows] - target[columns]
        projected = delta @ basis
        positive_weights = pair_weights[rows, columns, None]
        pos_action = delta.T @ (positive_weights * projected)  # C_pos @ U
        positive = (positive_weights * projected.square() * weights).sum()

        query, target, pair_weights = (t.detach().float() for t in far)
        # Center both pools by the SAME vector: pair differences are unchanged.
        # Weighted second moments avoid B-by-B-by-d differences and d-by-d
        # scatters. Centering also reduces cancellation in these moments.
        center = query.mean(dim=0, keepdim=True)
        query, target = query - center, target - center
        q_projected, p_projected = query @ basis, target @ basis
        q_action = pair_weights.sum(dim=1, keepdim=True) * q_projected - pair_weights @ p_projected
        p_action = pair_weights.sum(dim=0)[:, None] * p_projected - pair_weights.T @ q_projected
        neg_action = query.T @ q_action + target.T @ p_action  # C_neg @ U
        negative = ((q_projected * q_action).sum(dim=0)
                    + (p_projected * p_action).sum(dim=0)) @ weights

        denominator = positive + eps
        ratio = negative / denominator
        loss = -ratio
        # H = -2/D * (C_neg - (N/D)*C_pos) @ U @ W.
        gradient = (-2 / denominator) * (neg_action - ratio * pos_action) * weights
        if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
            raise FloatingPointError("Non-finite trace-ratio flag loss or gradient.")
        return loss, positive, negative, gradient
