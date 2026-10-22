"""MRL loss from shared flag-coordinate statistics."""

import torch
from torch.nn import functional as F


def _logits(dot, q_squared_norm, p_squared_norm, temperature):
    # Equivalent to F.normalize(..., eps=1e-12), including zero embeddings.
    # Clamp before sqrt so its backward is finite at zero.
    q_norm = q_squared_norm.clamp_min(1e-24).sqrt()
    p_norm = p_squared_norm.clamp_min(1e-24).sqrt()
    return dot / q_norm[:, None] / p_norm[None, :] / temperature


def _cross_dim_pair(log_small, log_large, target):
    """Detached per-query teacher selection, then KL(teacher || student)."""
    with torch.no_grad():
        positive_small = log_small.gather(1, target[:, None]).squeeze(1)
        positive_large = log_large.gather(1, target[:, None]).squeeze(1)
        # Log probabilities keep selection stable even if exp underflows.
        small_teaches = positive_small >= positive_large
        weights = (positive_small.exp() - positive_large.exp()).abs()
        teacher_log = torch.where(
            small_teaches[:, None], log_small.detach(), log_large.detach())
        teacher_prob = teacher_log.exp()

    student_log = torch.where(small_teaches[:, None], log_large, log_small)
    kl_rows = (teacher_prob * (teacher_log - student_log)).sum(dim=-1)
    loss = (weights * kl_rows).mean()
    with torch.no_grad():
        active = weights > 0
        metrics = {
            "loss": loss.detach(),
            "mean_weight": weights.mean(),
            "mean_kl": kl_rows.mean(),
            "small_teaches_fraction": (small_teaches & active).float().mean(),
            "large_teaches_fraction": (~small_teaches & active).float().mean(),
            "tie_or_underflow_fraction": (~active).float().mean(),
        }
    return loss, metrics


def encoder_losses(model, q, p, target, dims, weights, temperature, *, lambda_cross=0.0):
    """Inputs are unnormalized coordinates X @ U, including the full level.

    Compute each coordinate block's dot products and squared norms once. Their
    running sums give every prefix cosine. When KD is enabled, CE and adjacent
    dimension KL share one log_softmax per level. Candidates/targets follow MRL
    unchanged; these logits are finite and unmasked.
    """
    mrl_loss = q.new_zeros(())
    cross_loss = q.new_zeros(())
    cross_enabled = lambda_cross > 0 and len(dims) > 1 and p.size(0) > 1
    metrics = {}
    previous = 0
    previous_log = None
    prefix_dot = prefix_q_squared = prefix_p_squared = None
    for dim, weight in zip(dims, weights):
        q_block, p_block = q[:, previous:dim], p[:, previous:dim]
        block_dot = model.compute_similarity(q_block, p_block)
        block_q_squared = q_block.square().sum(dim=-1)
        block_p_squared = p_block.square().sum(dim=-1)
        if previous:
            dot = prefix_dot + block_dot
            q_squared = prefix_q_squared + block_q_squared
            p_squared = prefix_p_squared + block_p_squared
        else:
            dot, q_squared, p_squared = block_dot, block_q_squared, block_p_squared

        logits = _logits(dot, q_squared, p_squared, temperature)
        if cross_enabled:
            log_probs = F.log_softmax(logits, dim=-1)
            level_loss = F.nll_loss(log_probs, target)
            if previous_log is not None:
                pair_loss, pair_metrics = _cross_dim_pair(previous_log, log_probs, target)
                cross_loss = cross_loss + pair_loss
                metrics.update({f"cross_{previous}_{dim}/{key}": value
                                for key, value in pair_metrics.items()})
            previous_log = log_probs
        else:
            # Preserve the original fused CE path when KD is disabled.
            level_loss = F.cross_entropy(logits, target)
        mrl_loss = mrl_loss + weight * level_loss
        metrics[f"loss_dim_{dim}"] = level_loss.detach()
        metrics[f"contrastive_loss_dim_{dim}"] = level_loss.detach()

        prefix_dot, prefix_q_squared, prefix_p_squared = dot, q_squared, p_squared
        previous = dim
    if cross_enabled:
        cross_loss = cross_loss / (len(dims) - 1)
        weighted_cross_loss = lambda_cross * cross_loss
        total_loss = mrl_loss + weighted_cross_loss
    else:
        weighted_cross_loss = cross_loss
        total_loss = mrl_loss
    metrics.update(mrl_loss=mrl_loss.detach(), cross_loss=cross_loss.detach(),
                   weighted_cross_loss=weighted_cross_loss.detach())
    return total_loss, metrics
