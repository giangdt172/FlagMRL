"""MRL and block-local gain loss from shared flag-coordinate statistics."""

import torch
from torch.nn import functional as F


def _logits(dot, q_squared_norm, p_squared_norm, temperature):
    # Equivalent to F.normalize(..., eps=1e-12), including zero embeddings.
    # Clamp before sqrt so its backward is finite at zero.
    q_norm = q_squared_norm.clamp_min(1e-24).sqrt()
    p_norm = p_squared_norm.clamp_min(1e-24).sqrt()
    return dot / q_norm[:, None] / p_norm[None, :] / temperature


def encoder_losses(model, q, p, target, dims, weights, temperature, compute_gain):
    """Inputs are unnormalized coordinates X @ U, including the full level.

    Compute each coordinate block's dot products and squared norms once. Their
    running sums give every prefix cosine. Gain reuses the same block tensors,
    detaching previous sums on both sides *before* cosine normalization.
    """
    mrl_loss, gain_loss = q.new_zeros(()), q.new_zeros(())
    metrics = {}
    previous = 0
    prefix_dot = prefix_q_squared = prefix_p_squared = best = None
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

        per_sample = F.cross_entropy(
            _logits(dot, q_squared, p_squared, temperature), target, reduction="none")
        level_loss = per_sample.mean()
        mrl_loss = mrl_loss + weight * level_loss
        metrics[f"loss_dim_{dim}"] = level_loss.detach()
        metrics[f"contrastive_loss_dim_{dim}"] = level_loss.detach()

        if compute_gain:
            if previous:
                gain_per_sample = F.cross_entropy(
                    _logits(prefix_dot.detach() + block_dot,
                            prefix_q_squared.detach() + block_q_squared,
                            prefix_p_squared.detach() + block_p_squared, temperature),
                    target, reduction="none")
                level_gain = F.softplus(gain_per_sample - best).mean()
                gain_loss = gain_loss + weight * level_gain
                metrics[f"gain_loss_dim_{dim}"] = level_gain.detach()
            detached_loss = per_sample.detach()
            best = detached_loss if best is None else torch.minimum(best, detached_loss)

        prefix_dot, prefix_q_squared, prefix_p_squared = dot, q_squared, p_squared
        previous = dim
    return mrl_loss, gain_loss, metrics
