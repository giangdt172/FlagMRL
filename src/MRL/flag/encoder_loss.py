"""MRL loss from shared flag-coordinate statistics."""

from torch.nn import functional as F


def _logits(dot, q_squared_norm, p_squared_norm, temperature):
    # Equivalent to F.normalize(..., eps=1e-12), including zero embeddings.
    # Clamp before sqrt so its backward is finite at zero.
    q_norm = q_squared_norm.clamp_min(1e-24).sqrt()
    p_norm = p_squared_norm.clamp_min(1e-24).sqrt()
    return dot / q_norm[:, None] / p_norm[None, :] / temperature


def encoder_losses(model, q, p, target, dims, weights, temperature):
    """Inputs are unnormalized coordinates X @ U, including the full level.

    Compute each coordinate block's dot products and squared norms once. Their
    running sums give every prefix cosine.
    """
    mrl_loss = q.new_zeros(())
    metrics = {}
    previous = 0
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

        level_loss = F.cross_entropy(
            _logits(dot, q_squared, p_squared, temperature), target)
        mrl_loss = mrl_loss + weight * level_loss
        metrics[f"loss_dim_{dim}"] = level_loss.detach()
        metrics[f"contrastive_loss_dim_{dim}"] = level_loss.detach()

        prefix_dot, prefix_q_squared, prefix_p_squared = dot, q_squared, p_squared
        previous = dim
    return mrl_loss, metrics
