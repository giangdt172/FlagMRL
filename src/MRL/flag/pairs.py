"""Near positives and globally normalized valid cross-side negatives."""

import torch


@torch.no_grad()
def build_pair_weights(Q, P, target, pos_id_p=None, far_mode="uniform", far_tau=0.1):
    """Inputs are globally gathered normalized FP32 embeddings and column IDs."""
    with torch.autocast(device_type=Q.device.type, enabled=False):
        Q, P = Q.detach().float(), P.detach().float()
        if Q.ndim != 2 or P.ndim != 2 or Q.size(1) != P.size(1) or Q.size(0) == 0:
            raise ValueError("Q and P must be nonempty embedding matrices with matching dimensions.")
        B, columns = Q.size(0), P.size(0)
        if target.shape != (B,) or target.dtype != torch.long or target.min() < 0 or target.max() >= columns:
            raise ValueError("target must contain one valid int64 positive index per query.")
        if far_mode not in {"uniform", "hard"}:
            raise ValueError(f"Unknown flag_far_mode: {far_mode}")
        if far_mode == "hard" and far_tau <= 0:
            raise ValueError("flag_far_tau must be positive.")
        rows = torch.arange(B, device=Q.device)
        near = torch.zeros(B, columns, device=Q.device, dtype=torch.float32)
        near[rows, target] = 1 / B
        valid = torch.ones(B, columns, device=Q.device, dtype=torch.bool)
        valid[rows, target] = False
        if pos_id_p is not None:
            if pos_id_p.shape != (columns,):
                raise ValueError("pos_id_p must identify every positive column.")
            valid &= pos_id_p.unsqueeze(0) != pos_id_p[target].unsqueeze(1)
        if not valid.any():
            return near, torch.zeros_like(near), False
        if far_mode == "uniform":
            far = valid.float()
        else:
            scores = (Q @ P.T) / far_tau
            scores = scores.masked_fill(~valid, -torch.inf)
            far = torch.exp(scores - scores.max())
        far /= far.sum()
        return near, far, True
