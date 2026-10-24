"""Non-parameter flag state, always stored in FP32."""

from pathlib import Path
import math

import torch
from torch import nn
from torch.nn import functional as F

from src.utils import resolve_nested_dims


class FlagProjector(nn.Module):
    def __init__(self, full_dim, q_dims=None, init="random", eps=1e-4, lr=0.1, retraction="svd", seed=0,
                 lambda_neg=1.0):
        super().__init__()
        self.full_dim = int(full_dim)
        self.q_dims = resolve_nested_dims(q_dims, self.full_dim)
        if any(isinstance(d, bool) for d in self.q_dims):
            raise ValueError("Flag levels must be integer dimensions, not booleans.")
        if self.full_dim not in self.q_dims:
            self.q_dims.append(self.full_dim)
        # Flag subspaces and Matryoshka supervision use the same levels.
        self.mrl_dims = list(self.q_dims)
        if not math.isfinite(eps) or eps <= 0 or not math.isfinite(lr) or lr < 0:
            raise ValueError("flag_eps must be positive and flag_lr nonnegative, both finite.")
        if not math.isfinite(lambda_neg) or lambda_neg < 0:
            raise ValueError("flag_lambda_neg must be nonnegative and finite.")
        if init not in {"random", "identity", "spherical", "geneig"}:
            raise ValueError(f"Unknown flag_init: {init}")
        if retraction not in {"svd", "newton_schulz"}:
            raise ValueError(f"Unknown flag_retraction: {retraction}")
        self.eps, self.lr, self.retraction = eps, lr, retraction
        self.lambda_neg = lambda_neg
        with torch.no_grad(), torch.autocast(device_type="cpu", enabled=False):
            if init == "identity":
                initial = torch.eye(full_dim, dtype=torch.float32)[:, :self.q_dims[-1]]
            else:
                generator = torch.Generator().manual_seed(seed)
                initial = torch.linalg.qr(torch.randn(full_dim, self.q_dims[-1], generator=generator), mode="reduced").Q
            weights = []
            previous = 0
            for k, dim in enumerate(self.q_dims):
                weights.extend([(len(self.q_dims) - k) / len(self.q_dims)] * (dim - previous))
                previous = dim
        self.register_buffer("U", initial.contiguous())
        self.register_buffer("a_col", torch.tensor(weights, dtype=torch.float32))

    def _apply(self, fn, recurse=True):
        # Preserve FP32 even when a parent module is cast to bf16/half.
        original_u, original_weights = self.U, self.a_col
        super()._apply(fn, recurse=recurse)
        self.U = original_u.to(device=self.U.device, dtype=torch.float32)
        self.a_col = original_weights.to(device=self.a_col.device, dtype=torch.float32)
        return self

    def project(self, x, k):
        return self.project_at_dim(x, self.q_dims[k])

    def project_at_dim(self, x, dim):
        if not 0 < dim <= self.full_dim or (dim > self.U.size(1) and dim != self.full_dim):
            raise ValueError(f"Invalid flag projection dimension: {dim}")
        with torch.autocast(device_type=x.device.type, enabled=False):
            if dim == self.full_dim:
                return F.normalize(x.float(), dim=-1)
            return F.normalize(x.float() @ self.U[:, :dim].detach().clone(), dim=-1)

    @torch.no_grad()
    def step(self, near, far, *, objective="spherical"):
        from .flag_math import flag_loss_and_grad, stiefel_step, principal_angle, orthogonality_error
        if objective not in {"spherical", "trace_ratio"}:
            raise ValueError(f"Unknown flag objective: {objective}")
        with torch.autocast(device_type=self.U.device.type, enabled=False):
            if not far[2].any():
                return {"flag_skipped": 1.0}
            if objective == "trace_ratio":
                from .trace_ratio import trace_ratio_loss_and_grad
                loss, positive, negative, gradient = trace_ratio_loss_and_grad(
                    self.U, self.a_col, near, far, self.eps)
                metrics = {"flag_pos_scatter": positive, "flag_neg_scatter": negative,
                           "flag_trace_ratio": -loss}
            else:
                loss, positive, negative, gradient = flag_loss_and_grad(
                    self.U, self.a_col, near, far, self.eps, self.lambda_neg)
                metrics = {"flag_pos_cosine": positive, "flag_neg_cosine": negative,
                           "flag_gap": positive - negative}
            updated, norm = stiefel_step(self.U, gradient, self.lr, self.retraction)
            # The full-dimensional span never changes. Track the largest
            # proper prefix, as the old rectangular basis diagnostic did.
            proper_dims = [d for d in self.q_dims if d < self.full_dim]
            drift = (principal_angle(self.U[:, :proper_dims[-1]], updated[:, :proper_dims[-1]])
                     if proper_dims else self.U.new_zeros(()))
            self.U.copy_(updated)
            return {"flag_loss": loss, **metrics, "flag_gnorm": norm,
                    "flag_orth_err": orthogonality_error(self.U), "flag_drift": drift,
                    "flag_skipped": 0.0}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"U": self.U.detach().cpu(), "q_dims": self.q_dims, "full_dim": self.full_dim,
                    "mrl_dims": self.mrl_dims}, path)

    @torch.no_grad()
    def load(self, path, *, strict_mrl_dims=True):
        """Load a validated basis; require matching MRL training levels when resuming."""
        state = torch.load(path, map_location=self.U.device, weights_only=True)
        if state["q_dims"] != self.q_dims or state["full_dim"] != self.full_dim:
            raise ValueError("Flag checkpoint q_dims/full_dim does not match the model.")
        saved_mrl_dims = state.get("mrl_dims", sorted(set(state["q_dims"] + [state["full_dim"]])))
        if strict_mrl_dims and saved_mrl_dims != self.mrl_dims:
            raise ValueError("Flag checkpoint MRL dimensions do not match the Flag subspace dimensions.")
        basis = state["U"].float()
        if basis.shape != self.U.shape or not torch.isfinite(basis).all():
            raise ValueError("Invalid flag checkpoint basis.")
        error = (basis.T @ basis - torch.eye(basis.size(1), device=basis.device)).abs().max()
        if error > 1e-3:
            raise ValueError(f"Flag checkpoint is not orthonormal: {error.item()}")
        self.U.copy_(basis)
