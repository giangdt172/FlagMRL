"""Non-parameter flag state, always stored in FP32."""

from pathlib import Path
import math

import torch
from torch import nn
from torch.nn import functional as F

from src.utils import resolve_nested_dims


class FlagProjector(nn.Module):
    def __init__(self, full_dim, q_dims=None, init="random", eps=1e-4, lr=0.1, retraction="svd", seed=0,
                 lambda_neg=1.0, block_size=None, mrl_dims=None):
        super().__init__()
        self.full_dim = int(full_dim)
        self.block_size = block_size
        if block_size is not None:
            if (not isinstance(block_size, int) or isinstance(block_size, bool)
                    or block_size <= 0 or self.full_dim % block_size):
                raise ValueError("block_size must be a positive integer dividing full_dim.")
            block_dims = list(range(block_size, self.full_dim + 1, block_size))
            if q_dims is not None and list(q_dims) != block_dims:
                raise ValueError("Flag q_dims must match block_size, 2*block_size, ..., full_dim.")
            q_dims = block_dims
        self.q_dims = sorted(q_dims) if q_dims is not None else []
        if (not self.q_dims or any(not isinstance(d, int) or isinstance(d, bool) for d in self.q_dims)
                or len(set(self.q_dims)) != len(self.q_dims)
                or not 0 < self.q_dims[0] <= self.q_dims[-1] <= self.full_dim):
            raise ValueError("Flag levels must be unique positive integer dimensions at most full_dim.")
        # Geometry prefixes and encoder supervision dimensions are independent.
        self.mrl_dims = resolve_nested_dims(self.q_dims if mrl_dims is None else mrl_dims, self.full_dim)
        if self.full_dim not in self.mrl_dims:
            self.mrl_dims.append(self.full_dim)
        if any(d > self.q_dims[-1] and d != self.full_dim for d in self.mrl_dims):
            raise ValueError("Compressed MRL dimensions cannot exceed the flag basis width.")
        if not math.isfinite(eps) or eps <= 0 or not math.isfinite(lr) or lr < 0:
            raise ValueError("flag_eps must be positive and flag_lr nonnegative, both finite.")
        if not math.isfinite(lambda_neg) or lambda_neg < 0:
            raise ValueError("flag_lambda_neg must be nonnegative and finite.")
        if init not in {"random", "identity", "geneig"}:
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
    def step(self, near, far):
        from .flag_math import flag_loss_and_grad, stiefel_step, principal_angle, orthogonality_error
        with torch.autocast(device_type=self.U.device.type, enabled=False):
            if not far[2].any():
                return {"flag_skipped": 1.0}
            loss, positive, negative, gradient = flag_loss_and_grad(
                self.U, self.a_col, near, far, self.eps, self.lambda_neg)
            updated, norm = stiefel_step(self.U, gradient, self.lr, self.retraction)
            # The full-dimensional span never changes. Track the largest
            # proper prefix, as the old rectangular basis diagnostic did.
            proper_dims = [d for d in self.q_dims if d < self.full_dim]
            drift = (principal_angle(self.U[:, :proper_dims[-1]], updated[:, :proper_dims[-1]])
                     if proper_dims else self.U.new_zeros(()))
            self.U.copy_(updated)
            return {"flag_loss": loss, "flag_pos_cosine": positive,
                    "flag_neg_cosine": negative, "flag_gap": positive - negative, "flag_gnorm": norm,
                    "flag_orth_err": orthogonality_error(self.U), "flag_drift": drift,
                    "flag_skipped": 0.0}

    @torch.no_grad()
    def initialize_geneig(self, near_scatter, far_scatter):
        """Only initialization may materialize full p-by-p scatter matrices."""
        with torch.autocast(device_type=self.U.device.type, enabled=False):
            near, far = near_scatter.float(), far_scatter.float()
            identity = torch.eye(self.full_dim, device=self.U.device)
            near, far = (near + near.T) * 0.5, (far + far.T) * 0.5
            lower = torch.linalg.cholesky(near + self.eps * identity)
            left = torch.linalg.solve_triangular(lower, far, upper=False)
            whitened = torch.linalg.solve_triangular(lower, left.T, upper=False).T
            eigenvalues, eigenvectors = torch.linalg.eigh((whitened + whitened.T) * 0.5)
            order = torch.argsort(eigenvalues, descending=True)[:self.q_dims[-1]]
            vectors = torch.linalg.solve_triangular(lower.T, eigenvectors[:, order], upper=True)
            self.U.copy_(torch.linalg.qr(vectors, mode="reduced").Q)
            return eigenvalues[order]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"U": self.U.detach().cpu(), "q_dims": self.q_dims, "full_dim": self.full_dim,
                    "block_size": self.block_size, "mrl_dims": self.mrl_dims}, path)

    @torch.no_grad()
    def load(self, path):
        state = torch.load(path, map_location=self.U.device, weights_only=True)
        if state["q_dims"] != self.q_dims or state["full_dim"] != self.full_dim:
            raise ValueError("Flag checkpoint q_dims/full_dim does not match the model.")
        saved_mrl_dims = state.get("mrl_dims", sorted(set(state["q_dims"] + [state["full_dim"]])))
        if state.get("block_size") != self.block_size or saved_mrl_dims != self.mrl_dims:
            raise ValueError("Flag checkpoint block_size/MRL dimensions do not match the model.")
        basis = state["U"].float()
        if basis.shape != self.U.shape or not torch.isfinite(basis).all():
            raise ValueError("Invalid flag checkpoint basis.")
        error = (basis.T @ basis - torch.eye(basis.size(1), device=basis.device)).abs().max()
        if error > 1e-3:
            raise ValueError(f"Flag checkpoint is not orthonormal: {error.item()}")
        self.U.copy_(basis)
