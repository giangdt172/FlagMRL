"""Non-parameter flag state, always stored in FP32."""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class FlagProjector(nn.Module):
    def __init__(self, full_dim, q_dims, init="random", eps=1e-4, lr=0.1, retraction="svd", seed=0):
        super().__init__()
        self.full_dim = int(full_dim)
        self.q_dims = sorted(q_dims)
        if not self.q_dims or len(set(self.q_dims)) != len(self.q_dims) or not 0 < self.q_dims[0] <= self.q_dims[-1] < full_dim:
            raise ValueError("Flag levels must be unique positive dimensions strictly below full_dim.")
        if eps <= 0 or lr < 0:
            raise ValueError("flag_eps must be positive and flag_lr nonnegative.")
        if init not in {"random", "identity", "geneig"}:
            raise ValueError(f"Unknown flag_init: {init}")
        if retraction not in {"svd", "newton_schulz"}:
            raise ValueError(f"Unknown flag_retraction: {retraction}")
        self.eps, self.lr, self.retraction = eps, lr, retraction
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
        with torch.autocast(device_type=x.device.type, enabled=False):
            return F.normalize(x.float() @ self.U[:, :self.q_dims[k]].detach().clone(), dim=-1)

    @torch.no_grad()
    def step(self, near, far):
        from .flag_math import flag_loss_and_grad, stiefel_step, principal_angle, orthogonality_error
        with torch.autocast(device_type=self.U.device.type, enabled=False):
            loss, rho, gradient = flag_loss_and_grad(self.U, self.a_col, near, far, self.eps)
            updated, norm = stiefel_step(self.U, gradient, self.lr, self.retraction)
            drift = principal_angle(self.U, updated)
            self.U.copy_(updated)
            return {"flag_loss": loss, "flag_rho": rho, "flag_gnorm": norm,
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
        torch.save({"U": self.U.detach().cpu(), "q_dims": self.q_dims, "full_dim": self.full_dim}, path)

    @torch.no_grad()
    def load(self, path):
        state = torch.load(path, map_location=self.U.device, weights_only=True)
        if state["q_dims"] != self.q_dims or state["full_dim"] != self.full_dim:
            raise ValueError("Flag checkpoint q_dims/full_dim does not match the model.")
        basis = state["U"].float()
        if basis.shape != self.U.shape or not torch.isfinite(basis).all():
            raise ValueError("Invalid flag checkpoint basis.")
        error = (basis.T @ basis - torch.eye(basis.size(1), device=basis.device)).abs().max()
        if error > 1e-5:
            raise ValueError(f"Flag checkpoint is not orthonormal: {error.item()}")
        self.U.copy_(basis)
