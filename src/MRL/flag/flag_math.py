"""Matrix-free scatter products and FP32 Stiefel optimization."""

import torch


@torch.no_grad()
def cov_matmul(Q, P, W, X):
    """Apply the uncentered weighted difference scatter without pair vectors."""
    with torch.autocast(device_type=Q.device.type, enabled=False):
        Q, P, W, X = (t.detach().float() for t in (Q, P, W, X))
        qx, px = Q @ X, P @ X
        rows, columns = W.sum(1, keepdim=True), W.sum(0).unsqueeze(1)
        return Q.T @ (rows * qx - W @ px) + P.T @ (columns * px - W.T @ qx)


@torch.no_grad()
def flag_loss_and_grad(U, a_col, near, far, eps):
    if eps <= 0:
        raise ValueError("The ridge eps must be positive.")
    with torch.autocast(device_type=U.device.type, enabled=False):
        U, a_col = U.detach().float(), a_col.detach().float()
        cf_u, cn_u = cov_matmul(*far, U), cov_matmul(*near, U)
        numerator = (a_col * (U * cf_u).sum(0)).sum()
        denominator = (a_col * (U * cn_u).sum(0)).sum() + eps * a_col.sum()
        if not torch.isfinite(denominator) or denominator <= 0:
            raise FloatingPointError("The flag trace-ratio denominator is not positive and finite.")
        rho = numerator / denominator
        gradient = (-2 / denominator) * (cf_u - rho * (cn_u + eps * U)) * a_col.unsqueeze(0)
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("Non-finite flag gradient.")
        return -rho, rho, gradient


@torch.no_grad()
def tangent_gradient(U, H):
    with torch.autocast(device_type=U.device.type, enabled=False):
        U, H = U.detach().float(), H.detach().float()
        product = U.T @ H
        return H - U @ ((product + product.T) * 0.5)


@torch.no_grad()
def _polar_svd(Y):
    left, _, right = torch.linalg.svd(Y, full_matrices=False)
    return left @ right


@torch.no_grad()
def _polar_newton_schulz(Y):
    """Inverse square root on q-by-q Gram; fall back to SVD if ill-conditioned."""
    gram = Y.T @ Y
    scale = torch.linalg.matrix_norm(gram)
    identity = torch.eye(gram.size(0), device=Y.device, dtype=torch.float32)
    root, inverse = gram / scale, identity.clone()
    for _ in range(30):
        correction = (3 * identity - inverse @ root) * 0.5
        root, inverse = root @ correction, correction @ inverse
        if (identity - inverse @ root).abs().max() < 5e-7:
            break
    basis = Y @ inverse / scale.sqrt()
    if not torch.isfinite(basis).all() or orthogonality_error(basis) > 5e-6:
        return _polar_svd(Y)
    return basis


@torch.no_grad()
def stiefel_step(U, H, lr, retraction="svd"):
    if lr < 0:
        raise ValueError("Flag learning rate must be nonnegative.")
    with torch.autocast(device_type=U.device.type, enabled=False):
        U, H = U.detach().float(), H.detach().float()
        gradient = tangent_gradient(U, H)
        if lr == 0:
            return U.clone(), torch.linalg.vector_norm(gradient)
        candidate = U - lr * gradient
        if retraction == "svd":
            basis = _polar_svd(candidate)
        elif retraction == "newton_schulz":
            basis = _polar_newton_schulz(candidate)
        else:
            raise ValueError(f"Unsupported retraction: {retraction}")
        return basis, torch.linalg.vector_norm(gradient)


@torch.no_grad()
def orthogonality_error(U):
    with torch.autocast(device_type=U.device.type, enabled=False):
        U = U.detach().float()
        return (U.T @ U - torch.eye(U.size(1), device=U.device)).abs().max()


@torch.no_grad()
def principal_angle(U1, U2):
    with torch.autocast(device_type=U1.device.type, enabled=False):
        if torch.equal(U1, U2):
            return torch.zeros((), device=U1.device, dtype=torch.float32)
        values = torch.linalg.svdvals(U1.detach().float().T @ U2.detach().float())
        return values.min().clamp(0, 1).acos()
