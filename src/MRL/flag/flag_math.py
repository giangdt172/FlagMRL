"""Spherical discrimination and FP32 Stiefel steps."""

import math
import torch


@torch.no_grad()
def flag_loss_and_grad(U, a_col, near, far, eps, lambda_neg=1.0):
    """Minimize -S_positive + lambda_neg*S_negative with frozen embeddings.

    Pair weights average within each query, then over queries with negatives.
    Weighted coordinates implement A**(1/2), avoiding a full D-by-D metric.
    A local autograd leaf includes the derivative of cosine normalization even
    when this function is called from the no_grad flag-update context.
    """
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("The squared-norm regularizer eps must be positive and finite.")
    if not math.isfinite(lambda_neg) or lambda_neg < 0:
        raise ValueError("flag_lambda_neg must be nonnegative and finite.")
    with torch.enable_grad(), torch.autocast(device_type=U.device.type, enabled=False):
        basis = U.detach().float().requires_grad_(True)
        scale = a_col.detach().float().sqrt()

        def cosine_matrix(pairs):
            query, target = (t.detach().float() for t in pairs[:2])
            eq, et = (query @ basis) * scale, (target @ basis) * scale
            eq = eq / (eq.square().sum(-1, keepdim=True) + eps).sqrt()
            et = et / (et.square().sum(-1, keepdim=True) + eps).sqrt()
            return eq @ et.T

        near_cosine = cosine_matrix(near)
        # Training uses the same query/target pool for both relations.
        far_cosine = near_cosine if near[0] is far[0] and near[1] is far[1] else cosine_matrix(far)
        positive = (near_cosine * near[2].detach().float()).sum()
        negative = (far_cosine * far[2].detach().float()).sum()
        loss = -positive + lambda_neg * negative
        gradient, = torch.autograd.grad(loss, basis)
        if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
            raise FloatingPointError("Non-finite spherical flag loss or gradient.")
        return loss.detach(), positive.detach(), negative.detach(), gradient.detach()


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
    """Largest principal angle; residual sines remain accurate for small steps."""
    with torch.autocast(device_type=U1.device.type, enabled=False):
        if torch.equal(U1, U2):
            return torch.zeros((), device=U1.device, dtype=torch.float32)
        U1, U2 = U1.detach().float(), U2.detach().float()
        residual = U2 - U1 @ (U1.T @ U2)
        return torch.linalg.svdvals(residual).max().clamp(0, 1).asin()
