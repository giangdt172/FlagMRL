"""Spherical flag updates, then multilevel InfoNCE and optional cross-dim KD."""

import math
import time

import torch
from torch import nn
from torch.nn import functional as F

from src.MRL.base_mrl import contrastive_inputs
from src.MRL.flag.encoder_loss import encoder_losses
from src.MRL.flag.pairs import build_pair_weights
from src.utils import gather_tensor


class FlagMRLLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.lambda_cross = getattr(args, "flag_lambda_cross", 0.0)
        if not math.isfinite(self.lambda_cross) or self.lambda_cross < 0:
            raise ValueError("flag_lambda_cross must be nonnegative and finite.")
        self.register_buffer("step_idx", torch.zeros((), dtype=torch.long))
        if args.flag_update_every < 1:
            raise ValueError("flag_update_every must be positive.")
        if args.flag_far_mode not in {"uniform", "hard"} or args.flag_far_tau <= 0:
            raise ValueError("Invalid far mode or temperature.")

    def level_weights(self, count):
        if not self.args.flag_level_weights.strip():
            return [1 / count] * count
        weights = [float(w) for w in self.args.flag_level_weights.replace(",", " ").split()]
        if len(weights) != count or any(not math.isfinite(w) or w <= 0 for w in weights) or not math.isclose(sum(weights), 1.0, abs_tol=1e-6):
            raise ValueError(f"flag_level_weights must contain {count} positive weights summing to one.")
        return weights

    def forward(self, model_trainer, batch):
        model = model_trainer.model
        # Each projected prefix is normalized separately by the encoder loss.
        q, p, target = contrastive_inputs(model, batch, normalize=False)
        projector = model.flag_projector
        flag_metrics = {"flag_skipped": 1.0, "flag_step_ms": 0.0}
        if self.training and int(self.step_idx) % self.args.flag_update_every == 0:
            if q.device.type == "cuda":
                torch.cuda.synchronize(q.device)
            start = time.perf_counter()
            with torch.no_grad(), torch.autocast(device_type=q.device.type, enabled=False):
                qn, pn = F.normalize(q.detach().float(), dim=-1), F.normalize(p.detach().float(), dim=-1)
                identities = None
                if self.args.flag_use_neg_mask:
                    if "pos_id" not in batch:
                        raise ValueError("flag_use_neg_mask requires pos_id for every positive column.")
                    identities = gather_tensor(batch["pos_id"], keep_local_grad=False)
                near, far, valid = build_pair_weights(qn, pn, target, identities,
                                                      self.args.flag_far_mode, self.args.flag_far_tau)
                # No negative annotations are needed: other gathered targets
                # supply negatives. If none remain, keep U and still run MRL.
                if valid:
                    flag_metrics = projector.step((qn, pn, near), (qn, pn, far))
            if q.device.type == "cuda":
                torch.cuda.synchronize(q.device)
            flag_metrics["flag_step_ms"] = (time.perf_counter() - start) * 1000
        basis = projector.U.detach().clone()
        dims = projector.mrl_dims
        if basis.size(1) < dims[-1]:
            raise ValueError("The flag basis must cover every MRL dimension, including full_dim.")
        with torch.autocast(device_type=q.device.type, enabled=False):
            # One projection per side; all levels, including full_dim, use U.
            q_projected, p_projected = q.float() @ basis, p.float() @ basis
            loss, metrics = encoder_losses(
                model, q_projected, p_projected, target, dims, self.level_weights(len(dims)),
                model_trainer.temperature, lambda_cross=self.lambda_cross)
        if self.training:
            self.step_idx.add_(1)
        return {"loss": loss, "contrastive_loss": metrics["mrl_loss"],
                **metrics, **flag_metrics}
