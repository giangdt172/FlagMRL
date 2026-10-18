"""Multilevel contrastive loss using a detached nested flag."""

import math
import time

import torch
from torch import nn
from torch.nn import functional as F

from src.MRL.base_mrl import contrastive_inputs
from src.MRL.flag.pairs import build_pair_weights
from src.utils import gather_tensor


class FlagMRLLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
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
        q, p, target = contrastive_inputs(model, batch)
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
                if valid:
                    flag_metrics = projector.step((qn, pn, near), (qn, pn, far))
            if q.device.type == "cuda":
                torch.cuda.synchronize(q.device)
            flag_metrics["flag_step_ms"] = (time.perf_counter() - start) * 1000
        basis = projector.U.detach().clone()
        dims = projector.q_dims + [projector.full_dim]
        losses, metrics = [], {}
        with torch.autocast(device_type=q.device.type, enabled=False):
            q_full, p_full = q.float(), p.float()
            # Project into the largest learned flag space once. Every smaller
            # level is a prefix because the columns of U form a nested flag:
            # (X @ U)[:, :dim] == X @ U[:, :dim].
            q_projected = q_full @ basis
            p_projected = p_full @ basis
            for dim in dims:
                q_level = q_full if dim == projector.full_dim else q_projected[:, :dim]
                p_level = p_full if dim == projector.full_dim else p_projected[:, :dim]
                scores = model.compute_similarity(F.normalize(q_level, dim=-1), F.normalize(p_level, dim=-1))
                loss = F.cross_entropy(scores / model_trainer.temperature, target)
                losses.append(loss)
                metrics[f"loss_dim_{dim}"] = loss.detach()
                metrics[f"contrastive_loss_dim_{dim}"] = loss.detach()
        loss = sum(w * level for w, level in zip(self.level_weights(len(losses)), losses))
        if self.training:
            self.step_idx.add_(1)
        return {"loss": loss, "contrastive_loss": loss, **metrics, **flag_metrics}
