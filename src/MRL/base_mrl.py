"""Prefix Matryoshka contrastive loss following vlm_embed."""

import torch
from torch import nn
from torch.nn import functional as F

from src.utils import gather_tensor


def contrastive_inputs(model, batch):
    query_reps = gather_tensor(model.encode_input(batch["qry"])[0])
    positive_reps = gather_tensor(model.encode_input(batch["pos"])[0])
    if query_reps.size(0) == 0 or positive_reps.size(0) % query_reps.size(0) or positive_reps.size(0) < query_reps.size(0):
        raise ValueError("Positive count must be a nonzero integer multiple of query count.")
    target_per_query = positive_reps.size(0) // query_reps.size(0)
    target = torch.arange(query_reps.size(0), device=query_reps.device, dtype=torch.long) * target_per_query
    return query_reps, positive_reps, target


class MatryoshkaContrastiveLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.cross_entropy = nn.CrossEntropyLoss()
        self.nested_dims = getattr(args, "nested_dims", [64, 128, 256, 512, 1024])
        self.average_loss = getattr(args, "average_loss", True)

    def _dist_gather_tensor(self, tensor):
        return gather_tensor(tensor)

    def forward(self, model_trainer, input_data):
        model = model_trainer.model
        query_reps, positive_reps, target = contrastive_inputs(model, input_data)
        full_dim = query_reps.size(1)
        total_loss, num_dims = 0.0, 0
        dim_losses = {}
        for dim in self.nested_dims:
            if dim > full_dim:
                break
            if dim <= 0:
                raise ValueError("MRL dimensions must be positive.")
            query = F.normalize(query_reps[:, :dim], p=2, dim=1)
            positive = F.normalize(positive_reps[:, :dim], p=2, dim=1)
            scores = model.compute_similarity(query, positive).view(query.size(0), -1)
            loss = self.cross_entropy(scores / model_trainer.temperature, target)
            total_loss = total_loss + loss
            num_dims += 1
            dim_losses[f"contrastive_loss_dim_{dim}"] = loss.detach().item()
        if not num_dims:
            raise ValueError("MRL requires at least one nested_dim within the encoder's full dimension.")
        if self.average_loss:
            total_loss = total_loss / num_dims
        return {"loss": total_loss, "contrastive_loss": total_loss, **dim_losses}
