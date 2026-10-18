"""Shared dimensions, distributed helpers, and device movement."""

from collections.abc import Mapping

import torch
import torch.distributed as dist


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def print_rank(*values, **kwargs):
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f"[rank {rank}]", *values, **kwargs)


def print_master(*values, **kwargs):
    if is_main_process():
        print(*values, **kwargs)


def resolve_nested_dims(nested_dims, full_dim):
    """One fallback shared by attachment, losses, and evaluation."""
    if full_dim < 2:
        raise ValueError("The embedding dimension must be at least two.")
    dims = list(nested_dims) if nested_dims else [d for d in (64, 128, 256, 512, 768, 1024) if d < full_dim] + [full_dim]
    if not nested_dims and len(dims) == 1:
        dims.insert(0, full_dim // 2)
    if any(not isinstance(d, int) or d <= 0 or d > full_dim for d in dims):
        raise ValueError(f"nested_dims must lie in [1, {full_dim}]: {dims}")
    if len(set(dims)) != len(dims):
        raise ValueError("nested_dims must not contain duplicates.")
    return sorted(dims)


def gather_tensor(tensor, keep_local_grad=True):
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor
    tensor = tensor.contiguous()
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor.detach())
    if keep_local_grad:
        gathered[dist.get_rank()] = tensor
    return torch.cat(gathered, dim=0)


def to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, Mapping):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device) for v in obj)
    return obj
