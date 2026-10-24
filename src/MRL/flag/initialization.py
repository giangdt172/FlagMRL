"""Warm up the flag basis with the training objective and a frozen encoder."""

from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import functional as F

from src.MRL import criterion_dict
from src.MRL.base_mrl import contrastive_inputs
from src.MRL.flag.pairs import build_pair_weights
from src.utils import gather_tensor, is_main_process, print_master, to_device


@torch.no_grad()
def initialize_flag(module, loader, args):
    # Keep legacy init names; warm up with the selected training objective.
    if args.flag_init not in {"spherical", "geneig"} or not hasattr(module.model, "flag_projector"):
        return
    checkpoint = Path(module.model_args.checkpoint_path or module.model_args.model_name) / "flag_U.pt"
    if checkpoint.exists():
        return
    legacy_batches = getattr(args, "flag_geneig_batches", None)
    batches = args.flag_init_batches if legacy_batches is None else legacy_batches
    if batches < 1:
        raise ValueError("flag_init_batches must be positive.")
    projector = module.model.flag_projector
    objective = criterion_dict[args.kd_loss_type].flag_objective
    if len(loader) == 0:
        print_master(f"Skipped {objective} flag initialization: empty loader; keeping the initial basis.")
        return

    # Restore individual modes too, including deliberately frozen/eval submodules.
    training_modes = [(child, child.training) for child in module.modules()]
    module.eval()
    valid_count = 0
    try:
        iterator = iter(loader)
        for index in range(batches):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = to_device(batch, module.device)
            queries, targets, positive_indices = contrastive_inputs(module.model, batch)
            with torch.autocast(device_type=module.device.type, enabled=False):
                queries = F.normalize(queries.float(), dim=-1)
                targets = F.normalize(targets.float(), dim=-1)
                identities = None
                if args.flag_use_neg_mask:
                    if "pos_id" not in batch:
                        raise ValueError("flag_use_neg_mask requires pos_id for every positive column.")
                    identities = gather_tensor(batch["pos_id"], keep_local_grad=False)
                near, far, valid = build_pair_weights(
                    queries, targets, positive_indices, identities, args.flag_far_mode, args.flag_far_tau)
                if valid:
                    valid_count += 1
                    if is_main_process():
                        metrics = projector.step(
                            (queries, targets, near), (queries, targets, far), objective=objective)
                        diagnostic = (f"ratio={float(metrics['flag_trace_ratio']):.6f} "
                                      if objective == "trace_ratio" else f"gap={float(metrics['flag_gap']):.6f} ")
                        print_master(
                            f"Flag {objective} init {index + 1}/{batches}: "
                            f"loss={float(metrics['flag_loss']):.6f} "
                            f"{diagnostic}"
                            f"orth_err={float(metrics['flag_orth_err']):.3e}", flush=True)
        if dist.is_initialized():
            dist.broadcast(projector.U, src=0)
    finally:
        for child, was_training in training_modes:
            child.training = was_training
    if valid_count:
        print_master(f"Initialized flag with {valid_count} {objective} updates and a frozen encoder.", flush=True)
    else:
        print_master(f"Skipped {objective} flag initialization: no valid negative pairs; keeping the initial basis.",
                     flush=True)
