import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from transformers import (HfArgumentParser, get_constant_schedule,
                          get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup, set_seed)

from prepare_one_model_training import OneModelTrainer, TrainOneModelCollator, TrainOneModelDataset
from src.arguments import DataArguments, ModelArguments, TrainingArguments
from src.MRL import build_criterion
from src.utils import is_main_process, print_master, to_device, print_rank


class CombinedOptimizer(torch.optim.Optimizer):
    """Muon/AdamW wrapper; Optimizer inheritance is required by current schedulers."""

    def __init__(self, optimizers):
        self.optimizers = list(optimizers)
        super().__init__([group for optimizer in self.optimizers for group in optimizer.param_groups], {})

    def step(self, closure=None):
        loss = None
        for optimizer in self.optimizers:
            result = optimizer.step(closure) if closure is not None else optimizer.step()
            if result is not None:
                loss = result
        return loss

    def zero_grad(self, set_to_none=False):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"optimizers": [optimizer.state_dict() for optimizer in self.optimizers]}

    def load_state_dict(self, state_dict):
        states = state_dict["optimizers"]
        if len(states) != len(self.optimizers):
            raise ValueError("CombinedOptimizer checkpoint has a different number of optimizers.")
        for optimizer, state in zip(self.optimizers, states):
            optimizer.load_state_dict(state)
        self.param_groups = [group for optimizer in self.optimizers for group in optimizer.param_groups]


def get_optimizer_params(model, training_args):
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if p.requires_grad]},
    ]

    return optimizer_grouped_parameters


def get_optimizer(model, training_args):
    while isinstance(model, DDP):
        model = model.module
    optimizer_grouped_parameters = get_optimizer_params(model, training_args)
    optimizer_name = str(getattr(training_args, "optimizer_name", "adamw")).lower()

    if optimizer_name in {"adamw", "adam"}:
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=training_args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=training_args.weight_decay,
        )
        return optimizer

    if optimizer_name in {"moon", "muon"}:
        muon_cls = getattr(torch.optim, "Muon", None)
        if muon_cls is None:
            raise RuntimeError(
                "--optimizer_name moon requested, but torch.optim.Muon is not available in this PyTorch build. "
                "Please upgrade PyTorch to a version that includes Muon."
            )
        param_optimizer = list(model.named_parameters())
        muon_params = [p for _, p in param_optimizer if p.requires_grad and p.ndim == 2]
        non_muon_params = [p for _, p in param_optimizer if p.requires_grad and p.ndim != 2]
        non_2d_strategy = str(getattr(training_args, "moon_non_2d_strategy", "hybrid")).lower()
        if non_2d_strategy not in {"hybrid", "skip", "error"}:
            raise ValueError(
                f"Invalid moon_non_2d_strategy={non_2d_strategy}. "
                "Use one of: hybrid, skip, error."
            )

        if not muon_params:
            raise RuntimeError(
                "--optimizer_name moon requested, but no trainable 2D parameters were found for Muon."
            )

        muon_optimizer = muon_cls(
            [{"params": muon_params}],
            lr=training_args.learning_rate,
            weight_decay=training_args.weight_decay,
        )

        if non_muon_params and non_2d_strategy == "error":
            raise RuntimeError(
                "--optimizer_name moon with --moon_non_2d_strategy error: found non-2D trainable params. "
                "Use --moon_non_2d_strategy hybrid or skip."
            )

        if non_muon_params and non_2d_strategy == "hybrid":
            adamw_optimizer = AdamW(
                [{"params": non_muon_params}],
                lr=training_args.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=training_args.weight_decay,
            )
            print_rank(
                "Using hybrid optimizer: Muon for 2D params, AdamW for non-2D params "
                f"(counts: muon={len(muon_params)}, adamw={len(non_muon_params)})."
            )
            return CombinedOptimizer([muon_optimizer, adamw_optimizer])

        if non_muon_params and non_2d_strategy == "skip":
            print_rank(
                "Using Muon-only optimizer; non-2D trainable params are excluded from optimization "
                f"(counts: muon={len(muon_params)}, skipped={len(non_muon_params)})."
            )
        else:
            print_rank(f"Using torch.optim.Muon optimizer on all trainable params ({len(muon_params)} tensors).")

        return muon_optimizer

    raise ValueError(f"Unsupported optimizer_name={optimizer_name}. Use one of: adamw, moon")


def ddp_setup():
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    init_process_group(backend="gloo")


@torch.no_grad()
def initialize_geneig(module, loader, args):
    if args.flag_init != "geneig" or not hasattr(module.model, "flag_projector"):
        return
    checkpoint = Path(module.model_args.checkpoint_path or module.model_args.model_name) / "flag_U.pt"
    if checkpoint.exists():
        return
    if args.flag_geneig_batches < 1:
        raise ValueError("flag_geneig_batches must be positive.")
    from src.MRL.base_mrl import contrastive_inputs
    from src.MRL.flag.flag_math import cov_matmul
    from src.MRL.flag.pairs import build_pair_weights
    from src.utils import gather_tensor
    from torch.nn import functional as F
    projector = module.model.flag_projector
    p = projector.full_dim
    cn, cf = torch.zeros(p, p, device=module.device), torch.zeros(p, p, device=module.device)
    eye = torch.eye(p, device=module.device)
    was_training = module.training
    module.eval()
    iterator, valid_count = iter(loader), 0
    try:
        for _ in range(args.flag_geneig_batches):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = to_device(batch, module.device)
            q, pos, target = contrastive_inputs(module.model, batch)
            with torch.autocast(device_type=module.device.type, enabled=False):
                q, pos = F.normalize(q.float(), dim=-1), F.normalize(pos.float(), dim=-1)
                ids = gather_tensor(batch["pos_id"], keep_local_grad=False) if args.flag_use_neg_mask else None
                near, far, ok = build_pair_weights(q, pos, target, ids, args.flag_far_mode, args.flag_far_tau)
                if ok:
                    cn += cov_matmul(q, pos, near, eye)
                    cf += cov_matmul(q, pos, far, eye)
                    valid_count += 1
        if valid_count == 0:
            raise ValueError("geneig initialization has no valid negative pairs.")
        if is_main_process():
            projector.initialize_geneig(cn / valid_count, cf / valid_count)
        if dist.is_initialized():
            dist.broadcast(projector.U, src=0)
    finally:
        module.train(was_training)
    print_master(f"Initialized flag from {valid_count} frozen-encoder batches.")


class Trainer:
    def __init__(self, trainer, train_data, optimizer, lr_scheduler, criterion, model_args, training_args):
        self.module = trainer
        self.device = trainer.device
        self.train_data, self.optimizer, self.lr_scheduler = train_data, optimizer, lr_scheduler
        self.criterion, self.model_args, self.training_args = criterion, model_args, training_args
        self.trainer = trainer
        if dist.is_initialized():
            self.trainer = DDP(trainer, device_ids=[self.device.index] if self.device.type == "cuda" else None,
                               find_unused_parameters=False, static_graph=training_args.gradient_checkpointing)
        self.global_step, self.micro_step = 0, 0
        self.output_dir = Path(training_args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_epoch(self, epoch):
        args = self.training_args
        self.train_data.sampler.set_epoch(epoch)
        self.trainer.train()
        accum = args.gradient_accumulation_steps
        usable_batches = len(self.train_data) // accum * accum
        for batch_idx, batch in enumerate(self.train_data):
            if batch_idx >= usable_batches:
                break
            start = time.perf_counter()
            boundary = (batch_idx + 1) % accum == 0

            loss_dict = self.trainer(self.criterion, to_device(batch, self.device))
            loss = loss_dict["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss.")
            (loss / accum).backward()
            self.micro_step += 1
            if boundary:
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            step_ms = (time.perf_counter() - start) * 1000
            if is_main_process():
                metrics = {k: float(v.detach()) if isinstance(v, torch.Tensor) else float(v)
                           for k, v in loss_dict.items() if isinstance(v, (int, float)) or isinstance(v, torch.Tensor) and v.numel() == 1}
                metrics.update(step=self.global_step, micro_step=self.micro_step, epoch=epoch,
                               step_ms=step_ms, learning_rate=self.optimizer.param_groups[0]["lr"])
                with (self.output_dir / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(metrics, allow_nan=False) + "\n")
                if boundary and self.global_step % args.logging_steps == 0:
                    print_master(f"step={self.global_step} loss={float(loss.detach()):.6f}")
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            if args.max_steps > 0 and self.global_step >= args.max_steps:
                break

    def train(self):
        args = self.training_args
        steps_per_epoch = len(self.train_data) // args.gradient_accumulation_steps
        if steps_per_epoch < 1:
            raise ValueError("Dataset is too small for a complete optimizer step on every rank.")
        epochs = math.ceil(args.max_steps / steps_per_epoch) if args.max_steps > 0 else math.ceil(args.num_train_epochs)
        self.optimizer.zero_grad(set_to_none=True)
        for epoch in range(epochs):
            if args.max_steps > 0 and self.global_step >= args.max_steps:
                break
            self.run_epoch(epoch)
            if is_main_process() and getattr(args.save_strategy, "value", args.save_strategy) == "epoch":
                path = self.output_dir / f"checkpoint-epoch-{epoch}"
                path.mkdir(parents=True, exist_ok=True)
                self.module.model.save(path)
                processor = getattr(self.train_data.collate_fn, "processor", None)
                if processor is None:
                    processor = self.module.get_processor()
                processor.save_pretrained(path)
                print_master(f"Saved checkpoint to {path}")
            if dist.is_initialized():
                dist.barrier()
            if args.max_steps > 0 and self.global_step >= args.max_steps:
                break


def normalize_local_rank(argv):
    result = []
    for arg in argv:
        if arg.startswith(("--local_rank=", "--local-rank=")):
            result.extend(["--local_rank", arg.split("=", 1)[1]])
        else:
            result.append("--local_rank" if arg == "--local-rank" else arg)
    return result


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, args = parser.parse_args_into_dataclasses(args=normalize_local_rank(sys.argv[1:]))
    if not args.output_dir:
        raise ValueError("--output_dir is required.")
    set_seed(args.seed)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() and not args.use_cpu else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    module = OneModelTrainer(model_args, args, device)
    dataset = TrainOneModelDataset(data_args, model_args)
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=args.per_device_train_batch_size, sampler=sampler,
                        collate_fn=TrainOneModelCollator(module.get_processor(), model_args, data_args, args),
                        drop_last=True, num_workers=0, pin_memory=False,
                        generator=torch.Generator().manual_seed(args.seed))
    for name, parameter in module.model.named_parameters():
        if "mm_projector" in name or "multi_modal_projector" in name:
            parameter.requires_grad_(True)
        if "vision_tower" in name or args.image_encoder_freeze and ("visual" in name or "vision" in name):
            parameter.requires_grad_(False)
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.bfloat16)
    if hasattr(module.model, "flag_projector"):
        assert module.model.flag_projector.U.dtype == torch.float32
    optimizer = get_optimizer(module.model, args)
    total_steps = (len(dataset) // (args.per_device_train_batch_size * world) // args.gradient_accumulation_steps) * args.num_train_epochs
    if args.max_steps > 0:
        total_steps = args.max_steps
    if total_steps < 1:
        raise ValueError("No complete optimizer steps; increase data size or decrease batch/accumulation.")
    if args.lr_scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=(args.warmup_ratio or 0) * total_steps,
            num_training_steps=total_steps,
        )
    elif args.lr_scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=(args.warmup_ratio or 0) * total_steps,
            num_training_steps=total_steps,
        )
    else:
        scheduler = get_constant_schedule(optimizer)
    criterion = build_criterion(args).to(device)
    initialize_geneig(module, loader, args)
    Trainer(module, loader, optimizer, scheduler, criterion, model_args, args).train()


if __name__ == "__main__":
    start_time = time.perf_counter()
    ddp_setup()
    main_process = is_main_process()
    try:
        main()
    finally:
        destroy_process_group()
    if main_process:
        print(f"Total runtime: {time.perf_counter() - start_time:.2f} seconds", flush=True)
