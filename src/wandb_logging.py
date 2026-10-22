"""Optional, rank-zero experiment logging for the custom train/eval loops."""

from dataclasses import asdict
import json
import os
from pathlib import Path

from src.utils import is_main_process


WANDB_API_KEY = "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"


def build_run_name(model_args, data_args, args, job_type, nested_dims=None):
    """Keep an optional user prefix and append the effective experiment settings."""
    prefix = args.run_name if args.run_name != args.output_dir else None
    parts = ([prefix] if prefix else []) + [Path(model_args.model_name).name, job_type]
    if job_type == "train":
        batch = args.per_device_train_batch_size
        accum = args.gradient_accumulation_steps
        world = int(os.environ.get("WORLD_SIZE", "1"))
        parts += [args.kd_loss_type, args.optimizer_name, f"lr{args.learning_rate:g}",
                  f"bs{batch}", f"ga{accum}", f"ebs{batch * accum * world}",
                  f"ep{args.num_train_epochs:g}", f"wd{args.weight_decay:g}",
                  f"t{model_args.temperature:g}"]
        if model_args.lora:
            parts.append(f"lora-r{model_args.lora_r}-a{model_args.lora_alpha}")
    else:
        if model_args.checkpoint_path:
            checkpoint = Path(model_args.checkpoint_path)
            parts.append(f"ckpt-{checkpoint.parent.name}-{checkpoint.name}")
        parts += [f"bs{args.per_device_eval_batch_size}", f"prefix{int(data_args.tgt_prefix_mod)}"]
    dims = args.nested_dims if nested_dims is None else nested_dims
    parts.append("dims" + ("-".join(map(str, dims)) if dims else "full"))
    if job_type == "train" and args.kd_loss_type == "flag_mrl":
        parts += [f"flr{args.flag_lr:g}", f"lam{args.flag_lambda_neg:g}"]
        parts.append(f"init-{args.flag_init}")
        if args.flag_init in {"spherical", "geneig"}:
            batches = args.flag_geneig_batches if args.flag_geneig_batches is not None else args.flag_init_batches
            parts.append(f"initb{batches}")
    parts += [f"pool-{model_args.pooling}", f"norm{int(model_args.normalize)}",
              f"res-{data_args.image_resolution or 'original'}", f"seed{args.seed}"]
    return "__".join(parts)


class WandbLogger:
    def __init__(self, model_args, data_args, training_args, job_type, nested_dims=None):
        self.model_args = model_args
        self.data_args = data_args
        self.args = training_args
        self.job_type = job_type
        self.nested_dims = nested_dims
        self.run = None
        self._sums, self._counts, self._latest = {}, {}, {}

    def __enter__(self):
        if not self.args.wandb:
            return self
        # RANK covers standalone eval under torchrun before dist is initialized.
        if not is_main_process() or int(os.environ.get("RANK", "0")) != 0:
            return self
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("W&B logging requires wandb. Install with: python -m pip install wandb; "
                               "or disable with --wandb False.") from exc
        if WANDB_API_KEY and os.environ.get("WANDB_MODE", "online") not in {"offline", "disabled"}:
            wandb.login(key=WANDB_API_KEY, relogin=True)
        directory = (self.data_args.encode_output_path if self.job_type == "eval"
                     else self.args.output_dir)
        Path(directory).mkdir(parents=True, exist_ok=True)
        name = build_run_name(self.model_args, self.data_args, self.args, self.job_type, self.nested_dims)
        training_config = self.args.to_dict()
        training_config["run_name"] = name
        self.run = wandb.init(
            project=self.args.project_name or "Flag2Vec",
            name=name,
            job_type=self.job_type,
            dir=str(directory),
            config={"model": asdict(self.model_args), "data": asdict(self.data_args),
                    "training": training_config},
        )
        self.run.define_metric("train/global_step")
        self.run.define_metric("train/*", step_metric="train/global_step")
        self.run.define_metric("flag/*", step_metric="train/global_step")
        print(f"W&B {self.job_type} run: {self.run.url or self.run.name}", flush=True)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.run is not None:
            try:
                self.flush_train()
            finally:
                self.run.finish(exit_code=0 if exc_type is None else 1)

    def log_train(self, metrics, commit=False):
        """Average microbatch metrics over the interval between optimizer-step logs."""
        if self.run is None:
            return
        self._latest = {key: metrics[key] for key in ("step", "micro_step", "epoch", "learning_rate")}
        for key, value in metrics.items():
            if key not in self._latest:
                self._sums[key] = self._sums.get(key, 0.0) + value
                self._counts[key] = self._counts.get(key, 0) + 1
        if commit:
            self.flush_train()

    def flush_train(self):
        if self.run is None or not self._sums:
            return
        values = {}
        for key, total in self._sums.items():
            metric = f"flag/{key[5:]}" if key.startswith("flag_") else f"train/{key}"
            values[metric] = total / self._counts[key]
        values.update({f"train/{key}": value for key, value in self._latest.items() if key != "step"})
        values["train/global_step"] = self._latest["step"]
        self.run.log(values)
        self._sums.clear()
        self._counts.clear()

    def log_eval(self, subset, result, cached=False):
        if self.run is None:
            return
        prefix = f"eval/{subset or 'default'}"
        values = {f"{prefix}/{key}": result[key]
                  for key in ("acc", "num_correct", "num_pred", "num_data") if key in result}
        values[f"{prefix}/cached"] = int(cached)
        for dim, metrics in result.get("matryoshka", {}).items():
            values.update({f"{prefix}/dim_{dim}/{key}": value for key, value in metrics.items()})
        self.run.log(values)
        self.run.summary.update(values)

    def log_eval_average(self, results):
        """Macro-average subset accuracies; record coverage for each dimension."""
        if not results:
            return
        values = {"eval/average/acc": sum(result["acc"] for result in results) / len(results),
                  "eval/average/num_subsets": len(results)}
        per_dim = {}
        for result in results:
            for dim, metrics in result.get("matryoshka", {}).items():
                per_dim.setdefault(dim, []).append(metrics["acc"])
        for dim, accuracies in per_dim.items():
            values[f"eval/average/dim_{dim}/acc"] = sum(accuracies) / len(accuracies)
            values[f"eval/average/dim_{dim}/num_subsets"] = len(accuracies)
        print(json.dumps(values), flush=True)
        if self.run is None:
            return
        self.run.log(values)
        self.run.summary.update(values)
