#!/usr/bin/env python3
"""Standalone FlagMRL diagnostics; never modifies a production checkpoint.

Examples (run from the repository root)::

    python test.py self-test
    python test.py run --image-dir data/MMEB-eval
    python test.py run --datasets VOC2007 HatefulMemes --tests T00 T01
    python test.py run --cache-only --tests T00 T01 T02 T03 T04 T05 T06
    python test.py run --datasets VOC2007 --tests T07 T08 T09 \
        --training-config /path/to/original_training_config.json \
        --train-image-dir data/MMEB-train
    python test.py compare --baseline-cache /path/to/base/cache

All implementation and mathematical tests live in this file. Default checkpoint:
training/FlagMRL_FastVLM_cls/checkpoint-epoch-0. Outputs: test/<dataset>/,
test/summary.json and test/diagnostic_report.md. Raw arrays are memory-mapped NPY,
not old evaluator pickle caches (which can contain already-normalized vectors).
The default run performs A/B pilots first, then the remaining datasets. T06 is
opt-in; T07-T09 require original training configuration. T11 is deferred until
A-E evidence and complete controlled-training inputs exist. This command never
launches encoder training.
"""

import argparse
import contextlib
import csv
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import time
import traceback
import types
import unittest
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DIMS = [64, 128, 256, 512, 768, 896]
DATASETS = ["VOC2007", "HatefulMemes", "ImageNet-1K", "N24News", "SUN397"]
REFERENCE = {
    "ImageNet-1K": [44.5, 49.7, 52.3, 54.5, 55.2, 51.9],
    "N24News": [63.2, 65.9, 67.1, 67.3, 67.8, 65.5],
    "HatefulMemes": [56.8, 58.0, 58.8, 58.4, 58.4, 60.5],
    "VOC2007": [76.3, 79.1, 82.7, 82.4, 78.8, 73.6],
    "SUN397": [53.3, 60.2, 65.5, 66.0, 67.6, 64.4],
}
EPS = 1e-12
GAMMAS = [0, .25, .5, .75, 1, 1.25, 1.5, 2]
GROUPS = ["both_correct", "damage", "correction", "both_wrong"]


def clean(value):
    """Strict JSON: no NaN/Infinity, including nested NumPy scalars."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def digest(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checkpoint_hash(path):
    # Include every checkpoint artifact; base-model remote weights are a separately
    # recorded dependency, not silently claimed to be included in this hash.
    return digest({str(p.relative_to(path)): file_hash(p)
                   for p in sorted(Path(path).rglob("*")) if p.is_file()})


def revision():
    def git(*args):
        p = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    return {"commit": git("rev-parse", "HEAD"), "diff_hash": digest(git("diff")),
            "diagnostic_source_hash": file_hash(__file__)}


def quantiles(values):
    a = np.asarray([v for v in values if v is not None], dtype=np.float64)
    a = a[np.isfinite(a)]
    if not a.size:
        return {"count": 0, **{k: None for k in ("mean", "p05", "p25", "p50", "p75", "p95", "p99", "max")}}
    return {"count": a.size, "mean": a.mean(), "max": a.max(),
            **dict(zip(["p05", "p25", "p50", "p75", "p95", "p99"], np.quantile(a, [.05, .25, .5, .75, .95, .99])))}


class Writers:
    """Stream per-query/per-pair outputs, so score matrices never accumulate."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.handles, self.writers = {}, {}
        self.paths = defaultdict(set)

    def row(self, filename, row):
        if filename not in self.handles:
            self.handles[filename] = (self.directory / filename).open("w", newline="", encoding="utf-8")
            if filename.endswith(".csv"):
                self.writers[filename] = csv.DictWriter(self.handles[filename], fieldnames=list(row))
                self.writers[filename].writeheader()
        handle = self.handles[filename]
        if filename.endswith(".jsonl"):
            handle.write(json.dumps(clean(row), ensure_ascii=False, allow_nan=False) + "\n")
        else:
            self.writers[filename].writerow(clean(row))
        self.paths[filename[:3]].add(str(self.directory / filename))

    def json(self, filename, value):
        dump(self.directory / filename, value)
        self.paths[filename[:3]].add(str(self.directory / filename))

    def close(self):
        for handle in self.handles.values():
            handle.close()


class ScoreErrorStats:
    """Exact count/mean/max, bounded uniform reservoir for quantiles."""
    def __init__(self):
        self.count, self.total, self.maximum = 0, 0., 0.
        self.sample, self.priorities = np.empty(0), np.empty(0)
        self.rng = np.random.default_rng(2026)

    def append(self, values):
        a = np.asarray(values).ravel()
        a = a[np.isfinite(a)]
        if not a.size:
            return
        self.count += a.size
        self.total += a.astype(np.float64).sum()
        self.maximum = max(self.maximum, float(a.max()))
        values = np.concatenate((self.sample, a))
        priorities = np.concatenate((self.priorities, self.rng.random(len(a))))
        chosen = np.argpartition(priorities, -8192)[-8192:] if len(priorities) > 8192 else np.arange(len(priorities))
        self.sample, self.priorities = values[chosen], priorities[chosen]

    def report(self):
        return {**quantiles(self.sample), "count": self.count, "mean": self.total / self.count if self.count else None,
                "max": self.maximum if self.count else None, "quantile_sample_size": len(self.sample),
                "quantiles_exact": self.count <= len(self.sample), "sampling_seed": 2026}


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), EPS)


def score_chunks(q, candidates, dim, variant="projected_prefix", gamma=1., chunk=2048):
    """Scores in original pool order. q/candidates already in desired basis.

    S1 retains FULL original norms; gamma scales BOTH sides (gamma squared).
    Clamp individual vector norms, never the product of two norms.
    """
    if variant == "raw_full":
        dim = q.size
    qa, qb = q[:dim], q[dim:]
    qa2, qb2 = np.dot(qa, qa), np.dot(qb, qb)
    for start in range(0, len(candidates), chunk):
        c = candidates[start:start + chunk]
        a, b = c[:, :dim], c[:, dim:]
        A = a @ qa
        ca2, cb2 = np.sum(a * a, axis=1), np.sum(b * b, axis=1)
        if variant in {"projected_prefix", "S0", "raw_full"}:
            numerator, q2, c2 = A, qa2, ca2
        elif variant == "S1":
            numerator, q2, c2 = A, qa2 + qb2, ca2 + cb2
        elif variant in {"S2", "gamma"}:
            g2 = gamma * gamma if variant == "gamma" else 1.
            numerator, q2, c2 = A + g2 * (b @ qb), qa2 + g2 * qb2, ca2 + g2 * cb2
        elif variant == "tail_only":
            numerator, q2, c2 = b @ qb, qb2, cb2
        else:
            raise ValueError(f"Unknown score variant {variant}")
        yield start, numerator / max(float(np.sqrt(q2)), EPS) / np.maximum(np.sqrt(c2), EPS)


def scores(q, c, d, variant="projected_prefix", gamma=1., chunk=2048):
    return np.concatenate([s for _, s in score_chunks(q, c, d, variant, gamma, chunk)])


def positives(mapping):
    if "positive_positions" in mapping:
        return np.asarray(mapping["positive_positions"], dtype=np.int64)
    wanted = set(mapping["positive_rows"])
    return np.asarray([i for i, row in enumerate(mapping["candidate_rows"]) if row in wanted], dtype=np.int64)


def metric(s, mapping, topk=10):
    """Stable exact ties: first position in official pool, no tolerance in argmax."""
    if not np.isfinite(s).all():
        raise ValueError(f"Nonfinite valid score for {mapping['query_id']}")
    order = np.argsort(-s, kind="stable")
    pp = positives(mapping)
    mask = np.ones(len(s), dtype=bool)
    mask[pp] = False
    nn = np.flatnonzero(mask)
    p = int(pp[np.argmax(s[pp])]) if len(pp) else None
    n = int(nn[np.argmax(s[nn])]) if len(nn) else None
    gap = float(s[order[0]] - s[order[1]]) if len(order) > 1 else None
    return {"pred_position": int(order[0]), "pred_row": mapping["candidate_rows"][order[0]],
            "correct": bool(order[0] in pp), "best_positive_position": p,
            "best_positive_row": mapping["candidate_rows"][p] if p is not None else None,
            "hardest_negative_position": n,
            "hardest_negative_row": mapping["candidate_rows"][n] if n is not None else None,
            "margin": float(s[p] - s[n]) if p is not None and n is not None else None,
            "margin_missing_reason": "NO_POSITIVE" if p is None else "NO_NEGATIVE" if n is None else None,
            "positive_rank": int(np.flatnonzero(np.isin(order, pp))[0]) + 1 if p is not None else None,
            "topk_rows": [mapping["candidate_rows"][i] for i in order[:topk]],
            "topk_positions": order[:topk].tolist(), "top1_top2_gap": gap,
            "exact_tie": gap == 0 if gap is not None else False,
            "near_ties": {str(t): gap <= t if gap is not None else False for t in (1e-7, 1e-6, 1e-5)}}


def group(low, high):
    return GROUPS[(0 if low else 2) + (0 if high else 1)]


def paired_ci(low, high, mappings, samples=2000, seed=2026):
    changes = np.asarray(high, dtype=float) - np.asarray(low, dtype=float)
    clustered = all(m.get("group_id") is not None for m in mappings)
    units = defaultdict(list)
    for i, m in enumerate(mappings):
        units[str(m["group_id"]) if clustered else str(i)].append(i)
    sums = np.asarray([changes[ix].sum() for ix in units.values()])
    sizes = np.asarray([len(ix) for ix in units.values()])
    rng, draws = np.random.default_rng(seed), []
    for _ in range(samples):
        ix = rng.integers(0, len(units), len(units))
        draws.append(100 * sums[ix].sum() / sizes[ix].sum())
    return (*np.quantile(draws, [.025, .975]), "cluster_group_id" if clustered else "query_independence_assumed")


def pair_decomposition(q, c, d, paths, mapping, witness):
    m = metric(paths[2] if witness == "full_witness" else paths[0], mapping)
    p, n = m["best_positive_position"], m["hardest_negative_position"]
    if p is None or n is None:
        return None
    a, b = q[:d], q[d:]
    delta = [float(s[p] - s[n]) for s in paths]
    result = {"p_row": mapping["candidate_rows"][p], "n_row": mapping["candidate_rows"][n],
              "p_position": p, "n_position": n}
    qnorm0, qnorm1 = np.linalg.norm(a), np.linalg.norm(q)
    clamp = qnorm0 <= EPS or qnorm1 <= EPS
    for side, i in (("pos", p), ("neg", n)):
        cnorm0, cnorm1 = np.linalg.norm(c[i, :d]), np.linalg.norm(c[i])
        clamp |= cnorm0 <= EPS or cnorm1 <= EPS
        result.update({f"A_{side}": np.dot(a, c[i, :d]), f"B_{side}": np.dot(b, c[i, d:]),
                       f"D0_{side}": max(qnorm0, EPS) * max(cnorm0, EPS),
                       f"D1_{side}": max(qnorm1, EPS) * max(cnorm1, EPS)})
    en, et = delta[1] - delta[0], delta[2] - delta[1]
    tail_direct = result["B_pos"] / result["D1_pos"] - result["B_neg"] / result["D1_neg"]
    m0, m2 = metric(paths[0], mapping), metric(paths[2], mapping)
    result.update(delta0=delta[0], delta1=delta[1], delta2=delta[2], E_norm=en, E_tail=et,
                  identity_residual=delta[2] - delta[0] - en - et,
                  tail_direct_residual=tail_direct - et, clamp_active=bool(clamp),
                  p_changed_between_paths=m0["best_positive_position"] != m2["best_positive_position"],
                  n_changed_between_paths=m0["hardest_negative_position"] != m2["hardest_negative_position"])
    return result


def energy(x, d):
    an, bn = np.linalg.norm(x[..., :d], axis=-1), np.linalg.norm(x[..., d:], axis=-1)
    a2, b2 = an * an, bn * bn
    return {"r": b2 / np.maximum(a2 + b2, EPS), "t": b2 / np.maximum(a2, EPS),
            "norm_a": an, "norm_b": bn, "near_zero_a": an <= EPS, "near_zero_b": bn <= EPS}


class Cache:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        self.pools = json.loads((self.directory / "pools.json").read_text()) if (self.directory / "pools.json").exists() else {}
        self.mapping = [json.loads(line) for line in (self.directory / "mapping.jsonl").read_text().splitlines() if line]
        for m in self.mapping:
            if "candidate_rows" not in m:
                m["candidate_rows"] = self.pools[m["candidate_pool_id"]]
        def array(name, required=True):
            path = self.directory / (name + ".npy")
            return np.load(path, mmap_mode="r", allow_pickle=False) if required or path.exists() else None
        self.q, self.c, self.u = array("h_query"), array("h_candidate"), array("U")
        self.production_q, self.production_c = array("production_query", False), array("production_candidate", False)
        self.dims = self.manifest["mrl_dims"]

    def validate(self):
        errors, warnings = [], []
        counts = Counter()
        pool_sizes, positive_sizes, negative_sizes = Counter(), Counter(), Counter()
        if self.q.ndim != 2 or self.c.ndim != 2 or self.q.shape[1] != self.c.shape[1]:
            errors.append("Hidden state shapes do not match")
        D = self.q.shape[-1]
        if self.u.shape != (D, D):
            errors.append("U is not the full square checkpoint basis")
        if self.dims != DIMS:
            errors.append(f"Checkpoint mrl_dims={self.dims}; expected {DIMS}; no automatic correction")
        for name, a in (("h_query", self.q), ("h_candidate", self.c), ("U", self.u)):
            if a.dtype != np.float32:
                errors.append(f"{name} must be stored float32, got {a.dtype}")
            for start in range(0, len(a), 4096):
                counts[f"{name}_nonfinite"] += int((~np.isfinite(a[start:start + 4096])).sum())
            if counts[f"{name}_nonfinite"]:
                errors.append(f"{name} contains NaN/Inf")
        seen = set()
        for m in self.mapping:
            qid = m["query_id"]
            if qid in seen:
                errors.append(f"Duplicate query_id {qid}")
            seen.add(qid)
            pool = m["candidate_rows"]
            is_index = lambda v: isinstance(v, int) and not isinstance(v, bool)
            if not is_index(m["query_row"]) or not 0 <= m["query_row"] < len(self.q) or not pool or any(not is_index(r) or not 0 <= r < len(self.c) for r in pool):
                errors.append(f"Invalid query/pool indices {qid}")
                continue
            if any(not is_index(v) for v in m["positive_rows"]) or any(not is_index(v) for v in m.get("positive_positions", [])):
                errors.append(f"Noninteger positive indices {qid}")
                continue
            pp = positives(m)
            if any(not 0 <= p < len(pool) for p in pp) or len(set(pp.tolist())) != len(pp):
                errors.append(f"Invalid positive positions {qid}")
                continue
            if not set(m["positive_rows"]).issubset(pool):
                errors.append(f"Positive outside pool {qid}")
            if set(m["positive_rows"]) != {pool[i] for i in pp}:
                errors.append(f"Positive position/row disagreement {qid}")
            counts["queries_without_positive"] += int(not len(pp))
            counts["queries_without_negative"] += int(len(pp) == len(pool))
            counts["duplicate_candidate_row_occurrences"] += len(pool) - len(set(pool))
            counts["queries"] += 1
            counts["positive_annotations"] += len(pp)
            pool_sizes[len(pool)] += 1
            positive_sizes[len(pp)] += 1
            negative_sizes[len(pool) - len(pp)] += 1
        if not self.mapping:
            errors.append("Empty query mapping")
        files = self.manifest.get("file_hashes", {})
        for name, expected in files.items():
            if not (self.directory / name).exists() or file_hash(self.directory / name) != expected:
                errors.append(f"Cache artifact hash mismatch: {name}")
        if not files:
            errors.append("Cache has no artifact hashes/provenance; re-export raw h")
        if not self.manifest.get("raw_capture_verified"):
            errors.append("Cache does not verify raw h at pooling before normalization")
        if self.manifest.get("projection_convention") != "row_h_times_U":
            errors.append("Unknown projection convention")
        if counts["queries_without_positive"]:
            errors.append("Queries without positives; official metric requires explicit handling")
        if counts["duplicate_candidate_row_occurrences"]:
            warnings.append("Duplicate row occurrences retained; positional positives preserve official metric")
        if self.production_q is None or self.production_c is None:
            warnings.append("Production representations missing; official/reference comparison unavailable")
        else:
            for name, runtime, raw in (("production_query", self.production_q, self.q), ("production_candidate", self.production_c, self.c)):
                if runtime.shape != raw.shape:
                    errors.append(f"{name} shape differs from raw h")
                for start in range(0, len(runtime), 4096):
                    if not np.isfinite(runtime[start:start + 4096]).all():
                        errors.append(f"{name} has NaN/Inf")
                        break
        return {"validation_errors": errors, "warnings": warnings, "counts": counts,
                "pool_size_histogram": pool_sizes, "positive_count_histogram": positive_sizes,
                "negative_count_histogram": negative_sizes,
                "shapes": {"q": self.q.shape, "c": self.c.shape, "U": self.u.shape},
                "manifest": self.manifest}


def projection_mmap(source, u, path, chunk=2048):
    target = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=source.shape)
    for start in range(0, len(source), chunk):
        target[start:start + chunk] = source[start:start + chunk] @ u
    target.flush()
    return target


@contextlib.contextmanager
def reference_precision():
    import torch
    old_matmul, old_cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.autocast(device_type="cpu", enabled=False), torch.autocast(device_type="cuda", enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old_matmul, old_cudnn


def load_actual_model(args, ck_hash):
    """Reuse repository model/processor loader, audit actual adapter loading.

    A hook records LoRA parameters BEFORE the repository's merge_and_unload.
    This diagnostic does not override its loading or merge behavior.
    """
    import torch
    import peft.peft_model
    from peft import PeftModel
    from src.arguments import ModelArguments, DataArguments
    from src.model.model import MMEBModel
    from src.model.processor import load_processor
    ck = Path(args.checkpoint)
    required = ["flag_U.pt", "config.json", "adapter_config.json", "mm_projector.pth"]
    missing = [str(ck / name) for name in required if not (ck / name).is_file()]
    if not any((ck / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
        missing.append(str(ck / "adapter_model.{safetensors,bin}"))
    if missing:
        raise FileNotFoundError("Missing checkpoint inputs: " + ", ".join(missing))
    state = torch.load(ck / "flag_U.pt", map_location="cpu", weights_only=True)
    if state.get("mrl_dims", state["q_dims"]) != DIMS or state["full_dim"] != 896:
        raise ValueError(f"Checkpoint dimensions differ: q_dims={state['q_dims']}, mrl_dims={state.get('mrl_dims')}, full_dim={state['full_dim']}; no automatic correction")
    adapter = json.loads((ck / "adapter_config.json").read_text())
    ma = ModelArguments(model_name=args.model_name, model_backbone="llava_qwen2", checkpoint_path=str(ck),
                        lora=True, pooling="eos", normalize=True, lora_r=adapter.get("r", 64),
                        lora_alpha=adapter.get("lora_alpha", 64), processor_name=args.processor_name)
    da = DataArguments(dataset_name=args.dataset_name, dataset_split=args.split, image_dir=args.image_dir,
                       image_resolution="mid", tgt_prefix_mod=True)
    audit = {"checkpoint_hash": ck_hash, "basis_source": str(ck / "flag_U.pt"),
             "adapter_config": adapter, "load_state_results": [], "missing_inputs": []}
    original_set = peft.peft_model.set_peft_model_state_dict
    def record_set(*a, **kw):
        result = original_set(*a, **kw)
        audit["load_state_results"].append({"missing_keys": list(result.missing_keys),
                                             "unexpected_keys": list(result.unexpected_keys)})
        return result
    original_from = PeftModel.from_pretrained.__func__
    def record_from(cls, *a, **kw):
        result = original_from(cls, *a, **kw)
        names = [n for n, _ in result.named_parameters() if "lora_" in n]
        audit["lora_parameter_tensor_count"] = len(names)
        audit["lora_module_count"] = len({n.split(".lora_")[0] for n in names})
        audit["lora_names_hash"] = digest(names)
        return result
    from unittest.mock import patch
    with patch.object(peft.peft_model, "set_peft_model_state_dict", record_set), \
            patch.object(PeftModel, "from_pretrained", classmethod(record_from)):
        model = MMEBModel.load(ma, is_trainable=False).eval().to(args.device, dtype=torch.bfloat16)
    if not hasattr(model, "flag_projector"):
        raise ValueError("Checkpoint basis not loaded; no identity/random fallback allowed")
    if not torch.equal(model.flag_projector.U.cpu(), state["U"].float()):
        raise ValueError("Loaded basis differs from checkpoint")
    if not audit.get("lora_module_count"):
        raise ValueError("No LoRA layers were observed during checkpoint loading")
    bad = [r for r in audit["load_state_results"] if r["unexpected_keys"] or any("lora_" in k for k in r["missing_keys"])]
    if bad or not audit["load_state_results"]:
        raise ValueError(f"Adapter loading could not be verified: {bad or audit}")
    saved_projector = torch.load(ck / "mm_projector.pth", map_location="cpu", weights_only=True)
    loaded_projector = model.encoder.base_model.model.model.mm_projector.state_dict()
    if saved_projector.keys() != loaded_projector.keys() or any(
            not torch.equal(loaded_projector[k].cpu(), v.to(loaded_projector[k].dtype)) for k, v in saved_projector.items()):
        raise ValueError("Loaded mm_projector differs from checkpoint")
    audit["mm_projector_verified"] = True
    audit["encoder_dtypes"] = dict(Counter(str(p.dtype) for p in model.encoder.parameters()))
    audit["base_model_dependency"] = {"requested": args.model_name,
                                         "adapter_base": adapter.get("base_model_name_or_path"),
                                         "resolved_commit": getattr(model.encoder.config, "_commit_hash", None)}
    if adapter.get("base_model_name_or_path") != args.model_name:
        audit["missing_inputs"].append("Base model pairing requires review: CLI model_name differs from adapter base")
    # Use the same processor source as eval_mmeb.py; do not silently substitute a
    # checkpoint processor, which would be another evaluation experiment.
    processor = load_processor(ma, da)
    return model, processor, ma, da, audit


def export_cache(args, subset, model, processor, ma, da, load_audit, ck_hash):
    import torch
    from torch.utils.data import DataLoader
    from src.data.collator.eval_collator import EvalCollator
    from src.data.dataset.mmeb_dataset import EvalDataset
    from src.utils import to_device
    from eval_mmeb import POS_MOD_DICT
    directory = Path(args.output) / "cache" / subset
    pipeline_hash = digest({name: file_hash(ROOT / name) for name in (
        "src/model/model.py", "src/model/reference_model.py", "src/model/processor.py",
        "src/data/collator/eval_collator.py", "src/data/dataset/mmeb_dataset.py", "eval_mmeb.py")})
    if (directory / "manifest.json").exists():
        existing = Cache(directory)
        keys = {"checkpoint_hash": ck_hash, "split": args.split, "dataset_name": args.dataset_name,
                "seed": args.seed, "model_name": args.model_name, "processor_name": args.processor_name,
                "image_dir": str(Path(args.image_dir).resolve()), "pipeline_code_hash": pipeline_hash,
                "training_config_hash": file_hash(args.training_config) if args.training_config else None}
        mismatch = {k: [existing.manifest.get(k), v] for k, v in keys.items() if existing.manifest.get(k) != v}
        if mismatch:
            raise ValueError(f"Cache provenance mismatch: {mismatch}. Use a new --output directory")
        return existing
    directory.mkdir(parents=True, exist_ok=True)
    datasets, raw_paths, prod_paths = {}, {}, {}
    raw_checks = []
    original_pool = model._pooling
    captured = []
    def capture_pool(self, last_hidden_state, attention_mask, normalize=None):
        raw = original_pool(last_hidden_state, attention_mask, normalize=False)
        output = original_pool(last_hidden_state, attention_mask, normalize=normalize)
        captured.append(raw.detach())
        return output
    model._pooling = types.MethodType(capture_pool, model)
    try:
        for side, name in (("qry", "query"), ("tgt", "candidate")):
            ds = EvalDataset(da, ma, subset, f"{side}_text", f"{side}_img_path",
                             mod_instruction=POS_MOD_DICT.get(subset) if side == "tgt" else None)
            datasets[side] = ds
            if not len(ds):
                raise ValueError(f"Empty {subset}/{side}")
            shape = (len(ds), model.full_dim)
            raw_paths[name] = directory / f"h_{name}.npy"
            prod_paths[name] = directory / f"production_{name}.npy"
            raw = np.lib.format.open_memmap(raw_paths[name], mode="w+", dtype=np.float32, shape=shape)
            prod = np.lib.format.open_memmap(prod_paths[name], mode="w+", dtype=np.float32, shape=shape)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                collate_fn=EvalCollator(da, ma, processor))
            offset = 0
            for batch in loader:
                captured.clear()
                with torch.no_grad(), torch.autocast(device_type=torch.device(args.device).type, dtype=torch.bfloat16):
                    result = model(**{side: to_device(batch, torch.device(args.device))})[f"{side}_reps"]
                if len(captured) != 1:
                    raise ValueError(f"Expected exactly one pooling call, got {len(captured)}")
                h = captured[0]
                if offset == 0:
                    # Normalization autocast policy itself can promote BF16 to
                    # FP32. Reproduce the original context rather than assume
                    # normalization has the encoder's dtype.
                    with torch.autocast(device_type=torch.device(args.device).type, dtype=torch.bfloat16):
                        expected = torch.nn.functional.normalize(h, dim=-1) if model.normalize else h
                    error = (expected.float() - result.float()).abs().max().item()
                    raw_checks.append({"side": side, "pooled_runtime_max_error": error,
                                       "raw_dtype": str(h.dtype), "runtime_dtype": str(result.dtype)})
                    if error > 1e-6:
                        raise ValueError("Raw pooled tensor does not reproduce runtime representations")
                count = len(h)
                raw[offset:offset + count] = h.cpu().float().numpy()
                prod[offset:offset + count] = result.detach().cpu().float().numpy()
                offset += count
                print(f"[{subset}] encode {side}: {offset}/{len(ds)}", flush=True)
            raw.flush()
            prod.flush()
    finally:
        model._pooling = original_pool
    np.save(directory / "U.npy", model.flag_projector.U.cpu().numpy(), allow_pickle=False)
    qindex = {(p["text"], p["img_path"]): i for i, p in enumerate(datasets["qry"].paired_data)}
    cindex = {(p["text"], p["img_path"]): i for i, p in enumerate(datasets["tgt"].paired_data)}
    pools, mappings = {}, []
    with (directory / "mapping.jsonl").open("w") as stream:
        for i, row in enumerate(datasets["qry"].eval_data):
            keys = list(zip(row["tgt_text"], row["tgt_img_path"]))
            if not keys or len(row["tgt_text"]) != len(row["tgt_img_path"]):
                raise ValueError("Empty/misaligned official candidate lists")
            pool = [cindex[k] for k in keys]
            pool_id = digest(pool)
            pools[pool_id] = pool
            query_key = {"text": row["qry_text"], "img_path": row["qry_img_path"]}
            m = {"query_id": f"{subset}:{args.split}:{i}:{digest(query_key)}",
                 "query_row": qindex[(row["qry_text"], row["qry_img_path"])],
                 "candidate_pool_id": pool_id, "positive_rows": [pool[0]], "positive_positions": [0],
                 "candidate_order_source": "eval_mmeb.score_subset", "instruction_id": subset,
                 "group_id": str(row["qry_img_path"]) if row["qry_img_path"] else None}
            stream.write(json.dumps(m, ensure_ascii=False) + "\n")
            mappings.append(m)
    dump(directory / "pools.json", pools)
    dump(directory / "candidate_ids.json", [f"{subset}:target:{i}:{digest(pair)}"
                                            for i, pair in enumerate(datasets["tgt"].paired_data)])
    processor_config = {"class": type(processor).__name__, "tokenizer": getattr(processor.tokenizer, "init_kwargs", {}),
                        "image_processor": processor.image_processor.to_dict(), "data_arguments": vars(da)}
    # Any absent gain metadata remains unknown; today's source is not proof about
    # the objective used to create an older checkpoint.
    gain, training_meta = None, None
    if args.training_config:
        training_meta = json.loads(Path(args.training_config).read_text())
        gain = training_meta.get("loss_gain_weight")
    manifest = {"schema_version": "1.0", "run_id": args.run_id, "checkpoint_path": str(Path(args.checkpoint).resolve()),
                "checkpoint_hash": ck_hash, "checkpoint_step": None, "basis_hash": file_hash(directory / "U.npy"),
                "code_revision": revision(), "dataset": subset, "dataset_name": args.dataset_name,
                "split": args.split, "num_queries": len(mappings), "num_candidate_rows": len(datasets["tgt"]),
                "num_query_cache_rows": len(datasets["qry"]), "embedding_dim": model.full_dim,
                "mrl_dims": model.flag_projector.mrl_dims, "q_dims": model.flag_projector.q_dims,
                "mrl_weights": training_meta.get("flag_level_weights") if training_meta else None,
                "loss_gain_weight": gain, "encoder_dtype": load_audit["encoder_dtypes"],
                "reference_score_dtype": "float32", "production_projection_dtype": "float32_numpy",
                "production_normalization_dtype": {r["side"]: r["runtime_dtype"] for r in raw_checks},
                "production_matmul_dtype": "float32_numpy", "projection_convention": "row_h_times_U",
                "normalization_stage": "per_prefix_after_projection", "eps_norm": EPS,
                "candidate_protocol": "Per-query official ordered tgt_text/tgt_img_path; duplicate rows retained",
                "positive_protocol": "Official evaluator: candidate position 0 only; no inferred equivalence",
                "tie_policy": "First position in official pool; stable argsort; no score rounding",
                "pooling_description": "src/model/reference_model.py:_pooling(last_hidden_state, attention_mask, normalize=False); EOS/last valid token after multimodal expansion",
                "raw_capture_verified": True, "raw_capture_checks": raw_checks,
                "precision_limit": "FP32 cast of BF16 encoder states cannot restore discarded encoder precision",
                "instruction_config_hash": digest(POS_MOD_DICT), "processor_config_hash": digest(processor_config),
                "query_list_hash": digest([m["query_id"] for m in mappings]),
                "candidate_mapping_hash": digest({"pools": pools, "mapping": mappings,
                                                    "candidate_context": datasets["tgt"].paired_data}),
                "seed": args.seed, "candidate_chunk_size": args.candidate_chunk, "encoder_batch_size": args.batch_size,
                "model_name": args.model_name, "processor_name": args.processor_name,
                "image_dir": str(Path(args.image_dir).resolve()), "load_audit": load_audit,
                "pipeline_code_hash": pipeline_hash,
                "training_config_hash": file_hash(args.training_config) if args.training_config else None,
                "hf_dataset_fingerprint": getattr(datasets["qry"].eval_data, "_fingerprint", None),
                "original_run_verified": False,
                "missing_inputs": ["Original run query list/config/predictions not supplied; table reproduction unverified"]
                                  + (["Checkpoint gain weight metadata"] if gain is None else [])}
    manifest["file_hashes"] = {p.name: file_hash(p) for p in directory.iterdir() if p.suffix in {".npy", ".jsonl", ".json"} and p.name != "manifest.json"}
    dump(directory / "manifest.json", manifest)  # completion marker written last
    return Cache(directory)


def evaluate_cache(cache, args, writer):
    """T00-T06: one cached encoder pass; exact ragged-pool reference scores."""
    audit = cache.validate()
    writer.json("T00_input_audit.json", audit)
    statuses = {}
    subset = cache.manifest["dataset"]
    def status(test, state, reason):
        statuses[test] = {"status": state, "reason": reason, "num_queries": len(cache.mapping),
                          "input_hashes": {"checkpoint": cache.manifest["checkpoint_hash"],
                                           "basis": cache.manifest["basis_hash"],
                                           "mapping": cache.manifest["candidate_mapping_hash"]},
                          "output_paths": sorted(writer.paths[test])}
    if audit["validation_errors"]:
        status("T00", "FAIL", "; ".join(audit["validation_errors"]))
        for test in args.tests:
            if test in {"T01", "T02", "T03", "T04", "T05", "T06"}:
                status(test, "SKIPPED", "T00 validation failed; interpretation stopped")
        return statuses
    zq = projection_mmap(cache.q, cache.u, cache.directory / "reference_z_query.npy")
    zc = projection_mmap(cache.c, cache.u, cache.directory / "reference_z_candidate.npy")
    have_prod = cache.production_q is not None and cache.production_c is not None
    if have_prod:
        prod_zq = projection_mmap(cache.production_q, cache.u, cache.directory / "production_z_query.npy")
        prod_zc = projection_mmap(cache.production_c, cache.u, cache.directory / "production_z_candidate.npy")
    counts, official_counts, production_disagree = Counter(), Counter(), Counter()
    metrics = {d: [] for d in cache.dims}
    # Store only scalar/per-query metrics, never N*C matrices. Large pair outputs
    # are streamed through Writers.
    score_errors = defaultdict(ScoreErrorStats)
    norm_errors, norm_stats = {}, {}
    for side, raw, projected in (("query", cache.q, zq), ("candidate", cache.c, zc)):
        old, new = np.linalg.norm(raw, axis=1), np.linalg.norm(projected, axis=1)
        norm_errors[side] = quantiles(np.abs(new - old) / np.maximum(old, EPS))
        median = np.median(old[old > 0]) if (old > 0).any() else 0
        norm_stats[side] = {**quantiles(old), "clamp_active_count": int((old <= EPS).sum()),
                            "near_zero_warning_count": int((old <= median * 1e-6).sum())}
    gram = cache.u.T @ cache.u - np.eye(cache.u.shape[0], dtype=np.float32)
    singular = np.linalg.svd(cache.u, compute_uv=False)
    ef, emax = float(np.linalg.norm(gram) / math.sqrt(len(gram))), float(np.abs(gram).max())
    inv = {"orthogonality_frobenius_relative": ef, "orthogonality_max": emax,
           "singular_min": singular.min(), "singular_max": singular.max(),
           "norm_preservation_relative_error": norm_errors, "raw_norms": norm_stats,
           "reference_projection_normalization_matmul_dtype": "float32_numpy_cpu",
           "TF32": False, "autocast": False, "warning_threshold": 1e-4,
           "fp64_fallback": [], "safe_pairs_only_errors": {},
           "production_full_branch": "raw pooled runtime reps, not projected; exact repository get_pred"}
    disagreements, numerical_disagreements = 0, 0
    unsafe_queries, valid_pairs = 0, 0
    fp64_ids = set()
    for i, m in enumerate(cache.mapping):
        qi, ix = m["query_row"], m["candidate_rows"]
        hq, hc, q, c = cache.q[qi], cache.c[ix], zq[qi], zc[ix]
        by_dim, dim_scores, production_scores = {}, {}, {}
        for d in cache.dims:
            s = scores(q, c, d, chunk=args.candidate_chunk)
            result = metric(s, m, args.topk)
            result["degenerate"] = bool(np.linalg.norm(q[:d]) <= EPS or (np.linalg.norm(c[:, :d], axis=1) <= EPS).any())
            by_dim[d], dim_scores[d] = result, s
            counts[d] += result["correct"]
            metrics[d].append(result)
            if have_prod:
                # Call exactly the official get_pred, including its unsafeguarded
                # norm and first-position argmax, rather than imitate its metric.
                from evaluation.mmeb_baselines.eval_utils import get_pred
                oq = cache.production_q[qi] if d == q.size else prod_zq[qi, :d]
                oc = cache.production_c[ix] if d == q.size else prod_zc[ix, :d]
                oscore, opred = get_pred(oq, oc, normalization=True)
                production_scores[d] = oscore
                if not np.isfinite(oscore).all():
                    status("T00", "FAIL", "Official scorer produced nonfinite scores; degenerate norms require explanation")
                    return statuses
                official_counts[d] += int(opred in positives(m))
                production_disagree[d] += int(opred != result["pred_position"])
                writer.row("official_predictions.jsonl", {"query_id": m["query_id"], "dim": d,
                                                          "prediction_position": int(opred), "prediction_row": ix[opred]})
            writer.row("T02_predictions.jsonl", {"dataset": subset, "query_id": m["query_id"], "dim": d, **result})
        if "T01" in args.tests:
            raw = scores(hq, hc, q.size, "raw_full", chunk=args.candidate_chunk)
            full = dim_scores[q.size]
            normalized_q = normalize(hq) @ cache.u
            normalized_c = normalize(hc) @ cache.u
            checks = [("raw_full", raw, "projected_full", full),
                      ("projected_full", full, "prefix_scorer_D", scores(q, c, q.size, "S2", chunk=args.candidate_chunk))]
            if have_prod:
                prod_full, _ = get_pred(cache.production_q[qi], cache.production_c[ix], normalization=True)
                checks.append(("production_full", prod_full, "projected_full", full))
                checks.extend((f"production_prefix_{d}", production_scores[d], f"reference_prefix_{d}", dim_scores[d])
                              for d in cache.dims if d < q.size)
            for d in cache.dims:
                checks.append((f"normalize_h_first_{d}", scores(normalized_q, normalized_c, d, chunk=args.candidate_chunk),
                               f"raw_h_first_{d}", dim_scores[d]))
                # Deliberately wrong path is diagnostic evidence, never a metric.
                wrong = normalize(c)[:, :d] @ normalize(q)[:d]
                score_errors[f"slice_full_normalized_without_renormalizing_{d}_vs_correct"].append(np.abs(wrong - dim_scores[d]))
            safe = (np.linalg.norm(hq) > EPS) & (np.linalg.norm(q) > EPS) & \
                   (np.linalg.norm(hc, axis=1) > EPS) & (np.linalg.norm(c, axis=1) > EPS)
            unsafe_queries += int(not safe.all())
            valid_pairs += len(ix)
            for aname, a, bname, b in checks:
                error = np.abs(a - b)
                score_errors[f"{aname}_vs_{bname}"].append(error)
                if aname == "raw_full":
                    score_errors["safe_raw_vs_projected"].append(error[safe])
                ma, mb = metric(a, m), metric(b, m)
                if ma["pred_position"] != mb["pred_position"]:
                    maxerr = float(error.max())
                    tie = max(ma["top1_top2_gap"] or 0, mb["top1_top2_gap"] or 0) <= 4 * max(maxerr, 1e-12)
                    disagreements += 1
                    numerical_disagreements += int(tie)
                    writer.row("T01_prediction_disagreements.csv", {
                        "dataset": subset, "query_id": m["query_id"], "path_a": aname, "path_b": bname,
                        "pred_a": ma["pred_row"], "pred_b": mb["pred_row"],
                        "top1_top2_gap_a": ma["top1_top2_gap"], "top1_top2_gap_b": mb["top1_top2_gap"],
                        "max_score_error": maxerr, "near_tie_flag": tie})
                if (error.max() > 1e-4 or ma["pred_position"] != mb["pred_position"]) and len(fp64_ids) < 32 and i not in fp64_ids:
                    h64, c64, u64 = hq.astype(np.float64), hc.astype(np.float64), cache.u.astype(np.float64)
                    s64raw = scores(h64, c64, q.size, chunk=args.candidate_chunk)
                    s64z = scores(h64 @ u64, c64 @ u64, q.size, chunk=args.candidate_chunk)
                    fp64_ids.add(i)
                    inv["fp64_fallback"].append({"query_id": m["query_id"], "max_raw_rotated_error": np.abs(s64raw - s64z).max(),
                                                 "trigger_path": f"{aname}_vs_{bname}",
                                                 "max_full_scorer_algebra_error": np.abs(s64z - scores(h64 @ u64, c64 @ u64, q.size, "S2", chunk=args.candidate_chunk)).max(),
                                                 "max_normalize_before_after_prefix_errors": {
                                                     dim: np.abs(scores(normalize(h64) @ u64, normalize(c64) @ u64, dim, chunk=args.candidate_chunk) -
                                                                 scores(h64 @ u64, c64 @ u64, dim, chunk=args.candidate_chunk)).max() for dim in cache.dims},
                                                 "same_cache_no_reencode": True})
        if (i + 1) % 100 == 0:
            print(f"[{subset}] reference T00/T01 {i + 1}/{len(cache.mapping)}", flush=True)
    n = len(cache.mapping)
    for d in cache.dims:
        table = REFERENCE.get(subset, [None] * len(DIMS))[DIMS.index(d)]
        official_acc = official_counts[d] / n if have_prod else None
        writer.row("T00_reproduction.csv", {"dataset": subset, "dim": d, "n_queries": n,
                   "n_correct": official_counts[d] if have_prod else None, "acc": official_acc,
                   "reference_acc": counts[d] / n, "reference_table_acc": table / 100 if table is not None else None,
                   "delta_vs_table_pp": 100 * official_acc - table if official_acc is not None and table is not None else None,
                   "official_reference_disagreement_count": production_disagree[d] if have_prod else None,
                   "matches_table_rounded": round(100 * official_acc, 1) == table if official_acc is not None and table is not None else None})
    notes = (f"# {subset}: pipeline notes\n\n{cache.manifest['pooling_description']}\n\n"
             "Raw h is captured before normalization in the SAME production forward. The output is checked against configured pooling normalization.\n\n"
             "Basis is loaded from flag_U.pt without fallback. Official candidate pools/order, last-key cache lookup and position-0 positive semantics match eval_mmeb.py. "
             "Repeated image-only cache rows are retained. No candidate content-based positive expansion is applied.\n\n"
             "The official full branch uses unrotated runtime reps; the reference full branch uses raw h @ U. "
             "Production pooling may normalize in BF16 before conversion to FP32. Reference uses FP32 projection and normalization, with TF32/autocast disabled. "
             "Casting BF16 hidden states to FP32 cannot recover encoder precision. This command does not validate the historical table without original run evidence.\n")
    (writer.directory / "T00_pipeline_notes.md").write_text(notes)
    writer.paths["T00"].add(str(writer.directory / "T00_pipeline_notes.md"))
    status("T00", "INCONCLUSIVE", "NOT_REPRODUCED: original checkpoint-to-table/run/query-list identity unverified; structurally valid new diagnostic experiment")
    inv["num_valid_pairs"] = valid_pairs
    inv["unsafe_norm_queries"] = unsafe_queries
    inv["score_errors"] = {k: v.report() for k, v in score_errors.items()}
    inv["prediction_disagreements"] = disagreements
    inv["numerical_tie_disagreements"] = numerical_disagreements
    inv["near_ties_by_dim"] = {d: {str(t): sum(m["near_ties"][str(t)] for m in metrics[d]) for t in (1e-7, 1e-6, 1e-5)} for d in cache.dims}
    inv["exact_ties_by_dim"] = {d: sum(m["exact_tie"] for m in metrics[d]) for d in cache.dims}
    if "T01" in args.tests:
        raw_error = inv["score_errors"]["safe_raw_vs_projected"]["max"] or 0
        algebra_error = inv["score_errors"]["projected_full_vs_prefix_scorer_D"]["max"] or 0
        # 1e-4 is a starting WARNING threshold. The loader's 1e-3
        # orthogonality limit is a separate structural basis check.
        basis_warning = ef > 1e-4 or emax > 1e-4
        production_warning = any((v["max"] or 0) > 1e-4 for k, v in inv["score_errors"].items() if k.startswith("production_"))
        failed = ef > 1e-3 or emax > 1e-3 or raw_error > 1e-4 or algebra_error > 1e-5 or disagreements > numerical_disagreements
        state = "FAIL" if failed else "WARN" if disagreements or unsafe_queries or basis_warning or production_warning else "PASS"
        reason = "FAIL_BASIS/FAIL_PIPELINE: inspect measured errors and FP64 fallback" if failed else \
                 "Numerical tie, production precision difference, basis warning, or degenerate norms; inspect measured errors" if state == "WARN" else "Reference full invariance and implementation paths agree within recorded thresholds"
        writer.json("T01_invariance.json", inv)
        status("T01", state, reason)
        if failed:
            for test in args.tests:
                if test in {"T02", "T03", "T04", "T05", "T06"}:
                    status(test, "SKIPPED", "T01 failed; model interpretation stopped")
            return statuses
    if "T02" in args.tests:
        for d in cache.dims:
            writer.row("T02_accuracy.csv", {"dataset": subset, "dim": d, "n": n, "n_correct": counts[d], "acc": counts[d] / n})
        transitions = [(256, 896), (512, 896), (768, 896), (256, 512), (512, 768)]
        for low, high in transitions:
            ml, mh = metrics[low], metrics[high]
            buckets = Counter(group(a["correct"], b["correct"]) for a, b in zip(ml, mh))
            lo, hi, unit = paired_ci([m["correct"] for m in ml], [m["correct"] for m in mh], cache.mapping, args.bootstrap_samples, args.seed)
            delta = 100 * (counts[high] - counts[low]) / n
            residual = delta - 100 * (buckets["correction"] - buckets["damage"]) / n
            writer.row("T02_transitions.csv", {"dataset": subset, "low_dim": low, "high_dim": high, "n": n,
                       **{g: buckets[g] for g in GROUPS}, "damage_rate": buckets["damage"] / n,
                       "correction_rate": buckets["correction"] / n, "delta_acc_pp": delta,
                       "ci95_low_pp": lo, "ci95_high_pp": hi, "bootstrap_unit": unit, "identity_residual_pp": residual})
            summaries = defaultdict(list)
            for m, a, b in zip(cache.mapping, ml, mh):
                g = group(a["correct"], b["correct"])
                writer.row("T02_per_query.jsonl", {"dataset": subset, "query_id": m["query_id"],
                           "low_dim": low, "high_dim": high, "group": g, "low": a, "high": b})
                if a["margin"] is None or b["margin"] is None:
                    continue
                for feature, value in (("margin_low", a["margin"]), ("margin_high", b["margin"]), ("margin_delta", b["margin"] - a["margin"])):
                    summaries[(g, feature, "all", "all")].append(value)
                edges = [0, .001, .005, .01, .05, math.inf]
                slot = next(j for j in range(5) if edges[j] <= abs(a["margin"]) < edges[j + 1])
                interval = f"[{edges[slot]},{edges[slot + 1]})"
                summaries[(g, "margin_low", interval, "negative" if a["margin"] < 0 else "zero" if a["margin"] == 0 else "positive")].append(a["margin"])
            for (g, feature, interval, sign), values in summaries.items():
                writer.row("T02_margin_summary.csv", {"dataset": subset, "low_dim": low, "high_dim": high,
                           "group": g, "feature": feature, "boundary_bin": interval, "margin_sign": sign, **quantiles(values)})
        status("T02", "PASS", "Exact per-query transitions, signed margins and paired confidence intervals exported")
    if any(test in args.tests for test in ("T03", "T04", "T05")):
        try:
            analyze_tail(cache, zq, zc, metrics, args, writer)
            for test in ("T03", "T04", "T05"):
                if test in args.tests:
                    status(test, "PASS", "Computed from the same validated raw cache; test-split sweeps are diagnostic only")
        except Exception as exc:
            writer.json("T03_tail_exception.json", {"exception": repr(exc), "traceback": traceback.format_exc()})
            for test in ("T03", "T04", "T05"):
                if test in args.tests:
                    status(test, "FAIL", f"Tail diagnostic failed: {exc}; inspect T03_tail_exception.json")
    if "T06" in args.tests:
        try:
            basis_comparison(cache, args, writer)
            status("T06", "PASS", "Learned, identity, and three fixed random orthogonal inference bases compared")
        except Exception as exc:
            writer.json("T06_exception.json", {"exception": repr(exc), "traceback": traceback.format_exc()})
            status("T06", "FAIL", f"Basis comparison failed: {exc}")
    return statuses


def analyze_tail(cache, zq, zc, metrics, args, writer):
    subset, n = cache.manifest["dataset"], len(cache.mapping)
    rng = np.random.default_rng(args.seed)
    for d in (256, 512, 768):
        path_counts, patterns, gamma_counts, gamma_damage, gamma_correction = Counter(), Counter(), Counter(), Counter(), Counter()
        tail_counts, tail_n, tail_degenerate = Counter(), Counter(), Counter()
        decomposition = defaultdict(list)
        features = defaultdict(list)
        examples = defaultdict(list)
        path_transitions = Counter()
        if "T05" in args.tests:
            for side, vectors in (("query_population", zq), ("candidate_population_unique_cache_rows", zc)):
                e = energy(vectors, d)
                for feature, values in e.items():
                    writer.row("T05_energy_summary.csv", {"dataset": subset, "d": d, "query_or_candidate": side,
                               "conditioning_group": "all", "feature": feature, **quantiles(values)})
        max_residual, max_tail_residual = 0., 0.
        for i, m in enumerate(cache.mapping):
            q, c = zq[m["query_row"]], zc[m["candidate_rows"]]
            paths = [scores(q, c, d, variant, chunk=args.candidate_chunk) for variant in ("S0", "S1", "S2")]
            mm = [metric(s, m, args.topk) for s in paths]
            g = group(mm[0]["correct"], mm[2]["correct"])
            patterns["".join(str(int(a["correct"])) for a in mm)] += 1
            for r, a in enumerate(mm):
                path_counts[r] += a["correct"]
            for name, a, b in (("S0_to_S1", mm[0], mm[1]), ("S1_to_S2", mm[1], mm[2])):
                path_transitions[(name, group(a["correct"], b["correct"]))] += 1
            if "T04" in args.tests:
                for gamma in GAMMAS:
                    sg = scores(q, c, d, "gamma", gamma, args.candidate_chunk)
                    mg = metric(sg, m)
                    gamma_counts[gamma] += mg["correct"]
                    gamma_damage[gamma] += mm[0]["correct"] and not mg["correct"]
                    gamma_correction[gamma] += not mm[0]["correct"] and mg["correct"]
                    if gamma in (0, 1):
                        expected = paths[0 if gamma == 0 else 2]
                        if not np.allclose(sg, expected, atol=1e-6, rtol=1e-6):
                            raise AssertionError(f"Gamma endpoint mismatch {m['query_id']} d={d}")
            if "T05" in args.tests:
                sb = scores(q, c, d, "tail_only", chunk=args.candidate_chunk)
                mt = metric(sb, m)
                qe, ce = energy(q, d), energy(c, d)
                degenerate = bool(qe["near_zero_b"] or np.any(ce["near_zero_b"]))
                for cond in ("all", g):
                    tail_n[cond] += 1
                    tail_counts[cond] += mt["correct"]
                    tail_degenerate[cond] += degenerate
                for f, value in qe.items():
                    features[(g, "query", f)].append(value)
            if "T03" in args.tests or "T05" in args.tests:
                for witness in ("full_witness", "prefix_witness"):
                    dec = pair_decomposition(q, c, d, paths, m, witness)
                    if dec is None:
                        continue
                    row = {"dataset": subset, "query_id": m["query_id"], "d": d, "group": g,
                           "witness_type": witness, **dec}
                    if "T03" in args.tests:
                        writer.row("T03_pair_decomposition.csv", row)
                        for f in ("E_norm", "E_tail"):
                            decomposition[(g, witness, f)].append(dec[f])
                        max_residual = max(max_residual, abs(dec["identity_residual"]))
                        max_tail_residual = max(max_tail_residual, abs(dec["tail_direct_residual"]))
                        if d == 768 and g in {"damage", "correction"} and witness == "full_witness":
                            # Reservoir sample plus largest margin drops; never only outliers.
                            examples[g].append((rng.random(), dec["delta2"] - dec["delta0"], row))
                    if "T05" in args.tests:
                        p, neg = dec["p_position"], dec["n_position"]
                        wf = {"rq": qe["r"], "rpos": ce["r"][p], "rneg": ce["r"][neg],
                              "energy_difference": ce["r"][p] - ce["r"][neg],
                              "cos_tail_pos": sb[p], "cos_tail_neg": sb[neg], "tail_pair_margin": sb[p] - sb[neg],
                              "near_zero_query": bool(qe["near_zero_b"]), "near_zero_positive": bool(ce["near_zero_b"][p]),
                              "near_zero_negative": bool(ce["near_zero_b"][neg])}
                        writer.row("T05_witness_features.csv", {"dataset": subset, "query_id": m["query_id"], "d": d,
                                   "group": g, "witness_type": witness, **wf})
                        for f, value in wf.items():
                            features[(g, f"pair_occurrences_{witness}", f)].append(value)
            if (i + 1) % 100 == 0:
                print(f"[{subset}] tail d={d}: {i + 1}/{n}", flush=True)
        if "T03" in args.tests:
            for r, variant in enumerate(("S0", "S1", "S2")):
                writer.row("T03_score_paths.csv", {"dataset": subset, "d": d, "score_variant": variant, "n": n,
                           "n_correct": path_counts[r], "acc": path_counts[r] / n,
                           "delta_vs_S0_pp": 100 * (path_counts[r] - path_counts[0]) / n})
            for pattern in (f"{k:03b}" for k in range(8)):
                writer.row("T03_path_patterns.csv", {"dataset": subset, "d": d, "pattern": pattern, "count": patterns[pattern]})
            for transition in ("S0_to_S1", "S1_to_S2"):
                writer.row("T03_path_transitions.csv", {"dataset": subset, "d": d, "transition": transition,
                           **{g: path_transitions[(transition, g)] for g in GROUPS}})
            for (g, witness, feature), values in decomposition.items():
                writer.row("T03_decomposition_summary.csv", {"dataset": subset, "d": d, "group": g,
                           "witness_type": witness, "feature": feature, "negative_fraction": np.mean(np.asarray(values) < 0), **quantiles(values)})
            writer.json(f"T03_identity_check_{d}.json", {"max_identity_residual": max_residual,
                         "max_tail_direct_residual": max_tail_residual})
            if max_residual > 1e-6 or max_tail_residual > 2e-5:
                raise AssertionError("Fixed-witness score decomposition failed")
            for g, entries in examples.items():
                limit = 20 if g == "damage" else 10
                chosen = sorted(entries, key=lambda x: x[1])[:limit // 2]
                chosen_ids = {x[2]["query_id"] for x in chosen}
                chosen += sorted((x for x in entries if x[2]["query_id"] not in chosen_ids), key=lambda x: x[0])[:limit - len(chosen)]
                for _, _, row in chosen:
                    writer.row("T03_examples.jsonl", row)
        if "T04" in args.tests:
            for gamma in GAMMAS:
                writer.row("T04_gamma_sweep.csv", {"dataset": subset, "d": d, "gamma": gamma, "n": n,
                           "n_correct": gamma_counts[gamma], "acc": gamma_counts[gamma] / n,
                           "delta_vs_prefix_pp": 100 * (gamma_counts[gamma] - path_counts[0]) / n,
                           "delta_vs_full_pp": 100 * (gamma_counts[gamma] - path_counts[2]) / n,
                           "damage_count_vs_prefix": gamma_damage[gamma], "correction_count_vs_prefix": gamma_correction[gamma]})
        if "T05" in args.tests:
            for g in ("all", *GROUPS):
                writer.row("T05_tail_accuracy.csv", {"dataset": subset, "d": d, "group": g, "n": tail_n[g],
                           "n_correct": tail_counts[g], "acc": tail_counts[g] / tail_n[g] if tail_n[g] else None,
                           "degenerate_count": tail_degenerate[g]})
            for (g, side, feature), values in features.items():
                writer.row("T05_energy_summary.csv", {"dataset": subset, "d": d, "query_or_candidate": side,
                           "conditioning_group": g, "feature": feature, **quantiles(values)})


def orthogonal_basis(D, seed):
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((D, D)))
    sign = np.sign(np.diag(r))
    return (q * np.where(sign == 0, 1, sign)).astype(np.float32)


def basis_comparison(cache, args, writer):
    D = cache.q.shape[1]
    random_acc = defaultdict(list)
    for kind, seed in (("learned", None), ("identity", None), *(("random_orthogonal", s) for s in (0, 1, 2))):
        u = cache.u if kind == "learned" else np.eye(D, dtype=np.float32) if kind == "identity" else orthogonal_basis(D, seed)
        q = projection_mmap(cache.q, u, cache.directory / "counterfactual_query.npy")
        c = projection_mmap(cache.c, u, cache.directory / "counterfactual_candidate.npy")
        error, counts = 0., Counter()
        for m in cache.mapping:
            qi, ix = m["query_row"], m["candidate_rows"]
            for d in (256, 512, 768, D):
                s = scores(q[qi], c[ix], d, chunk=args.candidate_chunk)
                counts[d] += metric(s, m)["correct"]
                if d == D:
                    raw = scores(cache.q[qi], cache.c[ix], D, chunk=args.candidate_chunk)
                    error = max(error, float(np.abs(s - raw).max()))
        orth = float(np.linalg.norm(u.T @ u - np.eye(D)) / math.sqrt(D))
        for d, correct in counts.items():
            acc = correct / len(cache.mapping)
            writer.row("T06_basis_comparison.csv", {"dataset": cache.manifest["dataset"], "basis_kind": kind, "seed": seed,
                       "dim": d, "acc": acc, "full_score_error_vs_raw": error,
                       "orthogonality_error": orth, "basis_hash": hashlib.sha256(u.tobytes()).hexdigest()})
            if kind == "random_orthogonal":
                random_acc[d].append(acc)
        if error > 1e-4:
            raise AssertionError(f"T06 full invariance failed: {kind} seed={seed} error={error}")
    for d, values in random_acc.items():
        writer.row("T06_random_summary.csv", {"dataset": cache.manifest["dataset"], "dim": d,
                   "random_seeds": 3, "mean_acc": np.mean(values), "std_acc": np.std(values),
                   "interpretation": "random_subspace_variation_not_training_seed_variation"})


def training_probes(args, writer, ck_hash):
    """Opt-in actual loss/FlagStep probes on fresh model state, no encoder step.

    Config is a flat JSON of original training arguments plus loss_gain_weight,
    original_world_size, and (when available) trainable_parameter_names. Missing
    config is SKIPPED, never filled in using today's CLI defaults.
    """
    selected = set(args.tests) & {"T07", "T08", "T09"}
    status = {}
    def mark(test, state, reason):
        status[test] = {"status": state, "reason": reason, "num_queries": None,
                        "input_hashes": {"checkpoint": ck_hash,
                                         "training_config": file_hash(args.training_config) if args.training_config else None},
                        "output_paths": sorted(writer.paths[test])}
    if not selected:
        return status
    if not args.training_config or not args.train_image_dir:
        for test in selected:
            mark(test, "SKIPPED", "Missing --training-config (original flat JSON) and/or --train-image-dir")
        return status
    config = json.loads(Path(args.training_config).read_text())
    needed = ["temperature", "nested_dims", "flag_level_weights", "flag_lr", "flag_eps", "flag_retraction",
              "flag_lambda_neg", "flag_use_neg_mask", "flag_far_mode", "flag_far_tau", "loss_gain_weight",
              "dataset_name", "dataset_split", "image_resolution", "per_device_train_batch_size", "original_world_size"]
    missing = [key for key in needed if key not in config]
    if missing or config.get("loss_gain_weight") != 0:
        for test in selected:
            mark(test, "SKIPPED", f"Missing original config inputs {missing}; gain must be verified zero (got {config.get('loss_gain_weight')})")
        return status
    import copy
    import torch
    from torch.nn import functional as F
    from src.arguments import ModelArguments, DataArguments
    from src.model.model import MMEBModel
    from src.model.processor import load_processor
    from src.MRL.flag.flag_projector import FlagProjector
    from src.MRL.flag.encoder_loss import encoder_losses
    from src.MRL.flag.pairs import build_pair_weights
    from src.MRL.flag.flag_math import flag_loss_and_grad
    from src.MRL.flag_mrl import FlagMRLLoss
    from src.utils import to_device
    from prepare_one_model_training import TrainOneModelDataset, TrainOneModelCollator
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        raise ValueError("Launch training diagnostics as a single process; DDP-global graph recreation is not implemented")
    ma = ModelArguments(model_name=args.model_name, checkpoint_path=args.checkpoint, lora=True,
                        load_pretrained_lora=True, pooling=config.get("pooling", "eos"), normalize=True,
                        temperature=config["temperature"])
    # build(load_pretrained_lora=True) is the repository's UNMERGED adapter path;
    # load() merges/unloads LoRA and cannot be used for parameter gradient audits.
    model = MMEBModel.build(ma).to(args.device).eval()
    dtype_file = Path(args.checkpoint) / "encoder_dtypes.json"
    if dtype_file.exists():
        dtype_map = {"torch.float32": torch.float32, "torch.bfloat16": torch.bfloat16, "torch.float16": torch.float16}
        saved = json.loads(dtype_file.read_text())
        for name, p in model.encoder.named_parameters():
            if name in saved:
                p.data = p.data.to(dtype_map[saved[name]])
    if "trainable_parameter_names" in config:
        wanted = set(config["trainable_parameter_names"])
        actual = {n for n, _ in model.named_parameters()}
        if not wanted.issubset(actual):
            raise ValueError(f"Original trainable parameter names missing: {sorted(wanted - actual)[:20]}")
        for name, p in model.named_parameters():
            p.requires_grad_(name in wanted)
    state = torch.load(Path(args.checkpoint) / "flag_U.pt", map_location="cpu", weights_only=True)
    projector = FlagProjector(state["full_dim"], state["q_dims"], init="identity", lr=config["flag_lr"],
                              eps=config["flag_eps"], retraction=config["flag_retraction"],
                              lambda_neg=config["flag_lambda_neg"]).to(args.device)
    projector.load(Path(args.checkpoint) / "flag_U.pt")
    dims = projector.mrl_dims
    if dims != config["nested_dims"]:
        raise ValueError("Original training nested_dims do not match checkpoint; no automatic correction")
    loss_args = types.SimpleNamespace(**{**config, "flag_update_every": config.get("flag_update_every", 1), "kd_loss_type": "flag_mrl"})
    criterion = FlagMRLLoss(loss_args).eval()
    weights = criterion.level_weights(len(dims))
    model.flag_projector = projector
    processor = load_processor(ma)
    trainables = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainables and "T08" in selected:
        raise ValueError("No trainable parameters in unmerged checkpoint clone")
    surrogate = config["original_world_size"] != 1 or "trainable_parameter_names" not in config
    protocol = "single_GPU_surrogate: candidate/global gradient pool or original parameter list unverified" if surrogate else "single_process_original_pool_semantics"
    writer.json("T07_loss_config.json", {"config": config, "mrl_dims": dims, "q_dims": projector.q_dims,
                "weights": weights, "sum_weights": sum(weights), "full_weight": weights[dims.index(state["full_dim"])],
                "loss_gain_weight": 0, "temperature": config["temperature"], "reduction": "weighted sum of mean CE",
                "trainable_parameter_names": [n for n, _ in trainables], "gradient_protocol": protocol,
                "gather": "single process; rank0 local order == global order for this probe",
                "multi_positive_annotation": "not provided by TrainOneModelDataset; list entries are aligned query-positive pairs",
                "mode": "model.eval with autograd; U frozen; no optimizer.step"})
    batch_count, full_conflicts, cos_conflicts, residuals = 0, 0, Counter(), []
    full_loss_errors, u_full_errors, retraction_paths = [], [], []
    examples_count = 0
    n_audited, n_mask_differences = 0, 0
    for subset in args.datasets:
        if batch_count >= args.gradient_probe_max_batches:
            break
        train_subset = "ImageNet_1K" if subset == "ImageNet-1K" else subset
        da = DataArguments(dataset_name=config["dataset_name"], dataset_split=config["dataset_split"],
                           subset_name=[train_subset], image_dir=args.train_image_dir,
                           image_resolution=config["image_resolution"], max_len=config.get("max_len"))
        ds = TrainOneModelDataset(da, ma)
        # Fixed first valid batch per subset, explicit IDs persisted. This is a
        # deterministic probe, not a recreation of original shuffled batch order.
        indices = list(range(min(len(ds), config["per_device_train_batch_size"])))
        examples = [ds[i] for i in indices]
        batch = to_device(TrainOneModelCollator(processor, ma, da, loss_args)(examples), torch.device(args.device))
        qids, pids = [], []
        for ix, example in zip(indices, examples):
            qids.extend(f"{train_subset}:filtered:{ix}:pair:{j}" for j in range(len(example["query_text"])))
            pids.extend(f"{train_subset}:{pid}" for pid in example["pos_id"])
        with torch.set_grad_enabled("T08" in selected), torch.autocast(device_type=torch.device(args.device).type, dtype=torch.bfloat16):
            q = model.encode_input(batch["qry"], normalize=False)[0]
            c = model.encode_input(batch["pos"], normalize=False)[0]
        if len(q) != len(qids) or len(c) != len(pids) or len(q) != len(c):
            raise ValueError("Training collator order/shape does not match aligned query/positive source metadata")
        target = torch.arange(len(q), device=q.device, dtype=torch.long)
        qn, cn = F.normalize(q.detach().float(), dim=-1), F.normalize(c.detach().float(), dim=-1)
        mask_ids = batch["pos_id"] if config["flag_use_neg_mask"] else None
        near, far, valid = build_pair_weights(qn, cn, target, mask_ids, config["flag_far_mode"], config["flag_far_tau"])
        batch_id = f"{subset}:{batch_count}"
        writer.row("T07_batch_order.jsonl", {"batch_id": batch_id, "subset": subset, "filtered_indices": indices,
                   "query_ids_before_gather": qids, "query_ids_after_gather": qids,
                   "candidate_ids_before_gather": pids, "candidate_ids_after_gather": pids,
                   "target_indices": target.cpu().tolist(), "protocol": protocol})
        for i, qid in enumerate(qids):
            # Identical content hashes are only possible equivalence, without
            # stronger task annotations; never declare a false negative from text.
            possible = sum(pid == pids[i] for j, pid in enumerate(pids) if j != i)
            flag_neg = far[i] > 0
            mrl_neg = torch.ones(len(c), device=q.device, dtype=torch.bool)
            mrl_neg[i] = False
            difference = int((flag_neg != mrl_neg).sum())
            n_audited += 1
            n_mask_differences += int(difference > 0)
            writer.row("T07_supervision_audit.csv", {"subset": subset, "batch_id": batch_id, "query_id": qid,
                       "pos_index_valid": int(target[i]) == i, "positive_in_pool": True,
                       "positive_annotation_count": 1, "confirmed_equivalent_count": 0,
                       "possible_equivalent_count": possible, "flag_neg_count": int(flag_neg.sum()),
                       "mrl_neg_count": int(mrl_neg.sum()), "mask_difference_count": difference})
            if (possible or difference) and examples_count < 20:
                writer.row("T07_examples.jsonl", {"subset": subset, "batch_id": batch_id, "query_id": qid,
                           "positive_id": pids[i], "reason": "possible repeated target content and/or flag/MRL mask difference; annotation equivalence unverified"})
                examples_count += 1
        with torch.autocast(device_type=q.device.type, enabled=False):
            zq, zc = q.float() @ projector.U.detach().clone(), c.float() @ projector.U.detach().clone()
            actual_total, actual_metrics = encoder_losses(model, zq, zc, target, dims, weights, config["temperature"])
        full_loss = float(actual_metrics[f"loss_dim_{dims[-1]}"])
        direct_ce = float(F.cross_entropy(F.normalize(zq, dim=-1) @ F.normalize(zc, dim=-1).T / config["temperature"], target).detach())
        full_loss_errors.append(abs(full_loss - direct_ce))
        writer.row("T07_full_loss_check.csv", {"batch_id": batch_id, "full_loss": full_loss,
                   "direct_full_ce": direct_ce,
                   "total_requires_grad": actual_total.requires_grad, "full_weight": weights[-1]})
        if "T08" in selected:
            probe_gradients(model, zq, zc, target, dims, weights, config["temperature"], actual_total,
                            trainables, batch_id, subset, writer, residuals, cos_conflicts)
            # The full dot is collected by probe_gradients in a small side list.
            full_conflicts += int(probe_gradients.last_full_dot < 0)
        if "T09" in selected:
            old, new = copy.deepcopy(projector), copy.deepcopy(projector)
            def objective(u):
                return float(flag_loss_and_grad(u.U, u.a_col, (qn, cn, near), (qn, cn, far), u.eps, u.lambda_neg)[0]) if valid else None
            before = objective(old)
            import src.MRL.flag.flag_math as flag_math
            from unittest.mock import patch
            with patch.object(flag_math, "_polar_svd", wraps=flag_math._polar_svd) as svd_call:
                update_metrics = new.step((qn, cn, near), (qn, cn, far))
            path = "skipped" if update_metrics["flag_skipped"] else "svd" if svd_call.called else "newton_schulz"
            retraction_paths.append({"batch": batch_id, "configured": config["flag_retraction"], "actual": path})
            after = objective(new)
            ub, ua = old.U.cpu().numpy(), new.U.cpu().numpy()
            ob = float(np.linalg.norm(ub.T @ ub - np.eye(len(ub))) / math.sqrt(len(ub)))
            oa = float(np.linalg.norm(ua.T @ ua - np.eye(len(ua))) / math.sqrt(len(ua)))
            hbq, hbc = q.detach().cpu().float().numpy(), c.detach().cpu().float().numpy()
            full_error = float(np.abs(normalize(hbq @ ub) @ normalize(hbc @ ub).T - normalize(hbq @ ua) @ normalize(hbc @ ua).T).max())
            u_full_errors.append(full_error)
            for d in dims:
                sb = normalize((hbq @ ub)[:, :d]) @ normalize((hbc @ ub)[:, :d]).T
                sa = normalize((hbq @ ua)[:, :d]) @ normalize((hbc @ ua)[:, :d]).T
                margins_before, margins_after = [], []
                for i in range(len(q)):
                    mi = {"query_id": qids[i], "candidate_rows": list(range(len(c))), "positive_rows": [i]}
                    margins_before.append(metric(sb[i], mi)["margin"])
                    margins_after.append(metric(sa[i], mi)["margin"])
                # Use true projector distance on near-orthonormal bases; the
                # identity shortcut suffers cancellation, so also record residual.
                pd2 = 2 * d - 2 * np.square(ub[:, :d].astype(np.float64).T @ ua[:, :d].astype(np.float64)).sum()
                direct = np.linalg.norm(ua[:, :d].astype(np.float64) @ ua[:, :d].astype(np.float64).T - ub[:, :d].astype(np.float64) @ ub[:, :d].astype(np.float64).T)
                writer.row("T09_U_step.csv", {"batch": batch_id, "probe_type": "update_batch", "dim": d,
                           "objective_before": before, "objective_after": after,
                           "margin_mean_before": quantiles(margins_before)["mean"], "margin_mean_after": quantiles(margins_after)["mean"],
                           "acc_before": np.mean(np.argmax(sb, axis=1) == np.arange(len(q))),
                           "acc_after": np.mean(np.argmax(sa, axis=1) == np.arange(len(q))),
                           "projector_distance": direct if max(ob, oa) <= 1e-4 else None,
                           "projector_identity_squared": pd2 if max(ob, oa) <= 1e-4 else None,
                           "full_score_max_error": full_error, "retraction_path": path,
                           "orthogonality_before": ob, "orthogonality_after": oa, "step_skipped": float(update_metrics["flag_skipped"])})
            (writer.directory / "T09_notes.md").write_text(
                f"Actual FlagProjector.step on a clone; config retraction={config['flag_retraction']}, lr={config['flag_lr']}, eps={config['flag_eps']}. "
                f"The implementation may fall back from Newton-Schulz to SVD (see flag_math.py); no source checkpoint is saved or changed. Protocol: {protocol}.\n")
            writer.paths["T09"].add(str(writer.directory / "T09_notes.md"))
        batch_count += 1
        del q, c, zq, zc, actual_total
        model.zero_grad(set_to_none=True)
    writer.json("T08_summary.json", {"batch_count": batch_count, "fraction_batches_full_total_inner_product_negative": full_conflicts / batch_count if batch_count else None,
                "gradient_total_relative_residuals": residuals, "conflict_pair_counts": dict(cos_conflicts),
                "protocol": protocol, "limits": "eval mode, fixed first batch per subset, no optimizer update; not proof of validation accuracy mechanism"}) if "T08" in selected else None
    for test in selected:
        mark(test, "INCONCLUSIVE" if surrogate else "PASS", f"Measured {batch_count} fixed batches; {protocol}")
    if "T07" in selected and max(full_loss_errors, default=0) > 1e-4:
        mark("T07", "FAIL", "Full task loss disagrees with same-embedding direct cosine CE; inspect T07_full_loss_check.csv")
    if "T08" in selected and max(residuals, default=0) > 1e-2:
        mark("T08", "WARN", "Weighted per-level parameter gradients differ from actual total by >1%; inspect dtype accumulation and recorded residuals")
    if "T09" in selected:
        writer.json("T09_retraction_paths.json", retraction_paths)
        status["T09"]["output_paths"] = sorted(writer.paths["T09"])
        if max(u_full_errors, default=0) > 1e-4:
            mark("T09", "FAIL", "Isolated U update changes full cosine by >1e-4; investigate retraction/precision")
    if "T07" in selected:
        status["T07"]["num_queries"] = n_audited
        status["T07"]["mask_difference_query_count"] = n_mask_differences
    return status


def probe_gradients(model, zq, zc, target, dims, weights, temperature, actual_total,
                    trainables, batch_id, subset, writer, residuals, conflicts):
    import torch
    from src.MRL.flag.encoder_loss import encoder_losses
    parameters = [p for _, p in trainables]
    gradients = []
    for d in dims:
        loss, _ = encoder_losses(model, zq, zc, target, [d], [1.], temperature)
        g = torch.autograd.grad(loss, parameters + [zq, zc], retain_graph=True, allow_unused=True)
        gradients.append([v.detach().cpu().float() if v is not None else None for v in g[:-2]])
        for side, zg, z in (("query", g[-2], zq), ("candidate", g[-1], zc)):
            previous = 0
            for end in dims:
                block = zg[:, previous:end] if zg is not None else z.new_zeros((len(z), end - previous))
                norm = float(block.detach().float().norm())
                writer.row("T08_embedding_block_gradients.csv", {"batch": batch_id, "loss_level": d,
                           "block_start": previous, "block_end": end, "side": side, "frobenius_norm": norm,
                           "rms": norm / math.sqrt(block.numel()), "outside_prefix": previous >= d})
                if previous >= d and norm != 0:
                    raise AssertionError("Direct embedding gradient outside loss prefix is not zero")
                previous = end
    def inner(a, b):
        return sum(float((x.double() * y.double()).sum()) for x, y in zip(a, b) if x is not None and y is not None)
    weighted_total = []
    for j, p in enumerate(parameters):
        values = [w * g[j] for w, g in zip(weights, gradients) if g[j] is not None]
        weighted_total.append(sum(values) if values else None)
    actual_g = torch.autograd.grad(actual_total, parameters, retain_graph=False, allow_unused=True)
    actual_g = [v.detach().cpu().float() if v is not None else None for v in actual_g]
    residual2 = sum(float(((a if a is not None else torch.zeros_like(b)) - (b if b is not None else torch.zeros_like(a))).double().square().sum())
                    for a, b in zip(weighted_total, actual_g) if a is not None or b is not None)
    relative = math.sqrt(residual2) / max(math.sqrt(inner(actual_g, actual_g)), 1e-30)
    residuals.append(relative)
    full_total = inner(gradients[-1], weighted_total)
    probe_gradients.last_full_dot = full_total
    for k, (d, g, w) in enumerate(zip(dims, gradients, weights)):
        norm = math.sqrt(inner(g, g))
        writer.row("T08_parameter_gradient_norms.csv", {"batch": batch_id, "subset": subset, "level": d,
                   "raw_norm": norm, "weighted_norm": abs(w) * norm, "full_total_inner_product": full_total,
                   "full_weighted_level_inner_product": w * inner(gradients[-1], g),
                   "actual_total_gradient_relative_residual": relative})
        for l in range(k, len(dims)):
            dot = inner(g, gradients[l])
            denominator = norm * math.sqrt(inner(gradients[l], gradients[l]))
            cosine = dot / denominator if denominator > 1e-24 else None
            if cosine is not None and cosine < 0:
                conflicts[f"{d},{dims[l]}"] += 1
            writer.row("T08_parameter_gradient_cosines.csv", {"batch": batch_id, "subset": subset,
                       "level_a": d, "level_b": dims[l], "cosine": cosine, "inner_product": dot,
                       "zero_norm_flags": denominator <= 1e-24})


def compare_checkpoints(args, writer):
    """T10 consumes separately encoded caches; never mixes h and U checkpoints."""
    if not args.baseline_cache:
        return {"status": "SKIPPED", "reason": "Missing --baseline-cache encoded by the actual base/pre-training checkpoint",
                "input_hashes": {}, "num_queries": 0, "output_paths": []}
    inputs, total_queries = {}, 0
    for subset in args.datasets:
        now = Cache(Path(args.output) / "cache" / subset)
        current_errors = now.validate()["validation_errors"]
        if current_errors:
            raise ValueError(f"T10 current cache invalid: {current_errors}")
        base_dir = Path(args.baseline_cache) / subset
        manifest = json.loads((base_dir / "manifest.json").read_text())
        if not manifest.get("raw_capture_verified") or not manifest.get("checkpoint_hash") or not manifest.get("file_hashes"):
            raise FileNotFoundError("Baseline raw h provenance/hashes missing; re-export from the actual baseline encoder")
        for name, expected in manifest["file_hashes"].items():
            if not (base_dir / name).is_file() or file_hash(base_dir / name) != expected:
                raise ValueError(f"T10 baseline artifact hash mismatch: {name}")
        inputs[subset] = {"baseline": manifest.get("checkpoint_hash"), "current": now.manifest["checkpoint_hash"]}
        for key in ("dataset", "split", "query_list_hash", "candidate_mapping_hash", "instruction_config_hash", "processor_config_hash"):
            if manifest.get(key) != now.manifest.get(key):
                raise ValueError(f"T10 examples/protocol mismatch {subset}/{key}; do not pair incomparable examples")
        bq, bc = (np.load(base_dir / f"h_{side}.npy", mmap_mode="r", allow_pickle=False) for side in ("query", "candidate"))
        if bq.shape != now.q.shape or bc.shape != now.c.shape:
            raise ValueError(f"T10 cache shapes differ for {subset}")
        base_full, now_full = [], []
        for cache_label, q, c, metadata in (("base", bq, bc, manifest), ("current", now.q, now.c, now.manifest)):
            accs = Counter()
            row_correct = []
            for m in now.mapping:
                r = metric(scores(q[m["query_row"]], c[m["candidate_rows"]], q.shape[1], chunk=args.candidate_chunk), m)
                row_correct.append(r["correct"])
            if cache_label == "base":
                base_full = row_correct
            else:
                now_full = row_correct
            baseline_acc = np.mean(base_full)
            writer.row("T10_checkpoint_comparison.csv", {"checkpoint": metadata.get("checkpoint_path"),
                       "step": metadata.get("checkpoint_step"), "dataset": subset, "representation": "raw_full",
                       "dim": q.shape[1], "acc": np.mean(row_correct), "delta_full_vs_base_pp": 100 * (np.mean(row_correct) - baseline_acc)})
            basis_path = base_dir / "U.npy" if cache_label == "base" else now.directory / "U.npy"
            u = np.load(basis_path, allow_pickle=False) if basis_path.exists() else None
            dims = [d for d in metadata.get("mrl_dims", DIMS) if d < q.shape[1]]
            for m in now.mapping:
                qr, cr = q[m["query_row"]], c[m["candidate_rows"]]
                if u is not None:
                    qr, cr = qr @ u, cr @ u
                for d in dims:
                    accs[d] += metric(scores(qr, cr, d, chunk=args.candidate_chunk), m)["correct"]
            for d in dims:
                writer.row("T10_checkpoint_comparison.csv", {"checkpoint": metadata.get("checkpoint_path"),
                           "step": metadata.get("checkpoint_step"), "dataset": subset,
                           "representation": "learned_prefix" if u is not None else "identity_prefix_no_learned_U",
                           "dim": d, "acc": accs[d] / len(now.mapping), "delta_full_vs_base_pp": None})
        buckets = Counter(group(a, b) for a, b in zip(base_full, now_full))
        writer.row("T10_checkpoint_transitions.csv", {"dataset": subset, "n": len(now.mapping), **{g: buckets[g] for g in GROUPS},
                   "delta_full_vs_base_pp": 100 * (np.mean(now_full) - np.mean(base_full))})
        total_queries += len(now.mapping)
    return {"status": "PASS", "reason": "Paired separately encoded checkpoints with identical example/protocol hashes",
            "input_hashes": inputs, "num_queries": total_queries, "output_paths": sorted(writer.paths["T10"])}


class DiagnosticMathTests(unittest.TestCase):
    """The twelve required mathematical cases plus cache/report regressions."""
    def setUp(self):
        self.rng = np.random.default_rng(7)
        self.q = self.rng.normal(size=8)
        self.c = self.rng.normal(size=(7, 8))
        self.m = {"query_id": "toy", "query_row": 0, "candidate_rows": list(range(7)), "positive_rows": [0, 3]}

    def test_01_identity_basis(self):
        u = np.eye(8)
        np.testing.assert_allclose(scores(self.q, self.c, 8), scores(self.q @ u, self.c @ u, 8), atol=1e-14)

    def test_02_random_orthogonal(self):
        u = np.linalg.qr(self.rng.normal(size=(8, 8)))[0]
        np.testing.assert_allclose(scores(self.q, self.c, 8), scores(self.q @ u, self.c @ u, 8), atol=1e-14)
        self.assertFalse(np.allclose(scores(self.q, self.c, 3), scores(self.q @ u, self.c @ u, 3)))

    def test_03_zero_tail(self):
        self.q[4:], self.c[:, 4:] = 0, 0
        for variant in ("S0", "S1", "S2", "gamma"):
            np.testing.assert_allclose(scores(self.q, self.c, 4, variant, .75), scores(self.q, self.c, 4), atol=1e-14)
        self.assertTrue(energy(self.q, 4)["near_zero_b"])
        self.assertTrue((scores(self.q, self.c, 4, "tail_only") == 0).all())

    def test_04_equal_energy_ratio(self):
        r = .7
        def equalize(x):
            return np.concatenate((normalize(x[..., :4]), math.sqrt(r) * normalize(x[..., 4:])), axis=-1)
        q, c = equalize(self.q), equalize(self.c)
        s0, s1 = scores(q, c, 4, "S0"), scores(q, c, 4, "S1")
        np.testing.assert_allclose(s1, s0 / (1 + r), atol=1e-14)
        self.assertEqual(np.argmax(s0), np.argmax(s1))

    def known_flip(self, denominator=False):
        q = np.array([1., 0., 1., 0.])
        c = np.array([[.6, .8, 0., 2.], [.59, math.sqrt(1 - .59 ** 2), 0., .1]])
        if not denominator:
            q[2:] *= math.sqrt(1 / 6)
            c[:, 2:] = np.array([[0., 1.], [.1, math.sqrt(.99)]]) * math.sqrt(1 / 6)
        m = {"query_id": "known", "candidate_rows": [0, 1], "positive_rows": [0]}
        paths = [scores(q, c, 2, v) for v in ("S0", "S1", "S2")]
        return q, c, m, paths

    def test_05_known_tail_flip(self):
        q, c, m, paths = self.known_flip()
        self.assertEqual([np.argmax(s) for s in paths], [0, 0, 1])
        np.testing.assert_allclose(paths[2], [.5142857142857, .52], atol=1e-12)
        dec = pair_decomposition(q, c, 2, paths, m, "full_witness")
        self.assertLess(dec["E_tail"], 0)

    def test_06_known_denominator_flip(self):
        q, c, m, paths = self.known_flip(True)
        self.assertEqual([np.argmax(s) for s in paths], [0, 1, 1])
        np.testing.assert_allclose(paths[1], paths[2], atol=1e-14)
        dec = pair_decomposition(q, c, 2, paths, m, "full_witness")
        self.assertAlmostEqual(dec["E_tail"], 0)
        self.assertLess(dec["E_norm"], 0)

    def test_07_gamma_endpoints_and_square(self):
        for dtype in (np.float32, np.float64):
            q, c = self.q.astype(dtype), self.c.astype(dtype)
            for gamma in GAMMAS:
                qg, cg = q.copy(), c.copy()
                qg[3:] *= gamma
                cg[:, 3:] *= gamma
                np.testing.assert_allclose(scores(q, c, 3, "gamma", gamma), scores(qg, cg, 8), atol=3e-7, rtol=3e-6)
            np.testing.assert_allclose(scores(q, c, 3, "gamma", 0), scores(q, c, 3), atol=1e-7)
            np.testing.assert_allclose(scores(q, c, 3, "gamma", 1), scores(q, c, 8), atol=1e-7)

    def test_08_fixed_pair_decomposition(self):
        for pp in ([0], [0, 3]):
            self.m["positive_rows"] = pp
            paths = [scores(self.q, self.c, 3, v) for v in ("S0", "S1", "S2")]
            for witness in ("full_witness", "prefix_witness"):
                dec = pair_decomposition(self.q, self.c, 3, paths, self.m, witness)
                self.assertLess(abs(dec["identity_residual"]), 1e-14)
                self.assertLess(abs(dec["tail_direct_residual"]), 1e-14)

    def test_09_ragged_candidate_pool(self):
        all_scores = np.array([.99, .1, .8, .2, .7])
        m = {"query_id": "ragged", "candidate_rows": [4, 2], "positive_rows": [2]}
        r = metric(all_scores[m["candidate_rows"]], m)
        self.assertEqual(r["pred_row"], 2)
        self.assertTrue(r["correct"])
        self.assertAlmostEqual(r["margin"], .1)

    def test_10_multi_positive_and_ties(self):
        m = {"query_id": "multi", "candidate_rows": [9, 4, 7], "positive_rows": [9, 4]}
        a = metric(np.array([.8, .8, .7]), m)
        b = metric(np.array([.79, .81, .7]), m)
        self.assertTrue(a["correct"] and b["correct"])
        self.assertEqual(a["pred_row"], 9)
        self.assertTrue(a["exact_tie"])
        c = metric(np.array([.8, .8 + 1e-8, .7]), m)
        self.assertEqual(c["pred_row"], 4)
        self.assertTrue(c["near_ties"][str(1e-7)])
        self.m["positive_rows"] = [0, 3]
        paths = [np.array([.8, .2, .1, .7, .3, .4, .5]),
                 np.array([.7, .2, .1, .6, .3, .4, .5]),
                 np.array([.6, .2, .1, .9, .3, .4, .5])]
        dec = pair_decomposition(self.q, self.c, 3, paths, self.m, "full_witness")
        self.assertEqual(dec["p_row"], 3)
        self.assertTrue(dec["p_changed_between_paths"])
        self.assertAlmostEqual(dec["delta2"] - dec["delta0"], dec["E_norm"] + dec["E_tail"])

    def test_11_transitions_identity(self):
        low, high = [True, True, False, False, True], [True, False, True, False, False]
        buckets = Counter(group(a, b) for a, b in zip(low, high))
        self.assertEqual(sum(high) - sum(low), buckets["correction"] - buckets["damage"])

    def test_12_chunking_ties_masks(self):
        self.c[1] = self.c[0]
        self.m["candidate_rows"] = [1, 0, 5]
        for variant in ("S0", "S1", "S2", "gamma", "tail_only"):
            a = scores(self.q, self.c[self.m["candidate_rows"]], 3, variant, .5, 100)
            b = scores(self.q, self.c[self.m["candidate_rows"]], 3, variant, .5, 1)
            np.testing.assert_allclose(a, b, atol=1e-14)
            ma, mb = metric(a, self.m), metric(b, self.m)
            for field in ("pred_position", "pred_row", "correct", "topk_positions", "best_positive_position", "hardest_negative_position"):
                self.assertEqual(ma[field], mb[field])
            self.assertAlmostEqual(ma["margin"], mb["margin"])

    def test_13_clamp_and_missing_negative(self):
        q = np.array([0., 0., 1., 0.])
        c = np.array([[0., 0., 0., 1.]])
        m = {"query_id": "zero", "candidate_rows": [0], "positive_rows": [0]}
        for variant in ("S0", "S1", "S2", "gamma", "tail_only"):
            self.assertTrue(np.isfinite(scores(q, c, 2, variant)).all())
        r = metric(scores(q, c, 2), m)
        self.assertIsNone(r["margin"])
        self.assertEqual(r["margin_missing_reason"], "NO_NEGATIVE")
        self.assertEqual(clean({"nan": float("nan"), "inf": np.float32(np.inf)}), {"nan": None, "inf": None})

    def test_14_cluster_bootstrap_determinism(self):
        mappings = [{"query_id": str(i), "group_id": str(i // 2)} for i in range(6)]
        a = paired_ci([True] * 6, [False] * 6, mappings, 100, 17)
        b = paired_ci([True] * 6, [False] * 6, mappings, 100, 17)
        self.assertEqual(a, b)
        self.assertEqual(a, (-100., -100., "cluster_group_id"))

    def test_15_duplicate_rows_positional_official_metric(self):
        m = {"query_id": "duplicate", "candidate_rows": [4, 4, 9], "positive_rows": [4], "positive_positions": [0]}
        r = metric(np.array([.5, .6, .4]), m)
        self.assertFalse(r["correct"])
        self.assertEqual(r["hardest_negative_position"], 1)

    def test_16_actual_loss_parameter_gradient_accounting(self):
        import torch
        from src.MRL.flag.encoder_loss import encoder_losses
        class ToyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.eye(8))
                self.unused = torch.nn.Parameter(torch.zeros(2))

            def compute_similarity(self, q, c):
                return q @ c.T
        model = ToyEncoder()
        q = torch.tensor(self.rng.normal(size=(4, 8)), dtype=torch.float32) @ model.weight
        c = torch.tensor(self.rng.normal(size=(4, 8)), dtype=torch.float32) @ model.weight
        target, dims, weights = torch.arange(4), [2, 4, 8], [.2, .3, .5]
        total, _ = encoder_losses(model, q, c, target, dims, weights, .2)
        residuals = []
        with tempfile.TemporaryDirectory(prefix="flagmrl-toy-grad-") as directory:
            writer = Writers(directory)
            try:
                probe_gradients(model, q, c, target, dims, weights, .2, total, list(model.named_parameters()),
                                "synthetic", "synthetic", writer, residuals, Counter())
            finally:
                writer.close()
        self.assertLess(max(residuals), 1e-6)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_17_actual_flag_step_keeps_full_cosine(self):
        import torch
        from src.MRL.flag.flag_projector import FlagProjector
        from src.MRL.flag.pairs import build_pair_weights
        projector = FlagProjector(8, [2, 4, 8], init="random", seed=7)
        old = projector.U.clone()
        q = torch.tensor(normalize(self.rng.normal(size=(4, 8))), dtype=torch.float32)
        c = torch.tensor(normalize(self.rng.normal(size=(4, 8))), dtype=torch.float32)
        near, far, valid = build_pair_weights(q, c, torch.arange(4))
        self.assertTrue(valid)
        projector.step((q, c, near), (q, c, far))
        f = torch.nn.functional.normalize
        before, after = f(q @ old, dim=-1) @ f(c @ old, dim=-1).T, f(q @ projector.U, dim=-1) @ f(c @ projector.U, dim=-1).T
        self.assertLess(float((before - after).abs().max()), 1e-5)
        self.assertLess(float((projector.U.T @ projector.U - torch.eye(8)).abs().max()), 1e-5)


def self_tests(output=None):
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DiagnosticMathTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"status": "PASS" if result.wasSuccessful() else "FAIL", "tests_run": result.testsRun,
              "passed": result.testsRun - len(result.failures) - len(result.errors),
              "failures": [{"test": str(t), "assertion": details} for t, details in result.failures],
              "errors": [{"test": str(t), "traceback": details} for t, details in result.errors],
              "input": "small synthetic tensors only; not real checkpoint results"}
    if output:
        dump(Path(output) / "toy_test_results.json", report)
    return report


def write_report(args, summary):
    """Generate measured findings only; skipped tests never get fake outputs."""
    findings = []
    lines = ["# FlagMRL degradation diagnostic", "", f"Checkpoint: `{args.checkpoint}`.", "",
             f"Toy mathematics: {summary['toy_tests']['passed']}/{summary['toy_tests']['tests_run']} PASS.", "",
             "| Dataset/scope | Test | Status | Reason |", "|---|---|---|---|"]
    for scope, tests in summary["datasets"].items():
        for test, item in sorted(tests.items()):
            lines.append(f"| {scope} | {test} | {item['status']} | {item['reason'].replace('|', '/')} |")
    for test, item in sorted(summary.get("tests", {}).items()):
        lines.append(f"| shared | {test} | {item['status']} | {item['reason'].replace('|', '/')} |")
    for subset in args.datasets:
        directory = Path(args.output) / subset
        state = summary["datasets"].get(subset, {})
        if state.get("T01", {}).get("status") not in {"PASS", "WARN"}:
            continue
        path = directory / "T02_transitions.csv"
        if state.get("T02", {}).get("status") == "PASS" and path.exists():
            with path.open() as stream:
                for row in csv.DictReader(stream):
                    findings.append(f"{subset} {row['low_dim']}→{row['high_dim']}: damage={row['damage']}, correction={row['correction']}, Δacc={float(row['delta_acc_pp']):.3f} pp (paired CI [{float(row['ci95_low_pp']):.3f}, {float(row['ci95_high_pp']):.3f}]).")
        path = directory / "T03_score_paths.csv"
        if state.get("T03", {}).get("status") == "PASS" and path.exists():
            rows = list(csv.DictReader(path.open()))
            for d in (256, 512, 768):
                values = {r["score_variant"]: float(r["acc"]) for r in rows if int(r["d"]) == d}
                if len(values) == 3:
                    findings.append(f"{subset} d={d}: S0={100 * values['S0']:.3f}%, S1={100 * values['S1']:.3f}%, S2={100 * values['S2']:.3f}%; denominator path Δ={100 * (values['S1'] - values['S0']):.3f} pp, tail numerator path Δ={100 * (values['S2'] - values['S1']):.3f} pp. See fixed-witness effects by group in T03_decomposition_summary.csv.")
    summary["confirmed_findings"] = findings
    lines += ["", "Measured evidence:", ""] + ([f"- {finding}" for finding in findings] if findings else ["No real-checkpoint finding is available. Toy tests validate diagnostic algebra only."])
    lines += ["", "Limits:", "", "- Without original run/query-list evidence, table reproduction is INCONCLUSIVE even when cache validation passes. The new checkpoint experiment can still be analyzed after T01 passes.",
              "- BF16 encoder states cast to FP32 remain BF16-quantized. Exact pool scoring uses original candidate order and positional positives from the official evaluator.",
              "- Gamma sweeps on the test split are diagnostic, not validation-selected model results. Random bases measure inference subspaces, not training seeds.",
              "- T07–T09 require original config; single-GPU surrogate probes do not establish global DDP training gradient behavior. T11 is not launched.",
              "", "| Evidence | Supported finding | Not established |", "|---|---|---|",
              "| T01 mismatch outside near ties | Pipeline/basis requires investigation | Tail geometry as the model cause |",
              "| Negative T03 E_norm and S1 drop | Denominator change hurts these rankings | Why the encoder learned these norms |",
              "| Negative T03 E_tail on damage witnesses | Tail relation hurts those fixed pairs | Which semantic features are responsible |",
              "| Intermediate T04 gamma performs better | Tail weighting differs from this diagnostic optimum | A gate generalizes |",
              "| T06 learned prefix beats random | U selects a useful subspace | Training degraded full embeddings |",
              "| Repeated T08 conflicts | Optimization conflict on measured batches | Cause of validation degradation |",
              "| T10 full below base | Full validation accuracy fell between checkpoints | Every optimization/data cause |", ""]
    if summary["missing_inputs"]:
        lines += ["Missing inputs:", ""] + [f"- {item}" for item in summary["missing_inputs"]] + [""]
    (Path(args.output) / "diagnostic_report.md").write_text("\n".join(lines))
    dump(Path(args.output) / "summary.json", summary)


def aggregate_accuracy(args, summary):
    collected = defaultdict(list)
    for subset, status in summary["datasets"].items():
        if status.get("T02", {}).get("status") != "PASS":
            continue
        with (Path(args.output) / subset / "T02_accuracy.csv").open() as stream:
            for row in csv.DictReader(stream):
                collected[int(row["dim"])].append(row)
    if not collected:
        return
    writer = Writers(args.output)
    try:
        for d, rows in sorted(collected.items()):
            for row in rows:
                writer.row("T02_accuracy.csv", row)
            writer.row("T02_accuracy.csv", {"dataset": f"macro_average_{len(rows)}_datasets", "dim": d,
                       "n": sum(int(r["n"]) for r in rows), "n_correct": None,
                       "acc": np.mean([float(r["acc"]) for r in rows])})
            writer.row("T02_accuracy.csv", {"dataset": f"micro_average_{len(rows)}_datasets", "dim": d,
                       "n": sum(int(r["n"]) for r in rows), "n_correct": sum(int(r["n_correct"]) for r in rows),
                       "acc": sum(int(r["n_correct"]) for r in rows) / sum(int(r["n"]) for r in rows)})
    finally:
        writer.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["self-test", "run", "compare"], nargs="?", default="run")
    p.add_argument("--checkpoint", default="training/FlagMRL_FastVLM_cls/checkpoint-epoch-0")
    p.add_argument("--output", default="test")
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--tests", nargs="+", default=["T00", "T01", "T02", "T03", "T04", "T05"],
                   help="Space or comma separated IDs. 'all' includes T00–T10; T11 never starts training")
    p.add_argument("--model-name", default="apple/FastVLM-0.5B")
    p.add_argument("--processor-name", default=None)
    p.add_argument("--dataset-name", default="TIGER-Lab/MMEB-eval")
    p.add_argument("--split", default="test")
    p.add_argument("--image-dir", default="data/MMEB-eval")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--candidate-chunk", type=int, default=2048)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--bootstrap-samples", type=int, default=2000)
    p.add_argument("--cache-only", action="store_true", help="Require verified diagnostic raw caches; do not encode/download models")
    p.add_argument("--training-config", help="Flat JSON with original training arguments, loss_gain_weight, original_world_size")
    p.add_argument("--train-image-dir")
    p.add_argument("--gradient-probe-max-batches", type=int, default=8)
    p.add_argument("--baseline-cache", help="T10 raw caches at <dir>/<dataset>; base U optional")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if min(args.batch_size, args.candidate_chunk, args.topk, args.bootstrap_samples, args.gradient_probe_max_batches) < 1:
        raise SystemExit("Batch/chunk/top-k/bootstrap/probe limits must be positive")
    if args.gradient_probe_max_batches > 8:
        raise SystemExit("Pilot gradient probe budget is capped at 8 batches")
    # Avoid overwriting source, production checkpoint, or arbitrary cache files.
    output, checkpoint = Path(args.output).resolve(), Path(args.checkpoint).resolve()
    if output == ROOT or checkpoint == output or checkpoint in output.parents or output in checkpoint.parents:
        raise SystemExit("--output must be separate from repository root and checkpoint path")
    output.mkdir(parents=True, exist_ok=True)
    args.output = str(output)
    toy = self_tests(args.output)
    if args.command == "self-test":
        return 0 if toy["status"] == "PASS" else 1
    args.tests = list(dict.fromkeys(t for item in args.tests for t in item.split(",")))
    if "all" in args.tests:
        args.tests = [f"T{i:02d}" for i in range(11)]
    unknown = set(args.tests) - {f"T{i:02d}" for i in range(12)}
    if unknown:
        raise SystemExit(f"Unknown diagnostic tests: {sorted(unknown)}")
    if args.command == "compare":
        args.tests = ["T10"]
        args.cache_only = True
    inference = any(t in args.tests for t in ("T00", "T01", "T02", "T03", "T04", "T05", "T06"))
    if inference:
        args.tests = list(dict.fromkeys(["T00", "T01", *args.tests]))
    if set(args.tests) & {"T08", "T09"} and "T07" not in args.tests:
        args.tests.append("T07")
    args.datasets = sorted(dict.fromkeys(args.datasets), key=lambda s: (DATASETS.index(s) if s in DATASETS else len(DATASETS), s))
    args.run_id = f"diagnostic-{time.time_ns()}"
    summary = {"schema_version": "1.0", "run_id": args.run_id, "checkpoint_hash": None,
               "code_revision": revision(),
               "tests": {}, "datasets": {}, "confirmed_findings": [], "supported_hypotheses": [],
               "unresolved_questions": ["No claim about tail semantics without measured pair/group effects"],
               "missing_inputs": [], "toy_tests": toy, "arguments": vars(args).copy()}
    import importlib.util
    dependencies = {name: importlib.util.find_spec(name) is not None
                    for name in ("torch", "numpy", "transformers", "datasets", "peft", "timm", "safetensors", "PIL")}
    summary["environment"] = {"python": sys.version, "executable": sys.executable,
                               "available_dependencies": dependencies,
                               "cuda_available": bool(sys.modules["torch"].cuda.is_available()) if "torch" in sys.modules else None}
    missing_dependencies = [name for name, present in dependencies.items() if not present]
    if missing_dependencies and not args.cache_only:
        summary["missing_inputs"].append("Inference dependencies not installed: " + ", ".join(missing_dependencies))
    def skipped(reason):
        return {"status": "SKIPPED", "reason": reason, "input_hashes": {}, "num_queries": 0, "output_paths": []}
    if toy["status"] != "PASS":
        summary["tests"]["mathematics"] = {"status": "FAIL", "reason": "Diagnostic self-tests failed; real experiments stopped"}
        write_report(args, summary)
        return 1
    if not args.cache_only and not checkpoint.is_dir():
        reason = f"Missing requested checkpoint: {checkpoint}"
        summary["missing_inputs"].append(reason)
        for subset in args.datasets:
            summary["datasets"][subset] = {test: skipped(reason) for test in args.tests if test != "T11"}
        summary["tests"]["T11"] = skipped("Controlled training probes not launched; need A–E evidence and explicit experiment design")
        write_report(args, summary)
        print(reason + f"\nMathematics PASS; model tests SKIPPED. Report: {output / 'diagnostic_report.md'}", flush=True)
        return 2
    import torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_bundle = None
    if not args.cache_only:
        summary["checkpoint_hash"] = checkpoint_hash(checkpoint)
    try:
        with reference_precision():
            if inference:
                for subset in args.datasets:
                    print(f"Diagnostic dataset: {subset}", flush=True)
                    writer = Writers(output / subset)
                    try:
                        if args.cache_only:
                            cache = Cache(output / "cache" / subset)
                            if Path(cache.manifest["checkpoint_path"]).resolve() != checkpoint:
                                raise ValueError("Cache is from a different checkpoint path")
                            if summary["checkpoint_hash"] is not None and summary["checkpoint_hash"] != cache.manifest["checkpoint_hash"]:
                                raise ValueError("Dataset caches have different checkpoint hashes")
                            summary["checkpoint_hash"] = cache.manifest["checkpoint_hash"]
                        else:
                            if model_bundle is None:
                                model_bundle = load_actual_model(args, summary["checkpoint_hash"])
                            cache = export_cache(args, subset, *model_bundle, summary["checkpoint_hash"])
                        summary["datasets"][subset] = evaluate_cache(cache, args, writer)
                        summary["missing_inputs"].extend(f"{subset}: {item}" for item in cache.manifest.get("missing_inputs", []))
                    except (FileNotFoundError, ModuleNotFoundError) as exc:
                        summary["missing_inputs"].append(f"{subset}: {exc}")
                        summary["datasets"][subset] = {test: skipped(str(exc)) for test in args.tests if test in {"T00", "T01", "T02", "T03", "T04", "T05", "T06"}}
                    except Exception as exc:
                        writer.json("T00_exception.json", {"exception": repr(exc), "traceback": traceback.format_exc()})
                        summary["datasets"][subset] = {"T00": {"status": "FAIL", "reason": str(exc), "input_hashes": {},
                                                                            "num_queries": 0, "output_paths": [str(writer.directory / "T00_exception.json")]}}
                        for test in args.tests:
                            if test in {"T01", "T02", "T03", "T04", "T05", "T06"}:
                                summary["datasets"][subset][test] = skipped("Diagnostic exception; inspect T00_exception.json")
                    finally:
                        writer.close()
                    write_report(args, summary)
                    # A pilot structural/pipeline failure blocks later model
                    # interpretation. Missing input is reported, never substituted.
                    states = summary["datasets"][subset]
                    if any(item["status"] == "FAIL" for item in states.values()):
                        for remaining in args.datasets[args.datasets.index(subset) + 1:]:
                            summary["datasets"][remaining] = {test: skipped(f"Earlier dataset {subset} failed; stop A/B rollout") for test in args.tests if test in {"T00", "T01", "T02", "T03", "T04", "T05", "T06"}}
                        break
            # Free the inference model before constructing an unmerged gradient
            # model; the training probes always use fresh detached/clone state.
            model_bundle = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            writer = Writers(output / "training_probes")
            try:
                pipeline_failed = any(item["status"] == "FAIL" for tests in summary["datasets"].values() for item in tests.values())
                if pipeline_failed:
                    for test in set(args.tests) & {"T07", "T08", "T09"}:
                        summary["tests"][test] = skipped("T00/T01 failed; training diagnostics blocked")
                else:
                    summary["tests"].update(training_probes(args, writer, summary["checkpoint_hash"]))
            except (FileNotFoundError, ModuleNotFoundError) as exc:
                summary["missing_inputs"].append(str(exc))
                for test in set(args.tests) & {"T07", "T08", "T09"}:
                    summary["tests"][test] = skipped(str(exc))
            except Exception as exc:
                writer.json("training_probe_exception.json", {"exception": repr(exc), "traceback": traceback.format_exc()})
                for test in set(args.tests) & {"T07", "T08", "T09"}:
                    summary["tests"][test] = {"status": "FAIL", "reason": str(exc), "input_hashes": {}, "num_queries": 0,
                                             "output_paths": [str(writer.directory / "training_probe_exception.json")]}
            finally:
                writer.close()
            if "T10" in args.tests:
                writer = Writers(output / "checkpoint_comparison")
                try:
                    summary["tests"]["T10"] = compare_checkpoints(args, writer)
                except FileNotFoundError as exc:
                    summary["tests"]["T10"] = skipped(f"Missing comparison cache: {exc}")
                except Exception as exc:
                    summary["tests"]["T10"] = {"status": "FAIL", "reason": str(exc), "input_hashes": {}, "num_queries": 0, "output_paths": []}
                finally:
                    writer.close()
    finally:
        summary["tests"]["T11"] = skipped("Controlled training is not launched by this script; requires original optimizer/schedule/batch sequence and prior A–E evidence")
        aggregate_accuracy(args, summary)
        write_report(args, summary)
    all_statuses = [item["status"] for scope in summary["datasets"].values() for item in scope.values()] + [item["status"] for key, item in summary["tests"].items() if key != "T11"]
    print(f"Diagnostic complete: {output / 'diagnostic_report.md'}", flush=True)
    return 1 if "FAIL" in all_statuses else 2 if "SKIPPED" in all_statuses else 0


if __name__ == "__main__":
    sys.exit(main())
