"""Reference MMEB encode/cache/score flow with FlagMRL inference projection."""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import HfArgumentParser

from prepare_one_model_training import POS_MOD_DICT as TRAIN_POS_MOD_DICT
from src.arguments import DataArguments, ModelArguments, TrainingArguments
from src.data.collator.eval_collator import EvalCollator
from src.data.dataset.mmeb_dataset import EvalDataset
from evaluation.mmeb_baselines.eval_utils import get_pred
from src.model.model import MMEBModel
from src.model.processor import load_processor
from src.utils import resolve_nested_dims, to_device

# vlm_embed uses a different ImageNet subset name for evaluation than for training.
POS_MOD_DICT = {subset: prompt for subset, prompt in TRAIN_POS_MOD_DICT.items() if subset != "ImageNet_1K"}
POS_MOD_DICT["ImageNet-1K"] = TRAIN_POS_MOD_DICT["ImageNet_1K"]


def evaluation_dims(model, nested_dims):
    if nested_dims:
        resolve_nested_dims(nested_dims, model.full_dim)
        dims = list(nested_dims)
    elif hasattr(model, "flag_projector"):
        dims = model.flag_projector.q_dims + [model.full_dim]
    else:
        dims = []
    if hasattr(model, "flag_projector"):
        for dim in dims:
            if dim != model.full_dim and dim not in model.flag_projector.q_dims:
                raise ValueError(f"dim={dim} is not a trained flag level: {model.flag_projector.q_dims}")
    return dims


def subset_paths(data_args, subset):
    directory = Path(data_args.encode_output_path)
    name = subset or "default"
    return {kind: directory / f"{name}_{kind}" for kind in ("qry", "tgt", "score.json")}


def read_score(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


@torch.no_grad()
def encode_dataset(model, dataset, collator, batch_size, device, bf16=False, side="qry"):
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator,
                        shuffle=False, drop_last=False, num_workers=0)
    chunks = []
    for batch in tqdm(loader, desc=f"Encode {side}"):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
            output = model(**{side: to_device(batch, device)})
        chunks.append(output[f"{side}_reps"].detach().cpu().float().numpy())
    if not chunks:
        raise ValueError("Cannot evaluate an empty dataset.")
    return np.concatenate(chunks, axis=0)


def encode_subset(model, model_args, data_args, args, processor, subset):
    paths = subset_paths(data_args, subset)
    collator = EvalCollator(data_args, model_args, processor)
    for side in ("qry", "tgt"):
        if paths[side].exists():
            continue
        dataset = EvalDataset(
            data_args=data_args, model_args=model_args, subset=subset,
            text_field=f"{side}_text", img_path_field=f"{side}_img_path",
            mod_instruction=POS_MOD_DICT.get(subset) if side == "tgt" and data_args.tgt_prefix_mod else None)
        vectors = encode_dataset(model, dataset, collator, args.per_device_eval_batch_size,
                                 args.device, args.bf16 or args.device.type == "cuda", side)
        # Keep the reference cache format: full raw vectors and original pair keys.
        with paths[side].open("wb") as stream:
            pickle.dump((vectors, dataset.paired_data), stream)


def inference_embeddings(model, queries, targets, dims):
    """Rotate once for nested inference; full-dimensional scoring uses raw vectors."""
    if not hasattr(model, "flag_projector") or not any(dim < model.full_dim for dim in dims):
        return queries, targets
    basis = model.flag_projector.U.detach().cpu().float().numpy()
    return queries @ basis, targets @ basis


def write_predictions(path, predictions):
    with path.open("w") as stream:
        for prediction in predictions:
            stream.write(f"{prediction}\n")


def score_subset(model, model_args, data_args, args, subset):
    paths = subset_paths(data_args, subset)
    previous = read_score(paths["score.json"])
    if previous is not None:
        print(f"Found previous eval score, skipping {subset}")
        return previous
    with paths["qry"].open("rb") as stream:
        queries, query_pairs = pickle.load(stream)
    with paths["tgt"].open("rb") as stream:
        targets, target_pairs = pickle.load(stream)
    qindex = {(pair["text"], pair["img_path"]): i for i, pair in enumerate(query_pairs)}
    pindex = {(pair["text"], pair["img_path"]): i for i, pair in enumerate(target_pairs)}
    dims = evaluation_dims(model, args.nested_dims)
    nested_queries, nested_targets = inference_embeddings(model, queries, targets, dims)
    rows = load_dataset(data_args.dataset_name, subset, split=data_args.dataset_split)
    if not len(rows):
        raise ValueError("Cannot score an empty dataset.")
    correct, predictions = 0, []
    nested_correct = {dim: 0 for dim in dims}
    nested_predictions = {dim: [] for dim in dims}
    for row in tqdm(rows, desc=f"Calculate score - {subset}"):
        keys = list(zip(row["tgt_text"], row["tgt_img_path"]))
        if not keys or len(row["tgt_text"]) != len(row["tgt_img_path"]):
            raise ValueError("MMEB rows need nonempty aligned candidate text/image lists.")
        qi = qindex[(row["qry_text"], row["qry_img_path"])]
        pi = [pindex[key] for key in keys]
        query, candidates = queries[qi], targets[pi]
        _, prediction = get_pred(query, candidates, normalization=model_args.normalize)
        correct += int(prediction == 0)
        predictions.append(keys[prediction])
        for dim in dims:
            if dim == model.full_dim:
                query_dim, candidates_dim = query, candidates
            else:
                # Baseline: prefix. FlagMRL: prefix of the learned coordinates X @ U.
                query_dim, candidates_dim = nested_queries[qi, :dim], nested_targets[pi, :dim]
            _, prediction_dim = get_pred(query_dim, candidates_dim, normalization=model_args.normalize)
            nested_correct[dim] += int(prediction_dim == 0)
            nested_predictions[dim].append(keys[prediction_dim])
    result = {"acc": correct / len(rows), "num_correct": correct,
              "num_pred": len(predictions), "num_data": len(rows)}
    if dims:
        result["matryoshka"] = {
            str(dim): {"acc": nested_correct[dim] / len(rows), "num_correct": nested_correct[dim],
                       "num_pred": len(rows)} for dim in dims}
    directory = Path(data_args.encode_output_path)
    name = subset or "default"
    write_predictions(directory / f"{name}_pred.txt", predictions)
    for dim in dims:
        write_predictions(directory / f"{name}_pred_dim{dim}.txt", nested_predictions[dim])
    paths["score.json"].write_text(json.dumps(result, indent=4))
    return result


def evaluate_subset(model, model_args, data_args, args, processor, subset):
    """Evaluate one subset using the same persistent caches as the CLI."""
    Path(data_args.encode_output_path).mkdir(parents=True, exist_ok=True)
    previous = read_score(subset_paths(data_args, subset)["score.json"])
    if previous is not None:
        return previous
    encode_subset(model, model_args, data_args, args, processor, subset)
    return score_subset(model, model_args, data_args, args, subset)


def main():
    from train_ddp_one_model import normalize_local_rank
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, args = parser.parse_args_into_dataclasses(args=normalize_local_rank(sys.argv[1:]))
    if data_args.eval_dataset_name:
        data_args.dataset_name = data_args.eval_dataset_name
    if data_args.eval_subset_name:
        data_args.subset_name = data_args.eval_subset_name
    if data_args.eval_image_dir:
        data_args.image_dir = data_args.eval_image_dir
    if not data_args.encode_output_path:
        raise ValueError("--encode_output_path is required for evaluation.")
    Path(data_args.encode_output_path).mkdir(parents=True, exist_ok=True)
    model = MMEBModel.load(model_args, is_trainable=False).eval().to(args.device, dtype=torch.bfloat16)
    processor = load_processor(model_args, data_args)
    evaluation_dims(model, args.nested_dims)
    subsets = data_args.subset_name or [None]
    # First encode every subset; scoring can subsequently run from the CPU caches.
    for subset in subsets:
        if read_score(subset_paths(data_args, subset)["score.json"]) is not None:
            print(f"Found previous eval score, skipping {subset}")
            continue
        encode_subset(model, model_args, data_args, args, processor, subset)
    for subset in tqdm(subsets, desc="Iterate datasets to calculate scores"):
        result = score_subset(model, model_args, data_args, args, subset)
        print(json.dumps({"subset": subset, **result}))


if __name__ == "__main__":
    main()
