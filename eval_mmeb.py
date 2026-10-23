"""Reference MMEB encode/cache/score flow with FlagMRL inference projection."""

import json
import pickle
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, HfArgumentParser

from src.arguments import DataArguments, ModelArguments, TrainingArguments
from src.data.collator.eval_collator import EvalCollator
from src.data.dataset.mmeb_dataset import EvalDataset, load_eval_data
from evaluation.mmeb_baselines.eval_utils import get_pred
from src.model.model import MMEBModel
from src.model.processor import get_backbone_name, load_processor
from src.utils import print_rank, resolve_nested_dims
from src.wandb_logging import WandbLogger

POS_MOD_CLASS_LABEL = "Represent the class label: "
POS_MOD_IMAGE_CAPTION = "Represent the image caption: "
POS_MOD_ANSWER = "Represent the answer: "
POS_MOD_DICT = {
    **dict.fromkeys(["ImageNet-1K", "HatefulMemes", "SUN397", "N24News", "VOC2007",
                     "Place365", "ImageNet-A", "ImageNet-R", "ObjectNet", "Country211"], POS_MOD_CLASS_LABEL),
    **dict.fromkeys(["OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA",
                     "Visual7W", "ScienceQA", "GQA", "TextVQA", "VizWiz"], POS_MOD_ANSWER),
    **dict.fromkeys(["MSCOCO_i2t", "VisualNews_i2t"], POS_MOD_IMAGE_CAPTION),
}


def batch_to_device(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def evaluation_dims(model, nested_dims):
    if nested_dims:
        resolve_nested_dims(nested_dims, model.full_dim)
        dims = list(nested_dims)
    elif hasattr(model, "flag_projector"):
        dims = list(model.flag_projector.mrl_dims)
    else:
        dims = []
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
            output = model(**{side: batch_to_device(batch, device)})
        chunks.append(output[f"{side}_reps"].cpu().detach().float())
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


def inference_embeddings(model, queries, targets):
    """Apply X @ U once before the reference full/prefix scoring flow."""
    if not hasattr(model, "flag_projector"):
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
        qry_tensor, qry_index = pickle.load(stream)
    with paths["tgt"].open("rb") as stream:
        tgt_tensor, tgt_index = pickle.load(stream)
    # Cache encoder outputs exactly as in the reference; rotate only for inference.
    qry_tensor, tgt_tensor = inference_embeddings(model, qry_tensor, tgt_tensor)
    qry_key2emb, tgt_key2emb = OrderedDict(), OrderedDict()
    for qry_t, pair in zip(qry_tensor, qry_index):
        qry_key2emb[(pair["text"], pair["img_path"])] = qry_t
    for tgt_t, pair in zip(tgt_tensor, tgt_index):
        tgt_key2emb[(pair["text"], pair["img_path"])] = tgt_t
    dims = evaluation_dims(model, args.nested_dims)
    rows = load_eval_data(data_args, model_args, subset)
    if not len(rows):
        raise ValueError("Cannot score an empty dataset.")
    correct, predictions = 0, []
    nested_correct = {dim: 0 for dim in dims}
    nested_predictions = {dim: [] for dim in dims}
    for row in tqdm(rows, desc=f"Calculate score - {subset}"):
        keys = list(zip(row["tgt_text"], row["tgt_img_path"]))
        if not keys or len(row["tgt_text"]) != len(row["tgt_img_path"]):
            raise ValueError("MMEB rows need nonempty aligned candidate text/image lists.")
        qry_t = qry_key2emb[(row["qry_text"], row["qry_img_path"])]
        tgt_t = np.stack([tgt_key2emb[key] for key in keys], axis=0)
        qry_t = torch.from_numpy(np.array(qry_t))
        _, prediction = get_pred(qry_t, tgt_t, normalization=model_args.normalize)
        correct += int(prediction == 0)
        predictions.append(keys[prediction])
        for dim in dims:
            qry_t_dim = qry_t[:dim]
            tgt_t_dim = tgt_t[:, :dim]
            _, prediction_dim = get_pred(qry_t_dim, tgt_t_dim, normalization=model_args.normalize)
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
    argv = [arg.replace("--local-rank", "--local_rank", 1)
            if arg == "--local-rank" or arg.startswith("--local-rank=") else arg
            for arg in sys.argv[1:]]
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, args = parser.parse_args_into_dataclasses(args=argv)
    if data_args.eval_dataset_name:
        data_args.dataset_name = data_args.eval_dataset_name
    if data_args.eval_subset_name:
        data_args.subset_name = data_args.eval_subset_name
    if data_args.eval_image_dir:
        data_args.image_dir = data_args.eval_image_dir
    if not data_args.encode_output_path:
        raise ValueError("--encode_output_path is required for evaluation.")
    Path(data_args.encode_output_path).mkdir(parents=True, exist_ok=True)
    hf_config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
    if not model_args.model_backbone:
        model_args.model_backbone = get_backbone_name(hf_config=hf_config, model_type=model_args.model_type)
        args.model_backbone = model_args.model_backbone
    print_rank(f"model_backbone: {model_args.model_backbone}")
    processor = load_processor(model_args, data_args)
    model = MMEBModel.load(model_args, is_trainable=False)
    model.eval()
    model = model.to(args.device, dtype=torch.bfloat16)
    dims = evaluation_dims(model, args.nested_dims)
    subsets = data_args.subset_name or [None]
    with WandbLogger(model_args, data_args, args, "eval", nested_dims=dims) as logger:
        # First encode every subset; scoring can subsequently run from the CPU caches.
        for subset in subsets:
            if read_score(subset_paths(data_args, subset)["score.json"]) is not None:
                print(f"Found previous eval score, skipping {subset}")
                continue
            encode_subset(model, model_args, data_args, args, processor, subset)
        results = []
        for subset in tqdm(subsets, desc="Iterate datasets to calculate scores"):
            cached = read_score(subset_paths(data_args, subset)["score.json"]) is not None
            result = score_subset(model, model_args, data_args, args, subset)
            print(json.dumps({"subset": subset, **result}), flush=True)
            logger.log_eval(subset, result, cached=cached)
            results.append(result)
        logger.log_eval_average(results)


if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    print(f"Total runtime: {time.perf_counter() - start_time:.2f} seconds", flush=True)
