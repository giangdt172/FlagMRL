"""Dataset, collator, and encoder container for one-model training."""

import hashlib
import inspect
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset

from src.data.image_utils import load_image, resize_qwen2vl_train_image
from src.data.mmeb_utils import POS_MOD_ANSWER, POS_MOD_CLASS_LABEL, POS_MOD_DICT, POS_MOD_IMAGE_CAPTION
from src.model.model import MMEBModel
from src.model.processor import VLM_IMAGE_TOKENS, load_processor, process_vlm_inputs_fns
from src.utils import resolve_nested_dims


# Other backbones retain the CSM-VL training resize; Qwen2-VL uses Matryoshka below.
def process_image(image, resolution, max_dim=1024):
    if image is None:
        return None

    width, height = image.size
    max_side = max(width, height)

    if resolution == "high":
        target_max = 1024
    elif resolution == "mid":
        target_max = 672
    elif resolution == "low":
        target_max = 448
    else:
        try:
            target_max = int(resolution)
        except (TypeError, ValueError):
            target_max = max_dim

    # Tính tỉ lệ scale sao cho cạnh lớn nhất = target_max
    if max_side > target_max:
        scale = target_max / max_side
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = image.resize((new_width, new_height))

    return image


def stable_pos_id(text, image_path):
    """FlagMRL addition: identical positive content has a stable signed-int64 ID."""
    payload = (str(text) + "\0" + str(image_path or "")).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") & ((1 << 63) - 1)


class OneModelTrainer(nn.Module):
    def __init__(self, model_args, training_args, device, data_args=None):
        super().__init__()
        self.model_args, self.training_args = model_args, training_args
        self.data_args = data_args
        self.device = torch.device(device)
        self.temperature = model_args.temperature
        if self.temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.model = self._load_model()
        self._maybe_attach_flag_projector()

    def _load_model(self):
        model = MMEBModel.build(self.model_args)
        self._enable_gradient_checkpointing(model)
        return model.train().to(self.device)

    def _enable_gradient_checkpointing(self, model):
        if not self.training_args.gradient_checkpointing:
            return
        settings = self.training_args.gradient_checkpointing_kwargs or {}
        settings = json.loads(settings) if isinstance(settings, str) else dict(settings)
        settings.setdefault("use_reentrant", False)
        target = model.encoder
        target.config.use_cache = False
        if hasattr(target, "enable_input_require_grads"):
            target.enable_input_require_grads()
        if not hasattr(target, "gradient_checkpointing_enable"):
            raise ValueError("The encoder does not support gradient checkpointing.")
        enable = target.gradient_checkpointing_enable
        if "gradient_checkpointing_kwargs" in inspect.signature(enable).parameters:
            enable(gradient_checkpointing_kwargs=settings)
        else:
            enable()

    def _maybe_attach_flag_projector(self):
        if self.training_args.kd_loss_type != "flag_mrl":
            return
        from src.MRL.flag.flag_projector import FlagProjector
        args = self.training_args
        dims = resolve_nested_dims(args.nested_dims, self.model.full_dim)
        self.model.flag_projector = FlagProjector(
            self.model.full_dim, q_dims=dims,
            init=args.flag_init, eps=args.flag_eps, lr=args.flag_lr,
            retraction=args.flag_retraction, seed=args.seed,
            lambda_neg=args.flag_lambda_neg).to(self.device)
        checkpoint = Path(self.model_args.checkpoint_path or self.model_args.model_name) / "flag_U.pt"
        if checkpoint.exists():
            self.model.flag_projector.load(checkpoint)

    def get_processor(self):
        return load_processor(self.model_args, None)

    def forward(self, criterion, batch):
        return criterion(self, batch)


class TrainOneModelDataset(Dataset):
    def __init__(self, data_args, model_args):
        self.data_args, self.model_args = data_args, model_args
        from datasets import concatenate_datasets, load_dataset
        train_data = []
        for subset in data_args.subset_name or [None]:
            subset_data = load_dataset(data_args.dataset_name, subset, split=data_args.dataset_split)
            if subset == "WebQA" and "qry" in subset_data.column_names:
                subset_data = subset_data.map(
                    lambda row: {"qry": row["qry"].replace("<|image_1|>", "").strip()})
            # Filter before adding instructions and distributed sampling; empty sides cannot form pairs.
            subset_data = subset_data.filter(self._valid_row)
            instruction = POS_MOD_DICT.get(subset, "")
            positive_texts = [
                [instruction + (text or "") for text in texts] if isinstance(texts, list)
                else instruction + (texts or "") for texts in subset_data["pos_text"]
            ]
            subset_data = subset_data.add_column("pos_text_instruction", positive_texts)
            keep = {"qry", "qry_image_path", "pos_image_path", "pos_text_instruction"}
            subset_data = subset_data.remove_columns([column for column in subset_data.column_names if column not in keep])
            subset_data = subset_data.rename_column("pos_text_instruction", "pos_text")
            train_data.append(subset_data)
        self.train_data = concatenate_datasets(train_data)
        if not len(self.train_data):
            raise ValueError("No valid training samples remain.")

    @staticmethod
    def _valid_row(row):
        for text_key, path_key in (("qry", "qry_image_path"), ("pos_text", "pos_image_path")):
            texts = row[text_key] if isinstance(row[text_key], list) else [row[text_key]]
            paths = row[path_key] if isinstance(row[path_key], list) else [row[path_key]]
            if not texts or len(texts) != len(paths) or any(not (t or p) for t, p in zip(texts, paths)):
                return False
        queries = row["qry"] if isinstance(row["qry"], list) else [row["qry"]]
        positives = row["pos_text"] if isinstance(row["pos_text"], list) else [row["pos_text"]]
        return len(queries) == len(positives)

    def __len__(self):
        return len(self.train_data)

    def _get_image(self, img_path):
        image = load_image(img_path, self.data_args.image_dir)
        if self.data_args.image_resolution:
            if self.model_args.model_backbone == "qwen2_vl":
                return resize_qwen2vl_train_image(image, self.data_args.image_resolution)
            return process_image(image, self.data_args.image_resolution)
        return image

    def __getitem__(self, index):
        row = self.train_data[index]
        result = {}
        for text_key, image_key, output in (("qry", "qry_image_path", "query"), ("pos_text", "pos_image_path", "pos")):
            texts = row[text_key] if isinstance(row[text_key], list) else [row[text_key]]
            paths = row[image_key] if isinstance(row[image_key], list) else [row[image_key]]
            result[output + "_text"] = [(t or "").replace("<|image_1|>", VLM_IMAGE_TOKENS[self.model_args.model_backbone]) for t in texts]
            result[output + "_image"] = [self._get_image(p) for p in paths]
            if output == "pos":
                result["pos_id"] = [stable_pos_id(t, p) for t, p in zip(texts, paths)]
        return result


class TrainOneModelCollator:
    def __init__(self, processor, model_args, data_args, training_args, batch_size=None):
        self.processor, self.model_args = processor, model_args
        self.data_args, self.training_args = data_args, training_args
        self.batch_size = batch_size

    def _get_batch_inputs(self, batch, text_keyname, image_keyname):
        texts, visual_inputs = [], []
        for example in batch:
            if not example:
                texts.append(" ")
                visual_inputs.append(None)
            else:
                texts.extend(example[text_keyname])
                visual_inputs.extend(example[image_keyname])
        return {"text": texts, "images": visual_inputs}

    def __call__(self, examples):
        if not examples:
            raise ValueError("Cannot collate an empty batch.")
        student_qry_inputs = self._get_batch_inputs(examples, "query_text", "query_image")
        student_pos_inputs = self._get_batch_inputs(examples, "pos_text", "pos_image")
        batch_size = len(student_qry_inputs["text"])
        if not batch_size:
            raise ValueError("Cannot collate an empty batch.")
        if self.batch_size is not None and batch_size < self.batch_size:
            raise RuntimeError(f"Expected batch size {self.batch_size}, got {batch_size}.")
        process_student_fn = process_vlm_inputs_fns[self.model_args.model_backbone]
        result = {
            "qry": process_student_fn(student_qry_inputs, processor=self.processor, max_length=self.data_args.max_len),
            "pos": process_student_fn(student_pos_inputs, processor=self.processor, max_length=self.data_args.max_len),
        }
        if self.training_args.kd_loss_type == "flag_mrl":
            result["pos_id"] = torch.tensor([i for row in examples for i in row["pos_id"]], dtype=torch.long)
        return result
