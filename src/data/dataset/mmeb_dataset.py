"""MMEB query/target datasets for the reference evaluation protocol."""

from pathlib import Path

from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset

from prepare_one_model_training import process_image
from src.model.processor import VLM_IMAGE_TOKENS


def load_eval_data(data_args, model_args, subset):
    """Use the reference query cleanup for both cache keys and scoring rows."""
    rows = load_dataset(data_args.dataset_name, subset, split=data_args.dataset_split)
    if (subset in {"WebQA", "EDIS"} and "qry_text" in rows.column_names
            and model_args.model_backbone == "llava_qwen2"):
        rows = rows.map(lambda row: {"qry_text": row["qry_text"].replace("<|image_1|>", "").strip()})
    return rows


class EvalDataset(Dataset):
    def __init__(self, data_args, model_args, subset, text_field, img_path_field, mod_instruction=None):
        self.data_args, self.model_args = data_args, model_args
        self.eval_data = load_eval_data(data_args, model_args, subset)
        self.paired_data = self.get_paired_data(text_field, img_path_field)
        self.instruction = (mod_instruction or "") if text_field.startswith("tgt") and data_args.tgt_prefix_mod else ""

    def get_paired_data(self, text_field, img_path_field):
        seen, pairs = set(), []
        for row in self.eval_data:
            texts, paths = row[text_field], row[img_path_field]
            if isinstance(texts, str):
                # The reference retains repeated image-only rows in its cache index.
                if not texts:
                    paths = paths if isinstance(paths, list) else [paths]
                    pairs.extend({"text": texts, "img_path": path} for path in paths)
                    continue
                texts, paths = [texts], [paths]
            if not isinstance(paths, list) or len(texts) != len(paths):
                raise ValueError("MMEB text and image paths must be aligned.")
            for text, path in zip(texts, paths):
                key = (text, path)
                if key not in seen:
                    seen.add(key)
                    pairs.append({"text": text, "img_path": path})
        return pairs

    def __len__(self):
        return len(self.paired_data)

    def _get_image(self, image_path):
        if not image_path:
            return None
        path = Path(self.data_args.image_dir or ".") / image_path
        with Image.open(path) as source:
            image = source.copy()
        if self.data_args.image_resolution:
            limit = {"high": 1344, "mid": 672, "low": 448}.get(self.data_args.image_resolution, 1344)
            image = process_image(image, None, max_dim=limit)
        image.filename = str(path)
        return image

    def __getitem__(self, index):
        pair = self.paired_data[index]
        text = (self.instruction + (pair["text"] or "")).replace(
            "<|image_1|>", VLM_IMAGE_TOKENS[self.model_args.model_backbone])
        return text, self._get_image(pair["img_path"])
