import datasets
from datasets import load_dataset
from torch.utils.data import Dataset

from src.data.image_utils import load_image
from src.model.processor import VLM_IMAGE_TOKENS


def process_image(image, resolution, max_dim=1344):
    if image is None:
        return None
    if resolution == "high":
        # image = image.resize((1344, 1344))
        max_dim = 1344
    elif resolution == "mid":
        # image = image.resize((672, 672))
        max_dim = 672
    elif resolution == "low":
        # image = image.resize((448, 448))
        max_dim = 448
    cur_max_dim = max(image.size)
    if cur_max_dim > max_dim:
        scale = max_dim / cur_max_dim
        new_width = int(image.size[0] * scale)
        new_height = int(image.size[1] * scale)
        image = image.resize((new_width, new_height))
    return image


def load_eval_data(data_args, model_args, subset):
    """Use identical query cleanup when encoding pairs and scoring cached embeddings."""
    rows = load_dataset(data_args.dataset_name, subset, split=data_args.dataset_split)
    if (subset in {"WebQA", "EDIS"} and "qry_text" in rows.column_names
            and model_args.model_backbone == "llava_qwen2"):
        rows = rows.map(lambda row: {"qry_text": row["qry_text"].replace("<|image_1|>", "").strip()})
    return rows


class EvalDataset(Dataset):
    def __init__(self, data_args, model_args, subset, text_field, img_path_field, mod_instruction=None):
        self.data_args, self.model_args = data_args, model_args
        self.backbone = model_args.model_backbone
        self.eval_data = load_eval_data(data_args, model_args, subset)
        self.paired_data = self.get_paired_data(text_field, img_path_field)
        instruction = ""
        if "tgt" in text_field and mod_instruction is not None and data_args.tgt_prefix_mod:
            instruction = mod_instruction
        # Cache keys retain the original text; only encoder inputs receive the prefix.
        self.paired_dataset = datasets.Dataset.from_dict({
            "text": [instruction + pair["text"] for pair in self.paired_data],
            "img_path": [pair["img_path"] for pair in self.paired_data],
        })

    def __len__(self):
        return len(self.paired_dataset)

    def __getitem__(self, item):
        pair = self.paired_dataset[item]
        text = pair["text"]
        text = text.replace("<|image_1|>", VLM_IMAGE_TOKENS[self.backbone])
        return text, self._get_image(pair["img_path"])

    def _get_image(self, img_path):
        image = load_image(img_path, self.data_args.image_dir)
        if image is not None and self.data_args.image_resolution:
            filename = image.filename
            image = process_image(image, self.data_args.image_resolution)
            image.filename = filename
        return image

    def get_paired_data(self, text_field, img_path_field):
        """Deduplicate in input order, retaining repeated scalar image-only rows."""
        seen, pairs = set(), []
        for row in self.eval_data:
            texts, paths = row[text_field], row[img_path_field]
            if isinstance(texts, str):
                if not texts:
                    paths = paths if isinstance(paths, list) else [paths]
                    pairs.extend({"text": texts, "img_path": path} for path in paths)
                    continue
                texts, paths = [texts], [paths]
            elif not isinstance(texts, list):
                continue
            if not isinstance(paths, list) or len(texts) != len(paths):
                raise ValueError("MMEB text and image paths must be aligned.")
            for text, path in zip(texts, paths):
                key = (text, path)
                if key not in seen:
                    seen.add(key)
                    pairs.append({"text": text, "img_path": path})
        return pairs
