"""Image loading and the Qwen2-VL Matryoshka training resize."""

import os
from PIL import Image


def resize_qwen2vl_train_image(image, resolution, max_dim=1024):
    """Match VLM_Embed_Matryoshka-new_ap's one-model training resize."""
    if image is None:
        return None

    width, height = image.size
    max_side = max(width, height)

    if resolution == "high":
        target_max = 1024
    elif resolution == "mid":
        target_max = 504
    elif resolution == "low":
        target_max = 448
    else:
        target_max = max_dim

    if max_side > target_max:
        scale = target_max / max_side
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = image.resize((new_width, new_height))

    return image


def load_image(img_path, image_dir):
    """Load without resizing, close the source file, and retain the image path."""
    if not img_path:
        return None
    full_path = os.path.join(image_dir or ".", img_path)
    with Image.open(full_path) as source:
        image = source.copy()
    image.filename = full_path
    return image
