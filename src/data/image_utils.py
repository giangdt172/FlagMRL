"""Image loading shared by the separate CSM-VL train/eval resize paths."""

import os
from PIL import Image


def load_image(img_path, image_dir):
    """Load without resizing, close the source file, and retain the image path."""
    if not img_path:
        return None
    full_path = os.path.join(image_dir or ".", img_path)
    with Image.open(full_path) as source:
        image = source.copy()
    image.filename = full_path
    return image
