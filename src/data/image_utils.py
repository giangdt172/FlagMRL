"""Shared longest-side image resizing for training and evaluation."""

import os
from numbers import Integral

from PIL import Image


def resolve_image_resolution(resolution, max_dim=1344):
    """Resolve a positive pixel limit or a legacy preset; None keeps the image size."""
    if resolution is None:
        return None
    if isinstance(resolution, str):
        value = resolution.strip().lower()
        if value in {"", "none", "original"}:
            return None
        presets = {"high": max_dim, "mid": 672, "low": 448}
        if value in presets:
            resolution = presets[value]
        else:
            try:
                resolution = int(value)
            except ValueError:
                raise ValueError(
                    "image_resolution must be a positive integer (e.g. 768), "
                    "high/mid/low, or none."
                ) from None
    if isinstance(resolution, bool) or not isinstance(resolution, Integral) or resolution <= 0:
        raise ValueError("image_resolution must be a positive integer (e.g. 768).")
    return int(resolution)


def process_image(image, resolution, max_dim=1344):
    """Shrink the longest side to the limit, preserving aspect ratio and small images."""
    limit = resolve_image_resolution(resolution, max_dim=max_dim)
    if image is None or limit is None or max(image.size) <= limit:
        return image
    scale = limit / max(image.size)
    size = tuple(max(1, int(side * scale)) for side in image.size)
    resized = image.resize(size)
    if hasattr(image, "filename"):
        resized.filename = image.filename
    return resized


def load_image(img_path, image_dir, resolution=None, max_dim=1344):
    """Load on demand, close the source file, and retain its path after resizing."""
    if not img_path:
        return None
    full_path = os.path.join(image_dir or ".", img_path)
    with Image.open(full_path) as source:
        image = source.copy()
    image.filename = full_path
    return process_image(image, resolution, max_dim=max_dim)
