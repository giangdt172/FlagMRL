"""Repair Qwen2-VL size initialization in the training environment.

Run with the Python interpreter used for training (Transformers 4.57.6).
Restores the upstream initialization if it was commented out, then adds support
for legacy size={min_pixels, max_pixels} checkpoint configurations. FlagMRL's
CSM-VL processor loading flow is unchanged. The installed library is backed up
before the first repair; reinstalling Transformers removes the repair.

Usage: .venv/bin/python restore_qwen2vl_image_processor.py
"""

import ast
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
import re


# Upstream v4.57.6, Qwen2VLImageProcessor.__init__.
ORIGINAL_BLOCK = '''        if size is not None and ("shortest_edge" not in size or "longest_edge" not in size):
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
        else:
            size = {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}
'''


LEGACY_SIZE_BLOCK = '''        # FlagMRL compatibility for legacy Qwen2-VL checkpoint sizes.
        if isinstance(size, dict) and ("min_pixels" in size or "max_pixels" in size):
            if min_pixels is None:
                min_pixels = size.get("min_pixels", size.get("shortest_edge"))
            if max_pixels is None:
                max_pixels = size.get("max_pixels", size.get("longest_edge"))
            size = {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}
'''


def restore_source(source):
    if ORIGINAL_BLOCK in source:
        return source
    # Only accept the exact upstream block with its four lines commented out,
    # as shown in the failing environment. Reject other edits.
    pattern = re.compile(
        "".join(
            r"^[ \t]*#[ \t]*" + re.escape(line.strip()) + r"[ \t]*\n"
            for line in ORIGINAL_BLOCK.splitlines()
        ),
        re.MULTILINE,
    )
    repaired, count = pattern.subn(lambda _: ORIGINAL_BLOCK, source)
    if count != 1:
        raise RuntimeError("Unrecognized source; no changes made. Expected the four commented size initialization lines.")
    ast.parse(repaired)
    return repaired


def repair_source(source):
    """Restore known upstream code and normalize legacy sizes before validation."""
    source = restore_source(source)
    if LEGACY_SIZE_BLOCK + ORIGINAL_BLOCK in source:
        return source
    if source.count(ORIGINAL_BLOCK) != 1 or "# FlagMRL compatibility" in source:
        raise RuntimeError("Unrecognized source; no changes made. Expected one unmodified size initialization block.")
    repaired = source.replace(ORIGINAL_BLOCK, LEGACY_SIZE_BLOCK + ORIGINAL_BLOCK, 1)
    ast.parse(repaired)
    return repaired


def main():
    installed = version("transformers")
    if installed != "4.57.6":
        raise RuntimeError(f"This repair targets Transformers 4.57.6; found {installed}.")
    spec = find_spec("transformers")
    path = Path(spec.origin).parent / "models/qwen2_vl/image_processing_qwen2_vl.py"
    original = path.read_text()
    repaired = repair_source(original)
    if repaired == original:
        print(f"Already repaired (legacy sizes supported): {path}")
        return
    backup = path.with_suffix(".py.before-legacy-size-repair.bak")
    # Keep earlier backups, including ones from the original restore-only script.
    try:
        with backup.open("x") as stream:
            stream.write(original)
    except FileExistsError:
        if backup.read_text() != original:
            raise RuntimeError(f"Backup already exists with different contents; no changes made: {backup}")
    path.write_text(repaired)
    print(f"Repaired size initialization and enabled legacy checkpoint sizes: {path}")
    print(f"Backup: {backup}")
    print("FlagMRL's processor loader is unchanged; restart training to use the repair.")


if __name__ == "__main__":
    main()
