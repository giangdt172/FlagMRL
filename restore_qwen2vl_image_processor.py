"""Restore the commented size initialization in Transformers 4.57.6.

Run with the Python interpreter used for training. This restores the upstream
code in that environment; it does not change FlagMRL's CSM-VL processor loader.
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


def main():
    installed = version("transformers")
    if installed != "4.57.6":
        raise RuntimeError(f"This repair targets Transformers 4.57.6; found {installed}.")
    spec = find_spec("transformers")
    path = Path(spec.origin).parent / "models/qwen2_vl/image_processing_qwen2_vl.py"
    original = path.read_text()
    repaired = restore_source(original)
    if repaired == original:
        print(f"Already restored: {path}")
        return
    backup = path.with_suffix(".py.before-size-restore.bak")
    # Never replace an existing backup of the installed file.
    with backup.open("x") as stream:
        stream.write(original)
    path.write_text(repaired)
    print(f"Restored upstream size initialization: {path}")
    print(f"Backup: {backup}")


if __name__ == "__main__":
    main()
