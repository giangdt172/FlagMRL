"""Compare dataclass declarations using AST only; never import the reference."""

import ast
import sys
from pathlib import Path

REQUIRED_TRAINING = {
    "optimizer_name", "moon_non_2d_strategy", "image_encoder_freeze", "output_dir",
    "resume_from", "project_name", "logging_steps", "num_train_epochs",
    "kd_loss_type", "nested_dims",
}


def evaluate(node):
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return evaluate(node.left) * evaluate(node.right)
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return ast.dump(node, include_attributes=False)


def read_fields(path):
    classes = {}
    for node in ast.parse(Path(path).read_text()).body:
        if not isinstance(node, ast.ClassDef):
            continue
        fields = {}
        for item in node.body:
            if not isinstance(item, ast.AnnAssign):
                continue
            value = item.value
            if isinstance(value, ast.Call):
                options = {kw.arg: kw.value for kw in value.keywords}
                default = {k: evaluate(v) for k, v in options.items() if k in {"default", "default_factory"}}
                help_text = evaluate(options["metadata"]) if "metadata" in options else {}
            else:
                default, help_text = {"default": evaluate(value)}, {}
            fields[item.target.id] = (ast.unparse(item.annotation), default, help_text)
        classes[node.name] = fields
    return classes


def check(reference, candidate):
    ref, new = read_fields(reference), read_fields(candidate)
    errors = []
    count = 0
    for cls in ("ModelArguments", "DataArguments", "TrainingArguments"):
        names = REQUIRED_TRAINING if cls == "TrainingArguments" else ref[cls].keys()
        for name in sorted(names):
            count += 1
            if ref[cls].get(name) != new.get(cls, {}).get(name):
                errors.append(f"{cls}.{name}: reference={ref[cls].get(name)!r}; candidate={new.get(cls, {}).get(name)!r}")
    for error in errors:
        print(error)
    if not errors:
        print(f"Args parity OK: {count} fields (type, default, help).")
    return not errors


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: check_args_parity.py REFERENCE_ARGUMENTS NEW_ARGUMENTS")
    sys.exit(0 if check(*sys.argv[1:]) else 1)
