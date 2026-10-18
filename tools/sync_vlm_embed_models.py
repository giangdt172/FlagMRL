"""Vendor the three requested backbones from the local vlm_embed reference.

Only package imports/registries are narrowed. Model methods and processor
functions are extracted verbatim; backbone files are copied byte for byte.
Run with --check to verify provenance without modifying files.
"""

import argparse
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "VLM_Embed_Matryoshka-new_ap"


def extract(path, name):
    source = path.read_text()
    for node in ast.parse(source).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            return "\n".join(source.splitlines()[start - 1:node.end_lineno]) + "\n"
    raise ValueError(f"Missing {name} in {path}")


def expected_files():
    result = {}
    paths = [
        "llava/constants.py", "llava/mm_utils.py", "llava/processing_fastvlm.py",
        "llava/model/llava_arch.py", "llava/model/language_model/llava_qwen.py",
        "llava/model/language_model/modeling_qwen2.py",
        "llava/model/multimodal_projector/builder.py",
    ]
    for directory in ("llava/model/multimodal_encoder", "vlm_backbone/qwen2_vl",
                      "vlm_backbone/qwen3_vl_embedding"):
        paths.extend(str(path.relative_to(REFERENCE / "src/model"))
                     for path in sorted((REFERENCE / "src/model" / directory).rglob("*"))
                     if path.suffix in {".py", ".json"} and not path.name.startswith("demo"))
    for path in paths:
        result["src/model/" + path] = (REFERENCE / "src/model" / path).read_bytes()
    # Avoid eagerly importing unrelated Llama/MPT/Mistral models.
    for package in ("llava", "llava/model"):
        prefix = ".model.language_model" if package == "llava" else ".language_model"
        result[f"src/model/{package}/__init__.py"] = (
            f"from {prefix}.llava_qwen import LlavaQwen2ForCausalLM\n").encode()
    result["src/model/vlm_backbone/__init__.py"] = b""

    processor_path = REFERENCE / "src/model/processor.py"
    processor_source = processor_path.read_text()
    # Retain reference constants for names used in the original dispatch methods.
    constants = processor_source[processor_source.index("PHI_IMAGE_TOKEN_MAX_INPUT_ID"):
                                 processor_source.index("MODEL2BACKBONE =")]
    header = '''# Copied from vlm_embed; only imports and registries are scoped to three backbones.
import logging
import PIL
import torch
import numpy as np
from transformers.image_utils import ChannelDimension
from transformers import AutoTokenizer, AutoModel, AutoProcessor
from peft import PeftConfig
from src.utils import print_master, print_rank
from src.model.vlm_backbone.qwen2_vl import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from src.model.vlm_backbone.qwen3_vl_embedding.modeling_qwen3_vl_embedding import Qwen3VLForEmbedding
from src.model.llava.model.language_model.llava_qwen import LlavaQwen2ForCausalLM
from src.model.llava.processing_fastvlm import FastVLMProcessor

logger = logging.getLogger(__name__)
'''
    registry = '''
MODEL2BACKBONE = {LLAVA_QWEN2: LLAVA_QWEN2, QWEN2_VL: QWEN2_VL, QWEN3_VL: QWEN3_VL}
SUPPORTED_MODELS = set(MODEL2BACKBONE)
VLM_IMAGE_TOKENS = {LLAVA_QWEN2: "<image>", QWEN2_VL: "<|image_pad|>", QWEN3_VL: "<|image_pad|>"}
VLM_VIDEO_TOKENS = {LLAVA_QWEN2: "", QWEN2_VL: "<|video_pad|>", QWEN3_VL: "<|video_pad|>"}
backbone2model = {LLAVA_QWEN2: LlavaQwen2ForCausalLM, QWEN2_VL: Qwen2VLForConditionalGeneration,
                  QWEN3_VL: Qwen3VLForEmbedding}
'''
    functions = "\n".join(extract(processor_path, name) for name in (
        "load_processor", "get_backbone_name", "FastVLM_process_fn", "Qwen2_VL_process_fn", "Qwen3_VL_process_fn"))
    dispatch = '''
process_vlm_inputs_fns = {LLAVA_QWEN2: FastVLM_process_fn, QWEN2_VL: Qwen2_VL_process_fn,
                         QWEN3_VL: Qwen3_VL_process_fn}
'''
    result["src/model/processor.py"] = (header + constants + registry + "\n" + functions + dispatch).encode()
    model_header = '''# MMEBModel class copied verbatim from vlm_embed; imports scoped to three backbones.
from typing import Dict, Optional
import os
import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoConfig, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
from unittest.mock import patch
from src.arguments import ModelArguments, TrainingArguments
from src.model.processor import *

'''
    result["src/model/reference_model.py"] = (model_header + extract(
        REFERENCE / "src/model/model.py", "MMEBModel")).encode()
    collator_header = '''# EvalCollator copied verbatim from vlm_embed.
from dataclasses import dataclass
from transformers import ProcessorMixin
from src.arguments import DataArguments, ModelArguments
from src.model.processor import INTERN_VL3, LLAVA_QWEN2, process_vlm_inputs_fns

'''
    result["src/data/collator/eval_collator.py"] = (collator_header + extract(
        REFERENCE / "src/data/collator/eval_collator.py", "EvalCollator")).encode()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    files = expected_files()
    mismatches = []
    for relative, content in files.items():
        target = ROOT / relative
        if args.check:
            if not target.exists() or target.read_bytes() != content:
                mismatches.append(relative)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    if mismatches:
        raise SystemExit("Different from vlm_embed:\n" + "\n".join(mismatches))
    print(f"{'Verified' if args.check else 'Copied'} {len(files)} files from vlm_embed.")


if __name__ == "__main__":
    main()
