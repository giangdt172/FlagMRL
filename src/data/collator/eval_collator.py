"""Evaluation collation for FastVLM, Qwen2-VL, and Qwen3-VL."""

from dataclasses import dataclass
from transformers import ProcessorMixin
from src.arguments import DataArguments, ModelArguments
from src.model.processor import LLAVA_QWEN2, process_vlm_inputs_fns


@dataclass
class EvalCollator:
    data_args: DataArguments
    model_args: ModelArguments
    processor: ProcessorMixin

    def __call__(self, examples):
        """Batch text/image pairs using the selected backbone's processor."""
        examples = {'text': [e[0] for e in examples], 'images': [e[1] for e in examples]}

        process_fn = process_vlm_inputs_fns[self.model_args.model_backbone]
        if self.model_args.model_backbone == LLAVA_QWEN2:
            inputs = process_fn(examples, processor=self.processor)
        else:
            inputs = process_fn(examples, processor=self.processor, max_length=self.data_args.max_len)
            inputs['image_paths'] = [i.filename if i is not None else None for i in examples['images']]
            inputs['text'] = examples['text']
            inputs['images'] = examples['images']
        return inputs
