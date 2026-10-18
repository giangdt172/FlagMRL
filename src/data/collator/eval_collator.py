# EvalCollator copied verbatim from vlm_embed.
from dataclasses import dataclass
from transformers import ProcessorMixin
from src.arguments import DataArguments, ModelArguments
from src.model.processor import INTERN_VL3, LLAVA_QWEN2, process_vlm_inputs_fns

@dataclass
class EvalCollator:
    data_args: DataArguments
    model_args: ModelArguments
    processor: ProcessorMixin

    def __call__(self, examples):
        """
        :param examples: qry, qry_image, pos_text, pos_image
        """
        examples = {'text': [e[0] for e in examples], 'images': [e[1] for e in examples]}
        
        if(self.model_args.model_backbone==INTERN_VL3):
            examples['images'] = [[i.filename] if i is not None else None for i in examples['images']]
            inputs = process_vlm_inputs_fns[self.model_args.model_backbone](examples,
                                            processor = self.processor,
                                            max_length = self.data_args.max_len)
        elif(self.model_args.model_backbone==LLAVA_QWEN2):
            inputs = process_vlm_inputs_fns[self.model_args.model_backbone](examples,
                                            processor = self.processor)
        elif(self.model_args.model_backbone!=INTERN_VL3):
            inputs = process_vlm_inputs_fns[self.model_args.model_backbone](examples,
                                            processor = self.processor,
                                            max_length = self.data_args.max_len)
            inputs['image_paths'] = [i.filename if i is not None else None for i in examples['images']]
            inputs['text'] = examples['text']
            inputs['images'] = examples['images']
        else:
            assert False
        return inputs
