"""Processors and model registry for FastVLM, Qwen2-VL, and Qwen3-VL."""

import logging

import PIL
from transformers.image_utils import ChannelDimension

from src.model.vlm_backbone.qwen2_vl import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from src.model.vlm_backbone.qwen3_vl_embedding.modeling_qwen3_vl_embedding import Qwen3VLForEmbedding
from src.model.llava.model.language_model.llava_qwen import LlavaQwen2ForCausalLM
from src.model.llava.processing_fastvlm import FastVLMProcessor
from src.utils import print_master

logger = logging.getLogger(__name__)

LLAVA_QWEN2 = "llava_qwen2"
QWEN2_VL = "qwen2_vl"
QWEN3_VL = "qwen3_vl"

MODEL2BACKBONE = {LLAVA_QWEN2: LLAVA_QWEN2, QWEN2_VL: QWEN2_VL, QWEN3_VL: QWEN3_VL}
SUPPORTED_MODELS = set(MODEL2BACKBONE)
VLM_IMAGE_TOKENS = {LLAVA_QWEN2: "<image>", QWEN2_VL: "<|image_pad|>", QWEN3_VL: "<|image_pad|>"}
VLM_VIDEO_TOKENS = {LLAVA_QWEN2: "", QWEN2_VL: "<|video_pad|>", QWEN3_VL: "<|video_pad|>"}
backbone2model = {
    LLAVA_QWEN2: LlavaQwen2ForCausalLM,
    QWEN2_VL: Qwen2VLForConditionalGeneration,
    QWEN3_VL: Qwen3VLForEmbedding,
}


def load_processor(model_args, data_args=None):
    """Load the processor for one of the three supported VLM backbones."""
    if model_args.model_backbone not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported backbone {model_args.model_backbone!r}. "
            f"Supported backbones: {sorted(SUPPORTED_MODELS)}"
        )
    model_name_or_path = model_args.checkpoint_path or model_args.model_name
    print_master(f"Loading processor from: {model_name_or_path}")

    if model_args.model_backbone == QWEN2_VL:
        from transformers import Qwen2TokenizerFast, Qwen2VLImageProcessor

        if model_args.init_lora_model:
            from peft import PeftConfig

            model_name_or_path = PeftConfig.from_pretrained(model_args.model_name).base_model_name_or_path
        image_processor = Qwen2VLImageProcessor.from_pretrained(model_name_or_path)
        if data_args is not None:
            image_processor.min_pixels = data_args.resize_min_pixels
            image_processor.max_pixels = data_args.resize_max_pixels
        tokenizer = Qwen2TokenizerFast.from_pretrained(model_name_or_path)
        return Qwen2VLProcessor.from_pretrained(
            model_name_or_path, image_processor=image_processor, tokenizer=tokenizer
        )

    if model_args.model_backbone == QWEN3_VL:
        from src.model.vlm_backbone.qwen3_vl_embedding.processing_qwen3_vl import Qwen3VLProcessor

        return Qwen3VLProcessor.from_pretrained(model_name_or_path, trust_remote_code=True, padding_side="right")

    from transformers import AutoTokenizer, CLIPImageProcessor

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)
    image_processor = CLIPImageProcessor(
        crop_size={"height": 1024, "width": 1024},
        image_mean=[0.0, 0.0, 0.0],
        image_std=[1.0, 1.0, 1.0],
        size={"shortest_edge": 1024},
    )
    return FastVLMProcessor(image_processor=image_processor, tokenizer=tokenizer)


def get_backbone_name(hf_config, model_type=None):
    if model_type is not None:
        setattr(hf_config, 'model_type', model_type)
    assert hf_config.model_type in SUPPORTED_MODELS, f"Unknown backbone name {hf_config.model_type}.Supported models are {SUPPORTED_MODELS}"
    print(f"Detected model type: {hf_config.model_type}")
    model_backbone = MODEL2BACKBONE[hf_config.model_type]
    print(f"Determined model backbone: {model_backbone}")
    return MODEL2BACKBONE[hf_config.model_type]

def FastVLM_process_fn(model_inputs: dict, processor: FastVLMProcessor, max_length=None):
    texts, visual_inputs = model_inputs['text'], model_inputs['images']
    eos_token = getattr(processor.tokenizer, "eos_token", None)
    if eos_token:
        texts = [
            text if text.rstrip().endswith(eos_token) else text + eos_token
            for text in texts
        ]
    inputs = processor(
        images=visual_inputs,
        texts=texts,
    )
    return inputs

def Qwen2_VL_process_fn(model_inputs: dict, processor: Qwen2VLProcessor, max_length=None):
    # TODO: set separate max_len for text/visual inputs, currently max_length is only applied to text-only data
    input_ids, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw = [], [], [], [], []
    texts, visual_inputs = model_inputs['text'], model_inputs['images']
    image_exists = False
    vlm_image_token, vlm_video_token = VLM_IMAGE_TOKENS[QWEN2_VL], VLM_VIDEO_TOKENS[QWEN2_VL]
    # import ipdb; ipdb.set_trace()
    # 1. iterate each pair and process (since processors do not support batch processing)
    # print("texts:", texts)
    # print("visual_inputs types:", [type(img) for img in visual_inputs])
    for text, images in zip(texts, visual_inputs):
        # print(f"Processing text: {text}")
        if images is None or (type(images)==list and any(i is None for i in images)):
            # all images must be valid
            inputs = processor(text=[text], images=None, return_tensors="pt", max_length=max_length, truncation=True)
            input_id = inputs["input_ids"][0].tolist()
            input_ids.append(input_id)
            pixel_values.append(None)
            image_grid_thw.append(None)
            pixel_values_videos.append(None)
            video_grid_thw.append(None)
        else:
            if vlm_image_token in text:
                if isinstance(images, PIL.Image.Image):
                    # images is a single image
                    images = [images]
                    
                for iid, image in enumerate(images):
                    # rare case in MMEB eval: resize to 28*28 if either w or h is smaller than 28
                    if image.size[0] < 28 or image.size[1] < 28:
                        image = image.resize((56, 56))
                        images[iid] = image
                # for image in images:
                #     print(f"Image size (w, h): {image.size}, mode: {image.mode}, type: {type(image)}")
                inputs = processor(text=[text], images=images, return_tensors="pt", max_length=None, truncation=False, input_data_format=ChannelDimension.LAST)
            elif vlm_video_token in text:
                # TODO: check text/video data validity
                inputs = processor(text=[text], videos=[images], return_tensors="pt", max_length=None, truncation=False, input_data_format=ChannelDimension.LAST)
            else:
                raise NotImplementedError
            input_ids.append(inputs["input_ids"][0].tolist())
            if 'pixel_values' in inputs:
                pixel_values.append(inputs['pixel_values'])
                image_grid_thw.append(inputs['image_grid_thw'])
                pixel_values_videos.append(None)
                video_grid_thw.append(None)
            else:
                pixel_values.append(None)
                image_grid_thw.append(None)
                pixel_values_videos.append(inputs['pixel_values_videos'])
                video_grid_thw.append(inputs['video_grid_thw'])

    # 2. padding inputs
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    # print(f"Padded input_ids: {batch_encoding['input_ids']}")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    # manually enforce long type due to:
    # (1) [rank7]: RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.cuda.FloatTensor instead (while checking arguments for embedding)
    # (2) [rank7]:   File "/fsx/home/ruimeng/project/VLM2Vec/src/model.py", line 45, in _pooling
    #     [rank7]:     reps = last_hidden_state[
    #     [rank7]: IndexError: tensors used as indices must be long, int, byte or bool tensors
    inputs = {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask.long(), 
        'texts': texts,
        'images': visual_inputs,
    }
    inputs['pixel_values'] = pixel_values
    inputs['image_grid_thw'] = image_grid_thw
    inputs['pixel_values_videos'] = pixel_values_videos
    inputs['video_grid_thw'] = video_grid_thw

    return inputs

def Qwen3_VL_process_fn(model_inputs: dict, processor, max_length=None, instruction=None):
    from qwen_vl_utils.vision_process import process_vision_info
    
    texts = model_inputs['text']
    visual_inputs = model_inputs['images']
    
    vlm_image_token = VLM_IMAGE_TOKENS[QWEN3_VL]
    vlm_video_token = VLM_VIDEO_TOKENS[QWEN3_VL]
    
    default_instruction = instruction or "Represent the user's input."
    
    # Step 1: Build conversations in Qwen3-VL chat format
    conversations = []
    for text, images in zip(texts, visual_inputs):
        has_image = vlm_image_token in text
        has_video = vlm_video_token in text
        no_visual = images is None or (isinstance(images, list) and any(i is None for i in images))
        
        content = []
        
        if no_visual or (not has_image and not has_video):
            # Text-only: strip visual tokens, keep text
            clean_text = text.replace(vlm_image_token, '').replace(vlm_video_token, '').strip()
            if not clean_text:
                clean_text = "NULL"
            content.append({"type": "text", "text": clean_text})
        elif has_image:
            # Image input
            if isinstance(images, PIL.Image.Image):
                images = [images]
            
            # Resize small images
            for iid, image in enumerate(images):
                if image.size[0] < 28 or image.size[1] < 28:
                    images[iid] = image.resize((56, 56))
            
            # Add image entries
            for img in images:
                if isinstance(img, PIL.Image.Image):
                    content.append({"type": "image", "image": img})
                elif isinstance(img, str):
                    img_path = img if img.startswith(('http://', 'https://')) else 'file://' + img
                    content.append({"type": "image", "image": img_path})
            
            # Add text (strip the image token placeholder)
            clean_text = text.replace(vlm_image_token, '').strip()
            if clean_text:
                content.append({"type": "text", "text": clean_text})
        
        elif has_video:
            # Video input (frames as list of images)
            if isinstance(images, list):
                video_frames = images
            else:
                video_frames = [images]
            
            # Convert frames to proper format
            video_content = [
                ('file://' + f if isinstance(f, str) and not f.startswith(('http://', 'https://')) else f)
                for f in video_frames
            ]
            
            content.append({"type": "video", "video": video_content})
            
            # Add text (strip the video token placeholder)
            clean_text = text.replace(vlm_video_token, '').strip()
            if clean_text:
                content.append({"type": "text", "text": clean_text})
        
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": default_instruction}]},
            {"role": "user", "content": content}
        ]
        conversations.append(conversation)
    
    # Step 2: Apply chat template to get formatted text
    chat_texts = processor.apply_chat_template(
        conversations, add_generation_prompt=True, tokenize=False
    )
    
    # Step 3: Extract vision info using qwen_vl_utils
    try:
        all_images, video_inputs, video_kwargs = process_vision_info(
            conversations, image_patch_size=16,
            return_video_metadata=True, return_video_kwargs=True
        )
    except Exception as e:
        logger.error(f"Error in processing vision info: {e}")
        all_images = None
        video_inputs = None
        video_kwargs = {'do_sample_frames': False}
        # Fallback to text-only
        fallback_conv = [[{'role': 'user', 'content': [{'type': 'text', 'text': 'NULL'}]}]] * len(texts)
        chat_texts = processor.apply_chat_template(
            fallback_conv, add_generation_prompt=True, tokenize=False
        )
    
    if video_inputs is not None:
        videos, video_metadata = zip(*video_inputs)
        videos = list(videos)
        video_metadata = list(video_metadata)
    else:
        videos, video_metadata = None, None
    
    # Step 4: Process through processor (tokenize + vision encoding)
    inputs = processor(
        text=chat_texts,
        images=all_images,
        videos=videos,
        video_metadata=video_metadata,
        truncation=True,
        max_length=max_length or 8192,
        padding=True,
        do_resize=False,
        return_tensors='pt',
        **video_kwargs
    )
    
    # Step 5: Build final output dict
    result = {
        'input_ids': inputs['input_ids'].long(),
        'attention_mask': inputs['attention_mask'].long(),
    }
    
    if 'pixel_values' in inputs and inputs['pixel_values'] is not None:
        result['pixel_values'] = inputs['pixel_values']
    if 'image_grid_thw' in inputs and inputs['image_grid_thw'] is not None:
        result['image_grid_thw'] = inputs['image_grid_thw']
    if 'pixel_values_videos' in inputs and inputs['pixel_values_videos'] is not None:
        result['pixel_values_videos'] = inputs['pixel_values_videos']
    if 'video_grid_thw' in inputs and inputs['video_grid_thw'] is not None:
        result['video_grid_thw'] = inputs['video_grid_thw']
    
    return result

process_vlm_inputs_fns = {LLAVA_QWEN2: FastVLM_process_fn, QWEN2_VL: Qwen2_VL_process_fn,
                         QWEN3_VL: Qwen3_VL_process_fn}
