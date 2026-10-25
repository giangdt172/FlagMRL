"""MMEB encoder for FastVLM, Qwen2-VL, and Qwen3-VL."""
from typing import Dict, Optional
from types import SimpleNamespace
import os
import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model, PeftModel
from src.arguments import ModelArguments
from src.model.processor import (
    LLAVA_QWEN2,
    QWEN2_VL,
    QWEN3_VL,
    backbone2model,
    get_backbone_name,
)
from src.utils import print_master

class MMEBModel(nn.Module):
    TRANSFORMER_CLS = AutoModelForCausalLM

    def __init__(self,
                 encoder: PreTrainedModel,
                 pooling: str = 'last',
                 normalize: bool = False,
                 temperature: float = 0.02,
                 ):
        super().__init__()
        self.config = encoder.config
        self.encoder = encoder
        self.pooling = pooling
        self.normalize = normalize
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def _extract_last_hidden_state(self, model_output):
        last_hidden_state = getattr(model_output, "last_hidden_state", None)
        if last_hidden_state is not None:
            return last_hidden_state

        hidden_states = getattr(model_output, "hidden_states", None)
        if hidden_states is not None and len(hidden_states) > 0:
            return hidden_states[-1]

        raise AttributeError(
            "Model output does not contain `last_hidden_state` or `hidden_states`. "
            "Enable `output_hidden_states=True` for this backbone."
        )

    def encode_input(self, input, output_hidden_states: bool = False, output_attentions: bool = False,
                     normalize: Optional[bool] = None):
        # FlagMRL requests raw pooled states, then normalizes each flag prefix.
        model_backbone = getattr(self, "model_backbone", None)
        if model_backbone not in (LLAVA_QWEN2, QWEN2_VL, QWEN3_VL):
            raise ValueError(f"Unsupported backbone {model_backbone!r}.")
        request_hidden_states = output_hidden_states or model_backbone in (LLAVA_QWEN2, QWEN2_VL)
        # CSM-VL always requests and returns these outputs for Qwen2-VL.
        request_attentions = output_attentions or model_backbone == QWEN2_VL
        hidden_states = self.encoder(
            **input, return_dict=True, output_hidden_states=request_hidden_states,
            output_attentions=request_attentions,
        )
        attention_matrix = getattr(hidden_states, "attentions", None)
        saved_hidden_states = (
            hidden_states.hidden_states
            if output_hidden_states or model_backbone == QWEN2_VL else None
        )
        if model_backbone == QWEN3_VL:
            last_hidden_state = hidden_states.last_hidden_state
            attention_mask = hidden_states.attention_mask
            image_features = None
        elif model_backbone == QWEN2_VL:
            last_hidden_state = hidden_states.hidden_states[-1]
            attention_mask = input["attention_mask"]
            image_features = getattr(hidden_states, "batch_image_embeds", None)
        else:
            last_hidden_state = self._extract_last_hidden_state(hidden_states)
            attention_mask = input["attention_mask"]
            image_features = getattr(hidden_states, "batch_image_embeds", None)
        # Keep reference pooling unchanged; FlagMRL can request raw pooled states
        # through a temporary context without mutating the model configuration.
        pooling_context = self
        if normalize is not None and normalize != self.normalize:
            pooling_context = SimpleNamespace(
                model_backbone=model_backbone, pooling=self.pooling, normalize=normalize,
            )
        pooled_output = MMEBModel._pooling(pooling_context, last_hidden_state, attention_mask)
        return pooled_output, image_features, attention_matrix, saved_hidden_states

    # Identical to VLM_Embed_Matryoshka's pooling, including backbone behavior.
    def _pooling(self, last_hidden_state, attention_mask):
        if self.pooling == 'last' or self.pooling == 'eos':
            if getattr(self, "model_backbone", None) in [QWEN3_VL]:
                # print("Applying pooling for Qwen3VL backbone")
                flipped_tensor = attention_mask.flip(dims=[1])
                last_one_positions = flipped_tensor.argmax(dim=1)
                col = attention_mask.shape[1] - last_one_positions - 1
                row = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
                return last_hidden_state[row, col]
            else:
                left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
                batch_size = last_hidden_state.shape[0]
                if left_padding:
                    # Get the vectors at the last position
                    reps = last_hidden_state[torch.arange(batch_size), -1, :]
                else:
                    # Calculate last 1 position in the original tensor
                    max_length = last_hidden_state.size(1)
                    invert_mask = (attention_mask == 0).long()
                    num_padding_tokens = invert_mask.sum(dim=1)
                    eos_indices_positive = max_length - num_padding_tokens - 1
                    # Get the vectors at the last 1 position of each attention mask
                    reps = last_hidden_state[
                        torch.arange(batch_size, device=last_hidden_state.device), eos_indices_positive]

        else:
            raise NotImplementedError
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    @classmethod
    def build(cls, model_args: ModelArguments, **kwargs):
        config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
        model_backbone = get_backbone_name(hf_config=config)
        model_args.model_backbone = model_backbone
        print_master(f'Loading backbone [{model_backbone}] from {model_args.model_name}')
        config._attn_implementation = "eager"
        if model_backbone in (QWEN2_VL, QWEN3_VL):
            config.padding_side = "right" if model_backbone == QWEN3_VL else "left"
            config.use_cache = False
        base_model = backbone2model[model_backbone].from_pretrained(
            model_args.model_name,
            config=config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

        if model_args.load_pretrained_lora:
            model_name_or_path = model_args.checkpoint_path if model_args.checkpoint_path else model_args.model_name
            print_master(f'Loading Pre-trained LoRA model from {model_name_or_path}')
            lora_config = LoraConfig.from_pretrained(model_name_or_path)

            lora_model = PeftModel.from_pretrained(base_model, 
                                                   model_name_or_path, 
                                                   config=lora_config,
                                                   is_trainable=True)
            # lora_model.config.modules_to_save = ['mm_projector']
            # lora_model.peft_config['default'].modules_to_save = ['mm_projector']
            projector_path = os.path.join(model_name_or_path, "mm_projector.pth")

            if os.path.exists(projector_path):
                lora_model.base_model.model.model.mm_projector.load_state_dict(
                    torch.load(projector_path)
                )
                print("Successfully loading the projector's weight")

            model = cls(
                encoder=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )
            setattr(model, "model_backbone", model_backbone)

            return model
        elif model_args.lora:
            print_master(f'Initializing LoRA adapter from {base_model}')
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=model_args.lora_target_modules.split(','),
                lora_dropout=model_args.lora_dropout,
                init_lora_weights="gaussian",
                use_dora=True,
                inference_mode=False
            )
            lora_model = get_peft_model(base_model, lora_config)

            model = cls(
                encoder=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )

            setattr(model, "model_backbone", model_backbone)
            return model
        
        model = cls(
            encoder=base_model,
            pooling=model_args.pooling,
            normalize=model_args.normalize,
            temperature=model_args.temperature
        )
        setattr(model, 'model_backbone', model_backbone)
        return model


    @classmethod
    def load(cls, model_args: ModelArguments, is_trainable=True, **kwargs):
        model_name_or_path = model_args.checkpoint_path or model_args.model_name
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        model_backbone = get_backbone_name(hf_config=config)
        model_args.model_backbone = model_backbone
        print_master(f'Loading backbone [{model_backbone}] from {model_name_or_path}')
        if model_backbone in (QWEN2_VL, QWEN3_VL):
            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
            config._attn_implementation = "eager"
            config.vision_config._attn_implementation = "eager"
            base_model = backbone2model[model_backbone].from_pretrained(
                model_args.model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                config=config,
            )
        else:
            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
            config.use_cache = False
            base_model = cls.TRANSFORMER_CLS.from_pretrained(
                model_name_or_path, **kwargs, config=config,
                torch_dtype=torch.bfloat16, trust_remote_code=True,
            )
            print(f"Loaded base model from HF: {model_name_or_path}")

        # Building the model on top of the base
        if model_args.lora:
            print_master(f'Loading LoRA from {model_name_or_path}')
            lora_config = LoraConfig.from_pretrained(model_name_or_path)
            lora_model = PeftModel.from_pretrained(base_model, model_name_or_path, config=lora_config, is_trainable=is_trainable)
            lora_model.load_adapter(model_name_or_path, lora_model.active_adapter, is_trainable=is_trainable)
            
            lora_model.merge_and_unload()  # Merge LoRA weights into the base model and unload the adapter to free up memory

            projector_path = os.path.join(model_name_or_path, "mm_projector.pth")

            if os.path.exists(projector_path):
                lora_model.base_model.model.model.mm_projector.load_state_dict(
                    torch.load(projector_path)
                )
                
                print("Successfully loading the projector's weight from local path")
            else:
                try: 
                    from huggingface_hub import hf_hub_download
                    projector_path = hf_hub_download(
                        repo_id=model_name_or_path,
                        filename="mm_projector.pth",
                    )
                    lora_model.base_model.model.model.mm_projector.load_state_dict(
                        torch.load(projector_path)
                    )
                except:
                    print("No projector weight found in the hub.")
                    pass
                print("Successfully loading the projector's weight")
                
            # lora_model = lora_model.merge_and_unload()

            model = cls(
                encoder=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )

        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )

        model.model_backbone = model_args.model_backbone
        return model

    def save(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)

    def forward(self, qry: Dict[str, Tensor] = None, tgt: Dict[str, Tensor] = None, *args, **kwargs):
        # print(f"qry keys: {qry.keys() if qry else None}, tgt keys: {tgt.keys() if tgt else None}")
        qry_reps = self.encode_input(qry, output_hidden_states=False, output_attentions=False)[0] if qry else None  # (bsz_per_device, dim)
        tgt_reps = self.encode_input(tgt, output_hidden_states=False, output_attentions=False)[0] if tgt else None # (bsz_per_device, dim)

        if qry_reps is None or tgt_reps is None:
            return {"qry_reps": qry_reps, "tgt_reps": tgt_reps}

        if self.is_ddp:
            all_qry_reps = self._dist_gather_tensor(qry_reps)
            all_tgt_reps = self._dist_gather_tensor(tgt_reps)
        else:
            all_qry_reps = qry_reps
            all_tgt_reps = tgt_reps

        scores = self.compute_similarity(all_qry_reps, all_tgt_reps)
        scores = scores.view(all_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_qry_reps.size(0) // all_tgt_reps.size(0))
        loss = self.cross_entropy(scores / self.temperature, target)
        if self.is_ddp:
            loss = loss * self.world_size

        return loss

    def _dist_gather_tensor(self, t: Tensor):
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))
