"""FlagMRL projection and checkpoint APIs on top of the vlm_embed model."""

import json
from pathlib import Path

import torch
from torch.nn import functional as F

from src.model.reference_model import MMEBModel as ReferenceMMEBModel


class MMEBModel(ReferenceMMEBModel):
    @property
    def full_dim(self):
        config = self.encoder.config
        dim = getattr(config, "hidden_size", None)
        if dim is None:
            dim = getattr(getattr(config, "text_config", None), "hidden_size", None)
        if dim is None:
            raise ValueError("Encoder config has no hidden_size or text_config.hidden_size.")
        return int(dim)

    def encode_at_dim(self, inputs, dim):
        return self.project_at_dim(self.encode_input(inputs)[0], dim)

    def project_at_dim(self, reps, dim):
        if not 0 < dim <= self.full_dim:
            raise ValueError(f"Invalid inference dim: {dim}")
        with torch.autocast(device_type=reps.device.type, enabled=False):
            if dim == self.full_dim:
                return F.normalize(reps.float(), dim=-1)
            if hasattr(self, "flag_projector"):
                projector = self.flag_projector
                if dim not in projector.mrl_dims:
                    raise ValueError(f"dim={dim} is not a trained MRL level: {projector.mrl_dims}")
                return projector.project_at_dim(reps, dim)
            return F.normalize(reps.float()[..., :dim], dim=-1)

    @classmethod
    def load(cls, model_args, is_trainable=False, **kwargs):
        model = super().load(model_args, is_trainable=is_trainable, **kwargs)
        source = Path(model_args.checkpoint_path or model_args.model_name)
        dtype_path = source / "encoder_dtypes.json"
        if dtype_path.exists():
            saved_dtypes = json.loads(dtype_path.read_text())
            allowed = {"torch.float32": torch.float32, "torch.bfloat16": torch.bfloat16, "torch.float16": torch.float16}
            for name, parameter in model.encoder.named_parameters():
                if name in saved_dtypes:
                    parameter.data = parameter.data.to(allowed[saved_dtypes[name]])
        flag_path = source / "flag_U.pt"
        if flag_path.exists():
            from src.MRL.flag.flag_projector import FlagProjector
            state = torch.load(flag_path, map_location="cpu", weights_only=True)
            model.flag_projector = FlagProjector(
                model.full_dim, state["q_dims"], init="identity")
            model.flag_projector.load(flag_path)
        if not is_trainable:
            model.requires_grad_(False)
            model.eval()
        return model

    def save(self, output_dir):
        super().save(output_dir)
        # Same separate image-projector artifact as vlm_embed's trainer.
        if self.model_backbone == "llava_qwen2":
            torch.save(self.encoder.model.model.mm_projector.state_dict(),
                       Path(output_dir) / "mm_projector.pth")
        Path(output_dir, "encoder_dtypes.json").write_text(json.dumps(
            {name: str(parameter.dtype) for name, parameter in self.encoder.named_parameters()}, indent=2))
        self.encoder.config.save_pretrained(output_dir)
        if hasattr(self, "flag_projector"):
            self.flag_projector.save(Path(output_dir) / "flag_U.pt")
