#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import re
import copy
from timm.models import create_model
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.init import normal_

from transformers import CLIPImageProcessor
from transformers import AutoConfig, AutoModelForCausalLM, Qwen2Config, Qwen2Model, Qwen2ForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from functools import partial
from typing import List, Tuple, Optional, Union, Dict, Any

from timm.models import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropPath, SqueezeExcite

CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15
LOGDIR = "."
# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

class LlavaConfig(Qwen2Config):
    model_type = "llava_qwen2"

def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 256, 256),
        "pool_size": None,
        "crop_pct": 0.95,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    "fastvit_t": _cfg(crop_pct=0.9),
    "fastvit_s": _cfg(crop_pct=0.9),
    "fastvit_m": _cfg(crop_pct=0.95),
}


class SEBlock(nn.Module):
    """Squeeze and Excite module.
    Pytorch implementation of `Squeeze-and-Excitation Networks` -
    https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self, in_channels: int, rd_ratio: float = 0.0625) -> None:
        """Construct a Squeeze and Excite Module.
        Args:
            in_channels: Number of input channels.
            rd_ratio: Input channel reduction ratio.
        """
        super(SEBlock, self).__init__()
        self.reduce = nn.Conv2d(
            in_channels=in_channels,
            out_channels=int(in_channels * rd_ratio),
            kernel_size=1,
            stride=1,
            bias=True,
        )
        self.expand = nn.Conv2d(
            in_channels=int(in_channels * rd_ratio),
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply forward pass."""
        b, c, h, w = inputs.size()
        # x = F.avg_pool2d(inputs, kernel_size=[h, w])
        x = F.avg_pool2d(inputs, kernel_size=[16, 16])
        x = self.reduce(x)
        x = F.relu(x)
        x = self.expand(x)
        x = torch.sigmoid(x)
        x = x.view(-1, c, 1, 1)
        return inputs * x


class MobileOneBlock(nn.Module):
    """MobileOne building block.
    This block has a multi-branched architecture at train-time
    and plain-CNN style architecture at inference time
    For more details, please refer to our paper:
    `An Improved One millisecond Mobile Backbone` -
    https://arxiv.org/pdf/2206.04040.pdf
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        inference_mode: bool = False,
        use_se: bool = False,
        use_act: bool = True,
        use_scale_branch: bool = True,
        num_conv_branches: int = 1,
        activation: nn.Module = nn.GELU(),
    ) -> None:
        """Construct a MobileOneBlock module.
        Args:
            in_channels: Number of channels in the input.
            out_channels: Number of channels produced by the block.
            kernel_size: Size of the convolution kernel.
            stride: Stride size.
            padding: Zero-padding size.
            dilation: Kernel dilation factor.
            groups: Group number.
            inference_mode: If True, instantiates model in inference mode.
            use_se: Whether to use SE-ReLU activations.
            use_act: Whether to use activation. Default: ``True``
            use_scale_branch: Whether to use scale branch. Default: ``True``
            num_conv_branches: Number of linear conv branches.
        """
        super(MobileOneBlock, self).__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches

        # Check if SE-ReLU is requested
        if use_se:
            self.se = SEBlock(out_channels)
        else:
            self.se = nn.Identity()

        if use_act:
            self.activation = activation
        else:
            self.activation = nn.Identity()

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
            )
        else:
            # Re-parameterizable skip connection
            # Fallback, sometimes batchnorm tensors
            # do not get instantiated correctly on some processes
            # when using deepspeed + accelerate
            norm_layer = nn.BatchNorm2d(num_features=in_channels)
            if norm_layer.weight.shape[0] == 0:
                norm_layer.weight = nn.Parameter(torch.zeros(in_channels))
            if norm_layer.bias.shape[0] == 0:
                norm_layer.bias = nn.Parameter(torch.zeros(in_channels))

            self.rbr_skip = (
                norm_layer
                if out_channels == in_channels and stride == 1
                else None
            )

            # Re-parameterizable conv branches
            if num_conv_branches > 0:
                rbr_conv = list()
                for _ in range(self.num_conv_branches):
                    rbr_conv.append(
                        self._conv_bn(kernel_size=kernel_size, padding=padding)
                    )
                self.rbr_conv = nn.ModuleList(rbr_conv)
            else:
                self.rbr_conv = None

            # Re-parameterizable scale branch
            self.rbr_scale = None
            if not isinstance(kernel_size, int):
                kernel_size = kernel_size[0]
            if (kernel_size > 1) and use_scale_branch:
                self.rbr_scale = self._conv_bn(kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward pass."""
        # Inference mode forward pass.
        if self.inference_mode:
            return self.activation(self.se(self.reparam_conv(x)))

        # Multi-branched train-time forward pass.
        # Skip branch output
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        # Scale branch output
        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        # Other branches
        out = scale_out + identity_out
        if self.rbr_conv is not None:
            for ix in range(self.num_conv_branches):
                out += self.rbr_conv[ix](x)

        return self.activation(self.se(out))

    def reparameterize(self):
        """Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.inference_mode:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=True,
        )
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # Delete un-used branches
        self.__delattr__("rbr_conv")
        self.__delattr__("rbr_scale")
        if hasattr(self, "rbr_skip"):
            self.__delattr__("rbr_skip")

        self.inference_mode = True

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83
        Returns:
            Tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale, [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        if self.rbr_conv is not None:
            for ix in range(self.num_conv_branches):
                _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[ix])
                kernel_conv += _kernel
                bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(
        self, branch: Union[nn.Sequential, nn.BatchNorm2d]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95
        Args:
            branch: Sequence of ops to be fused.
        Returns:
            Tuple of (kernel, bias) after fusing batchnorm.
        """
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups

                kernel_size = self.kernel_size
                if isinstance(self.kernel_size, int):
                    kernel_size = (self.kernel_size, self.kernel_size)

                kernel_value = torch.zeros(
                    (self.in_channels, input_dim, kernel_size[0], kernel_size[1]),
                    dtype=branch.weight.dtype,
                    device=branch.weight.device,
                )
                for i in range(self.in_channels):
                    kernel_value[
                        i, i % input_dim, kernel_size[0] // 2, kernel_size[1] // 2
                    ] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self, kernel_size: int, padding: int) -> nn.Sequential:
        """Helper method to construct conv-batchnorm layers.
        Args:
            kernel_size: Size of the convolution kernel.
            padding: Zero-padding size.
        Returns:
            Conv-BN module.
        """
        # Fallback, sometimes batchnorm tensors
        # do not get instantiated correctly on some processes
        # when using deepspeed + accelerate
        norm_layer = nn.BatchNorm2d(num_features=self.out_channels)
        if norm_layer.weight.shape[0] == 0:
            norm_layer.weight = nn.Parameter(torch.zeros(self.out_channels))
        if norm_layer.bias.shape[0] == 0:
            norm_layer.bias = nn.Parameter(torch.zeros(self.out_channels))

        mod_list = nn.Sequential()
        mod_list.add_module(
            "conv",
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                padding=padding,
                groups=self.groups,
                bias=False,
            ),
        )
        mod_list.add_module("bn", norm_layer)
        return mod_list


class ReparamLargeKernelConv(nn.Module):
    """Building Block of RepLKNet
    This class defines overparameterized large kernel conv block
    introduced in `RepLKNet <https://arxiv.org/abs/2203.06717>`_
    Reference: https://github.com/DingXiaoH/RepLKNet-pytorch
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int,
        small_kernel: int,
        inference_mode: bool = False,
        use_se: bool = False,
        activation: nn.Module = nn.GELU(),
    ) -> None:
        """Construct a ReparamLargeKernelConv module.
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size of the large kernel conv branch.
            stride: Stride size. Default: 1
            groups: Group number. Default: 1
            small_kernel: Kernel size of small kernel conv branch.
            inference_mode: If True, instantiates model in inference mode. Default: ``False``
            activation: Activation module. Default: ``nn.GELU``
        """
        super(ReparamLargeKernelConv, self).__init__()

        self.stride = stride
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation

        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        self.padding = kernel_size // 2

        # Check if SE is requested
        if use_se:
            self.se = SqueezeExcite(out_channels, rd_ratio=0.25)
        else:
            self.se = nn.Identity()

        if inference_mode:
            self.lkb_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=self.padding,
                dilation=1,
                groups=groups,
                bias=True,
            )
        else:
            self.lkb_origin = self._conv_bn(
                kernel_size=kernel_size, padding=self.padding
            )
            if small_kernel is not None:
                assert (
                    small_kernel <= kernel_size
                ), "The kernel size for re-param cannot be larger than the large kernel!"
                self.small_conv = self._conv_bn(
                    kernel_size=small_kernel, padding=small_kernel // 2
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward pass."""
        if hasattr(self, "lkb_reparam"):
            out = self.lkb_reparam(x)
        else:
            out = self.lkb_origin(x)
            if hasattr(self, "small_conv"):
                out += self.small_conv(x)

        return self.activation(self.se(out))

    def get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepLKNet-pytorch
        Returns:
            Tuple of (kernel, bias) after fusing branches.
        """
        eq_k, eq_b = self._fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, "small_conv"):
            small_k, small_b = self._fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            eq_k += nn.functional.pad(
                small_k, [(self.kernel_size - self.small_kernel) // 2] * 4
            )
        return eq_k, eq_b

    def reparameterize(self) -> None:
        """
        Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        eq_k, eq_b = self.get_kernel_bias()
        self.lkb_reparam = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.lkb_origin.conv.dilation,
            groups=self.groups,
            bias=True,
        )

        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__("lkb_origin")
        if hasattr(self, "small_conv"):
            self.__delattr__("small_conv")

    @staticmethod
    def _fuse_bn(
        conv: torch.Tensor, bn: nn.BatchNorm2d
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method to fuse batchnorm layer with conv layer.
        Args:
            conv: Convolutional kernel weights.
            bn: Batchnorm 2d layer.
        Returns:
            Tuple of (kernel, bias) after fusing batchnorm.
        """
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self, kernel_size: int, padding: int = 0) -> nn.Sequential:
        """Helper method to construct conv-batchnorm layers.
        Args:
            kernel_size: Size of the convolution kernel.
            padding: Zero-padding size.
        Returns:
            A nn.Sequential Conv-BN module.
        """
        # Fallback, sometimes batchnorm tensors
        # do not get instantiated correctly on some processes
        # when using deepspeed + accelerate
        norm_layer = nn.BatchNorm2d(num_features=self.out_channels)
        if norm_layer.weight.shape[0] == 0:
            norm_layer.weight = nn.Parameter(torch.zeros(self.out_channels))
        if norm_layer.bias.shape[0] == 0:
            norm_layer.bias = nn.Parameter(torch.zeros(self.out_channels))

        mod_list = nn.Sequential()
        mod_list.add_module(
            "conv",
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                padding=padding,
                groups=self.groups,
                bias=False,
            ),
        )
        mod_list.add_module("bn", norm_layer)
        return mod_list


def convolutional_stem(
    in_channels: int, out_channels: int, inference_mode: bool = False, use_scale_branch: bool = True,
) -> nn.Sequential:
    """Build convolutional stem with MobileOne blocks.
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        inference_mode: Flag to instantiate model in inference mode. Default: ``False``
    Returns:
        nn.Sequential object with stem elements.
    """
    return nn.Sequential(
        MobileOneBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            inference_mode=inference_mode,
            use_se=False,
            num_conv_branches=1,
            use_scale_branch=use_scale_branch
        ),
        MobileOneBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=out_channels,
            inference_mode=inference_mode,
            use_se=False,
            num_conv_branches=1,
            use_scale_branch=use_scale_branch
        ),
        MobileOneBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            inference_mode=inference_mode,
            use_se=False,
            num_conv_branches=1,
            use_scale_branch=use_scale_branch
        ),
    )


class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_features, eps=1e-05) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x


class MHSA(nn.Module):
    """Multi-headed Self Attention module.
    Source modified from:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """

    def __init__(
        self,
        dim: int,
        head_dim: int = 32,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """Build MHSA module that can handle 3D or 4D input tensors.
        Args:
            dim: Number of embedding dimensions.
            head_dim: Number of hidden dimensions per head. Default: ``32``
            qkv_bias: Use bias or not. Default: ``False``
            attn_drop: Dropout rate for attention tensor.
            proj_drop: Dropout rate for projection tensor.
        """
        super().__init__()
        assert dim % head_dim == 0, "dim should be divisible by head_dim"
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        B, C, H, W = shape
        N = H * W
        if len(shape) == 4:
            x = torch.flatten(x, start_dim=2).transpose(-2, -1)  # (B, N, C)
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # trick here to make q@k.t more stable
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if len(shape) == 4:
            x = x.transpose(-2, -1).reshape(B, C, H, W)

        return x


class PatchEmbed(nn.Module):
    """Convolutional patch embedding layer."""

    def __init__(
        self,
        patch_size: int,
        stride: int,
        in_channels: int,
        embed_dim: int,
        inference_mode: bool = False,
        use_se: bool = False,
    ) -> None:
        """Build patch embedding layer.
        Args:
            patch_size: Patch size for embedding computation.
            stride: Stride for convolutional embedding layer.
            in_channels: Number of channels of input tensor.
            embed_dim: Number of embedding dimensions.
            inference_mode: Flag to instantiate model in inference mode. Default: ``False``
            use_se: If ``True`` SE block will be used.
        """
        super().__init__()
        block = list()
        block.append(
            ReparamLargeKernelConv(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=stride,
                groups=in_channels,
                small_kernel=3,
                inference_mode=inference_mode,
                use_se=use_se,
            )
        )
        block.append(
            MobileOneBlock(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                inference_mode=inference_mode,
                use_se=False,
                num_conv_branches=1,
            )
        )
        self.proj = nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x


class RepMixer(nn.Module):
    """Reparameterizable token mixer.
    For more details, please refer to our paper:
    `FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization <https://arxiv.org/pdf/2303.14189.pdf>`_
    """

    def __init__(
        self,
        dim,
        kernel_size=3,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        inference_mode: bool = False,
    ):
        """Build RepMixer Module.
        Args:
            dim: Input feature map dimension. :math:`C_{in}` from an expected input of size :math:`(B, C_{in}, H, W)`.
            kernel_size: Kernel size for spatial mixing. Default: 3
            use_layer_scale: If True, learnable layer scale is used. Default: ``True``
            layer_scale_init_value: Initial value for layer scale. Default: 1e-5
            inference_mode: If True, instantiates model in inference mode. Default: ``False``
        """
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.inference_mode = inference_mode

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=self.dim,
                out_channels=self.dim,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                groups=self.dim,
                bias=True,
            )
        else:
            self.norm = MobileOneBlock(
                dim,
                dim,
                kernel_size,
                padding=kernel_size // 2,
                groups=dim,
                use_act=False,
                use_scale_branch=False,
                num_conv_branches=0,
            )
            self.mixer = MobileOneBlock(
                dim,
                dim,
                kernel_size,
                padding=kernel_size // 2,
                groups=dim,
                use_act=False,
            )
            self.use_layer_scale = use_layer_scale
            if use_layer_scale:
                self.layer_scale = nn.Parameter(
                    layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "reparam_conv"):
            x = self.reparam_conv(x)
            return x
        else:
            if self.use_layer_scale:
                x = x + self.layer_scale * (self.mixer(x) - self.norm(x))
            else:
                x = x + self.mixer(x) - self.norm(x)
            return x

    def reparameterize(self) -> None:
        """Reparameterize mixer and norm into a single
        convolutional layer for efficient inference.
        """
        if self.inference_mode:
            return

        self.mixer.reparameterize()
        self.norm.reparameterize()

        if self.use_layer_scale:
            w = self.mixer.id_tensor + self.layer_scale.unsqueeze(-1) * (
                self.mixer.reparam_conv.weight - self.norm.reparam_conv.weight
            )
            b = torch.squeeze(self.layer_scale) * (
                self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias
            )
        else:
            w = (
                self.mixer.id_tensor
                + self.mixer.reparam_conv.weight
                - self.norm.reparam_conv.weight
            )
            b = self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias

        self.reparam_conv = nn.Conv2d(
            in_channels=self.dim,
            out_channels=self.dim,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            groups=self.dim,
            bias=True,
        )
        self.reparam_conv.weight.data = w
        self.reparam_conv.bias.data = b

        self.__delattr__("mixer")
        self.__delattr__("norm")
        if self.use_layer_scale:
            self.__delattr__("layer_scale")


class ConvFFN(nn.Module):
    """Convolutional FFN Module."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        """Build convolutional FFN module.
        Args:
            in_channels: Number of input channels.
            hidden_channels: Number of channels after expansion. Default: None
            out_channels: Number of output channels. Default: None
            act_layer: Activation layer. Default: ``GELU``
            drop: Dropout rate. Default: ``0.0``.
        """
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.conv = nn.Sequential()
        self.conv.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                padding=3,
                groups=in_channels,
                bias=False,
            ),
        )

        # Fallback, sometimes batchnorm tensors
        # do not get instantiated correctly on some processes
        # when using deepspeed + accelerate
        norm_layer = nn.BatchNorm2d(num_features=out_channels)
        if norm_layer.weight.shape[0] == 0:
            norm_layer.weight = nn.Parameter(torch.zeros(out_channels))
        if norm_layer.bias.shape[0] == 0:
            norm_layer.bias = nn.Parameter(torch.zeros(out_channels))

        self.conv.add_module("bn", norm_layer)
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RepCPE(nn.Module):
    """Implementation of conditional positional encoding.
    For more details refer to paper:
    `Conditional Positional Encodings for Vision Transformers <https://arxiv.org/pdf/2102.10882.pdf>`_
    In our implementation, we can reparameterize this module to eliminate a skip connection.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 768,
        spatial_shape: Union[int, Tuple[int, int]] = (7, 7),
        inference_mode=False,
    ) -> None:
        """Build reparameterizable conditional positional encoding
        Args:
            in_channels: Number of input channels.
            embed_dim: Number of embedding dimensions. Default: 768
            spatial_shape: Spatial shape of kernel for positional encoding. Default: (7, 7)
            inference_mode: Flag to instantiate block in inference mode. Default: ``False``
        """
        super(RepCPE, self).__init__()
        if isinstance(spatial_shape, int):
            spatial_shape = tuple([spatial_shape] * 2)
        assert isinstance(spatial_shape, Tuple), (
            f'"spatial_shape" must by a sequence or int, '
            f"get {type(spatial_shape)} instead."
        )
        assert len(spatial_shape) == 2, (
            f'Length of "spatial_shape" should be 2, '
            f"got {len(spatial_shape)} instead."
        )

        self.spatial_shape = spatial_shape
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.groups = embed_dim

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.embed_dim,
                kernel_size=self.spatial_shape,
                stride=1,
                padding=int(self.spatial_shape[0] // 2),
                groups=self.embed_dim,
                bias=True,
            )
        else:
            self.pe = nn.Conv2d(
                in_channels,
                embed_dim,
                spatial_shape,
                1,
                int(spatial_shape[0] // 2),
                bias=True,
                groups=embed_dim,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "reparam_conv"):
            x = self.reparam_conv(x)
            return x
        else:
            x = self.pe(x) + x
            return x

    def reparameterize(self) -> None:
        # Build equivalent Id tensor
        input_dim = self.in_channels // self.groups
        kernel_value = torch.zeros(
            (
                self.in_channels,
                input_dim,
                self.spatial_shape[0],
                self.spatial_shape[1],
            ),
            dtype=self.pe.weight.dtype,
            device=self.pe.weight.device,
        )
        for i in range(self.in_channels):
            kernel_value[
                i,
                i % input_dim,
                self.spatial_shape[0] // 2,
                self.spatial_shape[1] // 2,
            ] = 1
        id_tensor = kernel_value

        # Reparameterize Id tensor and conv
        w_final = id_tensor + self.pe.weight
        b_final = self.pe.bias

        # Introduce reparam conv
        self.reparam_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.embed_dim,
            kernel_size=self.spatial_shape,
            stride=1,
            padding=int(self.spatial_shape[0] // 2),
            groups=self.embed_dim,
            bias=True,
        )
        self.reparam_conv.weight.data = w_final
        self.reparam_conv.bias.data = b_final

        self.__delattr__("pe")


class RepMixerBlock(nn.Module):
    """Implementation of Metaformer block with RepMixer as token mixer.
    For more details on Metaformer structure, please refer to:
    `MetaFormer Is Actually What You Need for Vision <https://arxiv.org/pdf/2111.11418.pdf>`_
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        mlp_ratio: float = 4.0,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        inference_mode: bool = False,
    ):
        """Build RepMixer Block.
        Args:
            dim: Number of embedding dimensions.
            kernel_size: Kernel size for repmixer. Default: 3
            mlp_ratio: MLP expansion ratio. Default: 4.0
            act_layer: Activation layer. Default: ``nn.GELU``
            drop: Dropout rate. Default: 0.0
            drop_path: Drop path rate. Default: 0.0
            use_layer_scale: Flag to turn on layer scale. Default: ``True``
            layer_scale_init_value: Layer scale value at initialization. Default: 1e-5
            inference_mode: Flag to instantiate block in inference mode. Default: ``False``
        """

        super().__init__()

        self.token_mixer = RepMixer(
            dim,
            kernel_size=kernel_size,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            inference_mode=inference_mode,
        )

        assert mlp_ratio > 0, "MLP ratio should be greater than 0, found: {}".format(
            mlp_ratio
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(
            in_channels=dim,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        # Drop Path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Layer Scale
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
            )

    def forward(self, x):
        if self.use_layer_scale:
            x = self.token_mixer(x)
            x = x + self.drop_path(self.layer_scale * self.convffn(x))
        else:
            x = self.token_mixer(x)
            x = x + self.drop_path(self.convffn(x))
        return x


class AttentionBlock(nn.Module):
    """Implementation of metaformer block with MHSA as token mixer.
    For more details on Metaformer structure, please refer to:
    `MetaFormer Is Actually What You Need for Vision <https://arxiv.org/pdf/2111.11418.pdf>`_
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.BatchNorm2d,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
    ):
        """Build Attention Block.
        Args:
            dim: Number of embedding dimensions.
            mlp_ratio: MLP expansion ratio. Default: 4.0
            act_layer: Activation layer. Default: ``nn.GELU``
            norm_layer: Normalization layer. Default: ``nn.BatchNorm2d``
            drop: Dropout rate. Default: 0.0
            drop_path: Drop path rate. Default: 0.0
            use_layer_scale: Flag to turn on layer scale. Default: ``True``
            layer_scale_init_value: Layer scale value at initialization. Default: 1e-5
        """

        super().__init__()

        # Fallback, sometimes batchnorm tensors
        # do not get instantiated correctly on some processes
        # when using deepspeed + accelerate
        norm_layer_ = norm_layer(num_features=dim)
        if norm_layer_.weight.shape[0] == 0:
            norm_layer_.weight = nn.Parameter(torch.zeros(dim))
        if norm_layer_.bias.shape[0] == 0:
            norm_layer_.bias = nn.Parameter(torch.zeros(dim))

        self.norm = norm_layer_
        self.token_mixer = MHSA(dim=dim)

        assert mlp_ratio > 0, "MLP ratio should be greater than 0, found: {}".format(
            mlp_ratio
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(
            in_channels=dim,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Layer Scale
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
            )
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
            )

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.token_mixer(self.norm(x)))
            x = x + self.drop_path(self.layer_scale_2 * self.convffn(x))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm(x)))
            x = x + self.drop_path(self.convffn(x))
        return x


def basic_blocks(
    dim: int,
    block_index: int,
    num_blocks: List[int],
    token_mixer_type: str,
    kernel_size: int = 3,
    mlp_ratio: float = 4.0,
    act_layer: nn.Module = nn.GELU,
    norm_layer: nn.Module = nn.BatchNorm2d,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    use_layer_scale: bool = True,
    layer_scale_init_value: float = 1e-5,
    inference_mode=False,
) -> nn.Sequential:
    """Build FastViT blocks within a stage.
    Args:
        dim: Number of embedding dimensions.
        block_index: block index.
        num_blocks: List containing number of blocks per stage.
        token_mixer_type: Token mixer type.
        kernel_size: Kernel size for repmixer.
        mlp_ratio: MLP expansion ratio.
        act_layer: Activation layer.
        norm_layer: Normalization layer.
        drop_rate: Dropout rate.
        drop_path_rate: Drop path rate.
        use_layer_scale: Flag to turn on layer scale regularization.
        layer_scale_init_value: Layer scale value at initialization.
        inference_mode: Flag to instantiate block in inference mode.
    Returns:
        nn.Sequential object of all the blocks within the stage.
    """
    blocks = []
    for block_idx in range(num_blocks[block_index]):
        block_dpr = (
            drop_path_rate
            * (block_idx + sum(num_blocks[:block_index]))
            / (sum(num_blocks) - 1)
        )
        if token_mixer_type == "repmixer":
            blocks.append(
                RepMixerBlock(
                    dim,
                    kernel_size=kernel_size,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    drop=drop_rate,
                    drop_path=block_dpr,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                    inference_mode=inference_mode,
                )
            )
        elif token_mixer_type == "attention":
            blocks.append(
                AttentionBlock(
                    dim,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    drop=drop_rate,
                    drop_path=block_dpr,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                )
            )
        else:
            raise ValueError(
                "Token mixer type: {} not supported".format(token_mixer_type)
            )
    blocks = nn.Sequential(*blocks)
    return blocks


class GlobalPool2D(nn.Module):
    """This class implements global pooling with linear projection."""

    def __init__(self, in_dim: int, out_dim: int, *args, **kwargs) -> None:
        super().__init__()
        scale = in_dim**-0.5
        self.proj = nn.Parameter(scale * torch.randn(size=(in_dim, out_dim)))
        self.in_dim = in_dim
        self.out_dim = out_dim

    def pool(self, x) -> Tensor:
        if x.dim() == 4:
            dims = [-2, -1]
        elif x.dim() == 5:
            dims = [-3, -2, -1]
        x = torch.mean(x, dim=dims, keepdim=False)
        return x

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        # x is of shape [batch, in_dim]
        assert (
            x.dim() == 4
        ), "Input should be 4-dimensional (Batch x in_dim x in_height x in_width). Got: {}".format(
            x.shape
        )

        # [batch, in_dim, in_height, in_width] --> [batch, in_dim]
        x = self.pool(x)
        # [batch, in_dim]  x [in_dim, out_dim] --> [batch, out_dim]
        x = x @ self.proj
        return x


class FastViT(nn.Module):
    """
    This class implements `FastViT architecture <https://arxiv.org/pdf/2303.14189.pdf>`_
    """

    def __init__(
        self,
        layers,
        token_mixers: Tuple[str, ...],
        embed_dims=None,
        mlp_ratios=None,
        downsamples=None,
        se_downsamples=None,
        repmixer_kernel_size=3,
        norm_layer: nn.Module = nn.BatchNorm2d,
        act_layer: nn.Module = nn.GELU,
        num_classes=1000,
        pos_embs=None,
        down_patch_size=7,
        down_stride=2,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        init_cfg=None,
        pretrained=None,
        cls_ratio=2.0,
        inference_mode=False,
        stem_scale_branch=True,
        **kwargs,
    ) -> None:

        super().__init__()

        self.num_classes = num_classes
        if len(layers) == 4:
            self.out_indices = [0, 2, 4, 7]
        elif len(layers) == 5:
            self.out_indices = [0, 2, 4, 7, 10]
        else:
            raise NotImplementedError("FPN is not implemented for more than 5 stages.")

        if pos_embs is None:
            pos_embs = [None] * len(layers)

        if se_downsamples is None:
            se_downsamples = [False] * len(layers)

        # Convolutional stem
        self.patch_embed = convolutional_stem(3, embed_dims[0], inference_mode,
                                              use_scale_branch=stem_scale_branch)

        # Build the main stages of the network architecture
        network = []
        for i in range(len(layers)):
            # Add position embeddings if requested
            if pos_embs[i] is not None:
                network.append(
                    pos_embs[i](
                        embed_dims[i], embed_dims[i], inference_mode=inference_mode
                    )
                )
            stage = basic_blocks(
                embed_dims[i],
                i,
                layers,
                token_mixer_type=token_mixers[i],
                kernel_size=repmixer_kernel_size,
                mlp_ratio=mlp_ratios[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                inference_mode=inference_mode,
            )
            network.append(stage)
            if i >= len(layers) - 1:
                break

            # Patch merging/downsampling between stages.
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                network.append(
                    PatchEmbed(
                        patch_size=down_patch_size,
                        stride=down_stride,
                        in_channels=embed_dims[i],
                        embed_dim=embed_dims[i + 1],
                        inference_mode=inference_mode,
                        use_se=se_downsamples[i + 1],
                    )
                )
        self.network = nn.ModuleList(network)

        # Classifier head
        self.conv_exp = MobileOneBlock(
            in_channels=embed_dims[-1],
            out_channels=int(embed_dims[-1] * cls_ratio),
            kernel_size=3,
            stride=1,
            padding=1,
            groups=embed_dims[-1],
            inference_mode=inference_mode,
            use_se=True,
            num_conv_branches=1,
        )
        self.head = (
            nn.Linear(int(embed_dims[-1] * cls_ratio), num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.apply(self.cls_init_weights)
        self.init_cfg = copy.deepcopy(init_cfg)

    def cls_init_weights(self, m: nn.Module) -> None:
        """Init. for classification"""
        if isinstance(m, nn.Linear):
            normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        for idx, block in enumerate(self.network):
            x = block(x)
        return x

    def forward(self, x: torch.Tensor, *args, **kwargs) -> Union[Tensor, Dict[str, Tensor]]:
        # input embedding
        x = self.forward_embeddings(x)
        # through backbone
        x = self.forward_tokens(x)
        # for image classification/embedding
        x = self.conv_exp(x)
        cls_out = self.head(x)

        out_dict = dict()
        if kwargs.get("return_image_embeddings", False):
            out_dict.update({"logits": cls_out})
            out_dict.update({"image_embeddings": x})
            return out_dict
        else:
            return cls_out


@register_model
def fastvithd(pretrained=False, **kwargs):
    """Instantiate FastViTHD model variant."""
    layers = [2, 12, 24, 4, 2]
    embed_dims = [96, 192, 384, 768, 1536]
    mlp_ratios = [4, 4, 4, 4, 4]
    downsamples = [True, True, True, True, True]
    pos_embs = [None, None, None, partial(RepCPE, spatial_shape=(7, 7)), partial(RepCPE, spatial_shape=(7, 7))]
    token_mixers = ("repmixer", "repmixer", "repmixer", "attention", "attention")
    model = FastViT(
        layers,
        token_mixers=token_mixers,
        embed_dims=embed_dims,
        pos_embs=pos_embs,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        norm_layer=LayerNormChannel,
        stem_scale_branch=False,
        inference_mode=True,
        **kwargs,
    )
    model.default_cfg = default_cfgs["fastvit_m"]
    if pretrained:
        raise ValueError("Functionality not implemented.")
    return model

def load_model_config(
        model_name: str,
) -> Any:
    model_cfg = {
        "embed_dim": 768,
        "image_cfg": {
            "image_size": 1024,
            "model_name": "fastvithd",
            "embed_dim": 3072,
            "patch_size": 64
        },
        "text_cfg": {
            "context_length": 77,
            "vocab_size": 49408,
            "dim": 768,
            "ffn_multiplier_per_layer": 4.0,
            "n_heads_per_layer": 12,
            "n_transformer_layers": 12,
            "norm_layer": "layer_norm_fp32",
            "causal_masking": False,
            "model_name": "base"
        }
    }
    return model_cfg


class MCi(nn.Module):
    """
    This class implements `MCi Models <https://arxiv.org/pdf/2311.17049.pdf>`_
    """

    def __init__(self, model_name: str, *args, **kwargs) -> None:
        super().__init__()
        self.projection_dim = None
        if "projection_dim" in kwargs:
            self.projection_dim = kwargs.get("projection_dim")

        # Create model
        self.model = create_model(model_name, projection_dim=self.projection_dim)

        # Build out projection head.
        if self.projection_dim is not None:
            if hasattr(self.model, "head"):
                self.model.head = MCi._update_image_classifier(
                    image_classifier=self.model.head, projection_dim=self.projection_dim
                )

    def forward(self, x: Any, *args, **kwargs) -> Any:
        """A forward function of the model."""
        x = self.model(x, *args, **kwargs)
        return x

    @staticmethod
    def _get_in_feature_dimension(image_classifier: nn.Module) -> int:
        """Return the input feature dimension to the image classification head."""
        in_features = None
        if isinstance(image_classifier, nn.Sequential):
            # Classifier that uses nn.Sequential usually has global pooling and
            # multiple linear layers. Find the first linear layer and get its
            # in_features
            for layer in image_classifier:
                if isinstance(layer, nn.Linear):
                    in_features = layer.in_features
                    break
        elif isinstance(image_classifier, nn.Linear):
            in_features = image_classifier.in_features

        if in_features is None:
            raise NotImplementedError(
                f"Cannot get input feature dimension of {image_classifier}."
            )
        return in_features

    @staticmethod
    def _update_image_classifier(
        image_classifier: nn.Module, projection_dim: int, *args, **kwargs
    ) -> nn.Module:
        in_features = MCi._get_in_feature_dimension(image_classifier)
        new_img_classifier = GlobalPool2D(in_dim=in_features, out_dim=projection_dim)
        return new_img_classifier


class MobileCLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.tune_vision_tower = getattr(args, 'unfreeze_mm_vision_tower', False)
        self.input_image_size = int(vision_tower.split("_")[-1])

        # Delay load is disabled for now
        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            model_cfg = load_model_config(self.vision_tower_name)
            self.cfg_only = model_cfg

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        # Load model config
        model_cfg = load_model_config(self.vision_tower_name)

        # Override default image resolution
        model_cfg["image_cfg"]["image_size"] = self.input_image_size

        self.cfg_only = model_cfg

        # Build HF CLIPImageProcessor with MobileCLIP parameters
        self.image_processor = CLIPImageProcessor(crop_size={"height": model_cfg["image_cfg"]["image_size"],
                                                             "width": model_cfg["image_cfg"]["image_size"]},
                                                  image_mean=[0.0, 0.0, 0.0],
                                                  image_std=[1.0, 1.0, 1.0],
                                                  size={"shortest_edge": model_cfg["image_cfg"]["image_size"]})

        # Instantiate the image encoder
        self.vision_tower = MCi(model_name=model_cfg["image_cfg"]["model_name"],
                                           projection_dim=model_cfg["embed_dim"])

        if not self.tune_vision_tower:
            self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        # Features from penultimate layer
        image_features = image_forward_outs["image_embeddings"]

        # Reshape 4D tensor to 3D
        B, C, H, W = image_features.shape
        image_features = image_features.reshape(B, C, H*W)
        image_features = image_features.transpose(1, 2)
        return image_features

    def forward(self, images):
        if self.tune_vision_tower:
            return self.forward_images(images)
        else:
            with torch.no_grad():
                return self.forward_images(images)

    def forward_images(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), return_image_embeddings=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), return_image_embeddings=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device

    @property
    def config(self):
        return self.cfg_only

    @property
    def hidden_size(self):
        return self.config["image_cfg"]["embed_dim"]

    @property
    def num_patches_per_side(self):
        return self.config["image_cfg"]["image_size"] // self.config["image_cfg"]["patch_size"]

    @property
    def num_patches(self):
        return (self.config["image_cfg"]["image_size"] // self.config["image_cfg"]["patch_size"]) ** 2

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}

def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    return MobileCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.
    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].
    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float('inf')

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit

def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.
    Args:
        image_size (tuple): The size of the input image in the format (width, height).
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        patch_size (int): The size of each image patch.
    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    import ast
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size

class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            if hasattr(self.get_vision_tower(), 's2_image_size'):
                                img_size = self.get_vision_tower().s2_image_size
                            elif isinstance(self.get_vision_tower().config, dict):
                                img_size = self.get_vision_tower().config["image_cfg"]["image_size"]
                            else:
                                img_size = self.get_vision_tower().config.image_size

                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, img_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False


class LlavaQwen2Model(LlavaMetaModel, Qwen2Model):
    config_class = LlavaConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwen2Model, self).__init__(config)


class LlavaQwen2ForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = LlavaQwen2Model(config)
        # self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs


AutoConfig.register("llava_qwen2", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaQwen2ForCausalLM)