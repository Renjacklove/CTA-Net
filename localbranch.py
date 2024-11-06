import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional
from ODStar import ODConv2d,DropPath

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Local_block(nn.Module):
    """
    Local Feature Block with ODConv and selective gating mechanism.

    Args:
        dim (int): Number of input channels.
        drop_rate (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_rate=0.):
        super().__init__()
        # ODConv with depthwise convolution
        self.dwconv = ODConv2d(dim, dim, kernel_size=3, padding=1, groups=dim, reduction=0.0625, kernel_num=4)

        # Two branches for selective gating
        self.branch1 = nn.Conv2d(dim, dim, kernel_size=1)  # pointwise conv for branch 1
        self.branch2 = nn.Conv2d(dim, dim, kernel_size=1)  # pointwise conv for branch 2

        # Activation function
        self.act = nn.GELU()

        # Normalization layer
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")

        # DropPath for stochastic depth
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection
        shortcut = x

        # ODConv depthwise convolution
        x = self.dwconv(x)

        # Apply two branches for selective gating
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        # Gating with element-wise multiplication
        x = self.act(x1) * x2  # selective gating mechanism

        # Permute to (N, H, W, C) for normalization
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # back to (N, C, H, W)

        # Residual connection with DropPath
        x = shortcut + self.drop_path(x)

        return x


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True, group=1):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x