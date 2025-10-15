import torch
from torch import nn
import torch.nn.functional as F
import einops


class ResNetBlock(nn.Module):
    """
    Defines a ResNetBlock module used for residual learning in convolutional neural
    networks.

    This class represents a residual block in a ResNet-style architecture, designed
    to process input tensors and produce output tensors with similar or modified
    channel dimensions. It employs normalization, nonlinear activation functions,
    and convolutions to achieve effective feature extraction. For cases where the
    input and output channel dimensions differ, a shortcut connection is created
    using a 1x1 convolution; otherwise, an identity mapping is used.

    Attributes:
        in_channels (int): Number of input channels for the block.
        out_channels (int): Number of output channels for the block.
        norm1 (nn.GroupNorm): First group normalization layer applied to the input.
        conv1 (nn.Conv2d): First convolutional layer with kernel size 3x3.
        norm2 (nn.GroupNorm): Second group normalization layer applied after the
            first convolution.
        conv2 (nn.Conv2d): Second convolutional layer with kernel size 3x3.
        nin_shortcut (Union[nn.Conv2d, nn.Identity]): Shortcut connection; uses
            either a 1x1 convolution if in_channels and out_channels differ, or
            identity mapping otherwise.
    """

    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return self.nin_shortcut(x) + h


# class AttentionBlock(nn.Module):
#     """
#     Implements a multi-head attention mechanism using torch.nn.MultiheadAttention.
#
#     This block reshapes the input tensor from (B, C, H, W) to the sequence format
#     expected by MultiheadAttention, performs self-attention, and then reshapes it
#     back. This allows capturing long-range dependencies across spatial dimensions.
#
#     Attributes:
#         norm (nn.GroupNorm): Group normalization layer applied to the input.
#         attn (nn.MultiheadAttention): The core multi-head attention layer from PyTorch.
#     """
#
#     def __init__(self, channels: int, num_heads: int = 8, num_groups: int = 32):
#         super().__init__()
#         self.norm = nn.GroupNorm(num_groups, channels)
#         # Ensure channel dimension is divisible by num_heads
#         if channels % num_heads != 0:
#             raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")
#
#         self.attn = nn.MultiheadAttention(
#             embed_dim=channels,
#             num_heads=num_heads,
#             batch_first=False  # Expects (SeqLen, Batch, EmbedDim)
#         )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         b, c, h, w = x.shape
#
#         # Normalize and reshape for attention
#         h_ = self.norm(x)
#         h_ = h_.reshape(b, c, h * w).permute(2, 0, 1)  # (H*W, B, C)
#
#         # Apply self-attention
#         attn_output, _ = self.attn(h_, h_, h_)
#
#         # Reshape back to image format and add residual connection
#         attn_output = attn_output.permute(1, 2, 0).reshape(b, c, h, w)
#
#         return x + attn_output


class AttentionBlock(nn.Module):
    """
    Defines an AttentionBlock for self-attention in neural networks.

    This class implements a self-attention mechanism with support for
    multi-head attention. It normalizes input feature maps using group normalization
    and computes the attention scores using learnable query, key, and value projections.
    The attention mechanism is scaled using the dot product attention function and
    can process inputs with multi-heads for better representation learning. It is especially
    useful in scenarios requiring spatial or channel-wise attention.
    """
    def __init__(self, channels: int, num_heads: int = 1, num_groups: int = 32):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels)
        self.q = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(x)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)

        b, c, h, w = q.shape
        q, k, v = map(
            lambda t: einops.rearrange(t, "b c h w -> b (h w) c"), (q, k, v)
        )

        # Use num_heads by reshaping the channel dimension
        q = einops.rearrange(q, "b l (h d) -> (b h) l d", h=self.num_heads)
        k = einops.rearrange(k, "b l (h d) -> (b h) l d", h=self.num_heads)
        v = einops.rearrange(v, "b l (h d) -> (b h) l d", h=self.num_heads)

        h_ = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        h_ = einops.rearrange(h_, "(b h) l d -> b l (h d)", h=self.num_heads)
        h_ = einops.rearrange(h_, "b (h w) c -> b c h w", h=h, w=w)

        h_ = self.proj_out(h_)
        return x + h_


class Downsample(nn.Module):
    """
    Implements a downsampling layer.

    This class defines a layer that performs downsampling on a given input
    tensor by applying a convolution operation. The convolution has a
    kernel size of 3x3, a stride of 2, and padding of 1. The primary purpose
    of this layer is to reduce the spatial dimensions of the input tensor
    while maintaining its channel depth. The layer is commonly used in
    convolutional neural networks for feature extraction and dimensionality
    reduction.

    Attributes:
        conv (nn.Conv2d): A 2D convolution operation with fixed kernel size,
            stride, and padding for performing downsampling.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """
    Upsample is a neural network module for upsampling feature maps.

    This class provides functionality for upsampling a given input tensor
    using nearest-neighbor interpolation followed by a convolution operation.
    It is typically used for increasing the spatial resolution of feature maps,
    for example, in image processing or generative models.

    Attributes:
        conv (nn.Conv2d): A convolutional layer with a kernel size of 3, stride
            of 1, and padding of 1 configured to process input tensors with the
            same number of channels as initialized.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)