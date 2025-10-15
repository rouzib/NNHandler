from typing import List

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .blocks import AttentionBlock, ResNetBlock, Downsample, Upsample


class Encoder(nn.Module):
    """
    Encoder module for encoding input tensors into latent representations.

    This class defines an encoder architecture using convolutional layers,
    residual blocks, and attention mechanisms to process input tensors and
    generate latent representations suitable for downstream tasks such as
    generation, compression, or feature extraction. The encoder uses a
    multi-scale architecture with downsampling and a bottleneck structure
    for hierarchical feature extraction.

    Attributes:
        conv_in (torch.nn.Conv2d): Initial convolutional layer for input
            processing.
        down_blocks (torch.nn.ModuleList): A list of module blocks that
            perform downsampling and residual processing at multiple scales.
            Includes attention blocks and downsampling operations.
        bottleneck (torch.nn.Sequential): The bottleneck layer consisting
            of residual blocks, attention blocks, normalization, and a final
            convolutional layer to map to the latent space.
    """

    def __init__(self, in_channels: int, base_channels: int, channel_multipliers: List[int], num_res_blocks: int,
                 z_channels: int, num_groups: int, num_heads: int, attention_layers: List[bool],
                 use_checkpointing: bool):
        super().__init__()

        if len(attention_layers) != len(channel_multipliers):
            raise ValueError("Length of attention_layers must match length of channel_multipliers")

        self.use_checkpointing = use_checkpointing
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)

        ch = base_channels
        self.down_blocks = nn.ModuleList()
        for i, mult in enumerate(channel_multipliers):
            block_group = nn.ModuleList()
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                block_group.append(ResNetBlock(ch, out_ch, num_groups=num_groups))
                ch = out_ch

            if attention_layers[i]:
                block_group.append(AttentionBlock(ch, num_heads=num_heads, num_groups=num_groups))

            downsample_layer = None
            if i != len(channel_multipliers) - 1:
                downsample_layer = Downsample(ch)

            self.down_blocks.append(nn.ModuleDict({
                'block_group': block_group,
                'downsample': downsample_layer
            }))

        self.bottleneck = nn.Sequential(
            ResNetBlock(ch, ch, num_groups=num_groups),
            AttentionBlock(ch, num_heads=num_heads, num_groups=num_groups),
            ResNetBlock(ch, ch, num_groups=num_groups)
        )

        self.final_conv = nn.Sequential(
            nn.GroupNorm(num_groups, ch),
            nn.SiLU(),
            nn.Conv2d(ch, 2 * z_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)

        for stage in self.down_blocks:
            for block in stage['block_group']:
                # Conditional Gradient Checkpointing
                if self.use_checkpointing and self.training:
                    h = checkpoint(block, h, use_reentrant=False)
                else:
                    h = block(h)

            if stage['downsample'] is not None:
                h = stage['downsample'](h)

        for block in self.bottleneck:
            if self.use_checkpointing and self.training:
                h = checkpoint(block, h, use_reentrant=False)
            else:
                h = block(h)

        h = self.final_conv(h)
        return h


class Decoder(nn.Module):
    """
    Decoder module for processing feature maps in a neural network.

    This class implements a convolutional-decoder architecture with ResNet blocks,
    attention mechanisms, and upsampling stages. The decoder processes input feature
    maps (`z`) through a sequence of operations, recovering spatial resolution and
    producing an output tensor with specified channels. It is designed to handle
    multi-scale information through its hierarchical structure and is suitable for
    tasks like image reconstruction or feature decoding in generative models.

    Attributes:
        conv_in (torch.nn.Conv2d): Initial convolutional layer that processes input feature map.
        bottleneck (torch.nn.Sequential): Bottleneck module with ResNet and Attention blocks for deeper feature
            extraction.
        up_blocks (torch.nn.ModuleList): List of upsampling blocks with ResNet, Attention, and Upsample layers,
            arranged in a multi-scale structure.
        conv_out (torch.nn.Sequential): Final output module composed of Group Normalization, SiLU activation,
            and a convolution layer.
    """

    def __init__(self, out_channels: int, base_channels: int, channel_multipliers: List[int], num_res_blocks: int,
                 z_channels: int, num_groups: int, num_heads: int, attention_layers: List[bool],
                 use_checkpointing: bool):
        super().__init__()

        if len(attention_layers) != len(channel_multipliers):
            raise ValueError("Length of attention_layers must match length of channel_multipliers")

        self.use_checkpointing = use_checkpointing
        ch = base_channels * channel_multipliers[-1]

        self.conv_in = nn.Conv2d(z_channels, ch, kernel_size=3, stride=1, padding=1)

        self.bottleneck = nn.Sequential(
            ResNetBlock(ch, ch, num_groups=num_groups),
            AttentionBlock(ch, num_heads=num_heads, num_groups=num_groups),
            ResNetBlock(ch, ch, num_groups=num_groups)
        )

        self.up_blocks = nn.ModuleList()
        # Reversed loop for building the decoder stages
        for i, mult in reversed(list(enumerate(channel_multipliers))):
            block_group = nn.ModuleList()
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                block_group.append(ResNetBlock(ch, out_ch, num_groups=num_groups))
                ch = out_ch

            if attention_layers[i]:
                block_group.append(AttentionBlock(ch, num_heads=num_heads, num_groups=num_groups))

            upsample_layer = None
            if i != 0:
                upsample_layer = Upsample(ch)

            # Prepend to keep the order correct for the forward pass
            self.up_blocks.insert(0, nn.ModuleDict({
                'block_group': block_group,
                'upsample': upsample_layer
            }))

        self.conv_out = nn.Sequential(
            nn.GroupNorm(num_groups, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)

        for block in self.bottleneck:
            if self.use_checkpointing and self.training:
                h = checkpoint(block, h, use_reentrant=False)
            else:
                h = block(h)

        for stage in self.up_blocks:
            for block in stage['block_group']:
                if self.use_checkpointing and self.training:
                    h = checkpoint(block, h, use_reentrant=False)
                else:
                    h = block(h)

            if stage['upsample'] is not None:
                h = stage['upsample'](h)

        h = self.conv_out(h)
        return h