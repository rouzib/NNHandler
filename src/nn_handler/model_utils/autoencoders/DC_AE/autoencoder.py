from typing_extensions import Unpack

import torch
import torch.nn as nn

from einops import rearrange
from torch import Tensor
from torch.nn.functional import cosine_similarity
from typing import Any, Dict, Optional, Sequence, Tuple

from .blocks import DCDecoder, DCEncoder


class AutoEncoder(nn.Module):
    r"""Creates an auto-encoder module.

    Arguments:
        encoder: An encoder module.
        decoder: A decoder module.
        saturation: The type of latent saturation.
        noise: The latent noise's standard deviation.
    """

    def __init__(self, in_channels: int, lat_channels: int, spatial: int = 2, saturation: str = None,
                 noise: float = 0.0, encoder_only=None, decoder_only=None, use_fp16=False, **kwargs: Any):
        super().__init__()

        if decoder_only is None:
            decoder_only = {}
        if encoder_only is None:
            encoder_only = {}
        self.encoder = DCEncoder(in_channels=in_channels, out_channels=lat_channels, spatial=spatial, **encoder_only,
                                 **kwargs)

        self.decoder = DCDecoder(in_channels=lat_channels, out_channels=in_channels, spatial=spatial, **decoder_only,
                                 **kwargs)
        self.saturation = saturation
        self.noise = noise
        self.use_fp16 = use_fp16

    def saturate(self, x: Tensor) -> Tensor:
        if self.saturation is None:
            return x
        elif self.saturation == "softclip":
            return x / (1 + abs(x) / 5)
        elif self.saturation == "softclip2":
            return x * torch.rsqrt(1 + torch.square(x / 5))
        elif self.saturation == "tanh":
            return torch.tanh(x / 5) * 5
        elif self.saturation == "arcsinh":
            return torch.arcsinh(x)
        elif self.saturation == "rmsnorm":
            return x * torch.rsqrt(torch.mean(torch.square(x), dim=1, keepdim=True) + 1e-5)
        else:
            raise ValueError(f"unknown saturation '{self.saturation}'")

    def encode(self, x: Tensor) -> Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_fp16):
            z = self.encoder(x)
            z = self.saturate(z)
            return z

    def decode(self, z: Tensor, noisy: bool = True) -> Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_fp16):
            if noisy and self.noise > 0:
                z = z + self.noise * torch.randn_like(z)

            return self.decoder(z)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_fp16):
            z = self.encode(x)
            y = self.decode(z)
            return y, z


class AutoEncoderLoss(nn.Module):
    r"""Creates a weighted auto-encoder loss module."""

    def __init__(
            self,
            losses=None,
            weights=None,
            device: Optional[torch.device] = None
    ):
        super().__init__()

        if weights is None:
            weights = [1.0]
        if losses is None:
            losses = ["mse"]
        assert len(losses) == len(weights)

        self.losses = list(losses)
        self.register_buffer("weights", torch.as_tensor(weights, device=device))

    def forward(self, autoencoder_output, x: Tensor, **kwargs) -> Tensor:
        r"""
        Arguments:
            autoencoder_output: The output of the auto-encoder.
            x: A clean tensor :math:`x`, with shape :math:`(B, C, ...)`.
            kwargs: Optional keyword arguments.
        Returns:
            The weighted loss.
        """

        y, z = autoencoder_output

        values = []

        for loss in self.losses:
            if loss == "mse":
                l = (x - y).square().mean()
            elif loss == "mae":
                l = (x - y).abs().mean()
            elif loss == "vmse":
                x = rearrange(x, "B C ... -> B C (...)")
                y = rearrange(y, "B C ... -> B C (...)")
                l = (x - y).square().mean(dim=2) / (x.var(dim=2) + 1e-2)
                l = l.mean()
            elif loss == "vrmse":
                x = rearrange(x, "B C ... -> B C (...)")
                y = rearrange(y, "B C ... -> B C (...)")
                l = (x - y).square().mean(dim=2) / (x.var(dim=2) + 1e-2)
                l = torch.sqrt(l).mean()
            elif loss == "similarity":
                f = rearrange(z, "B ... -> B (...)")
                l = cosine_similarity(f[None, :], f[:, None], dim=-1)
                l = l[Unpack[torch.triu_indices(*l.shape, offset=1, device=l.device)]]
                l = l.mean()
            else:
                raise ValueError(f"unknown loss '{loss}'.")

            values.append(l)

        values = torch.stack(values)

        return torch.vdot(self.weights, values)
