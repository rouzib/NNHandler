from typing import List, Optional

import torch
from einops import rearrange
from torch import nn, Tensor, cosine_similarity
from typing_extensions import Unpack

from .blocks import DCEncoder, DCDecoder
from ..VAE import DiagonalGaussianDistribution


class AutoEncoderKL(nn.Module):
    def __init__(self, in_channels: int, lat_channels: int, lat_size: int, image_shape: List[int] = None,
                 spatial: int = 2, saturation: str = None, noise: float = 0.0, sample: bool = True, encoder_only=None,
                 decoder_only=None, use_fp16=False, **kwargs):
        super().__init__()

        if image_shape is None:
            image_shape = [128, 128]
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
        self.image_shape = image_shape
        self.encoder_compression_factor = 2 ** (len(kwargs.get("hid_channels", 1)) - 1)
        self.proj_in_params = lat_channels * image_shape[0] // self.encoder_compression_factor * image_shape[
            1] // self.encoder_compression_factor
        self.proj_out_shape = (lat_channels, image_shape[0] // self.encoder_compression_factor,
                               image_shape[1] // self.encoder_compression_factor)
        self.quant = nn.Linear(self.proj_in_params, 2 * lat_size)
        self.post_quant = nn.Linear(lat_size, self.proj_in_params)

    def encode(self, x: Tensor) -> DiagonalGaussianDistribution:
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=self.use_fp16 or torch.is_autocast_enabled()):
            z = self.encoder(x)
            z = self.quant(z.flatten(start_dim=1))
            return DiagonalGaussianDistribution(z)

    def decode(self, z: Tensor, noisy: bool = True) -> Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=self.use_fp16 or torch.is_autocast_enabled()):
            if noisy and self.noise > 0:
                z = z + self.noise * torch.randn_like(z)

            z = self.post_quant(z).reshape(-1, *self.proj_out_shape)
            return self.decoder(z)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=self.use_fp16 or torch.is_autocast_enabled()):
            posterior = self.encode(x)
            z = posterior.sample() if self.sample else posterior.mean
            y = self.decode(z)
            return y, posterior.kl()


class AutoEncoderKLLoss(nn.Module):
    r"""Creates a weighted auto-encoder loss module."""

    def __init__(
            self,
            losses=None,
            weights=None,
            beta: float = 1.0,
            device: Optional[torch.device] = None
    ):
        super().__init__()

        self.beta = beta

        if weights is None:
            weights = [1.0]
        if losses is None:
            losses = ["mse"]
        assert len(losses) == len(weights)

        weights.insert(0, self.beta)
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

        y, kl = autoencoder_output

        values = [kl.mean()]

        for loss in self.losses:
            if loss == "mse":
                l = (x - y).square().mean()
            elif loss == "mse_log":
                l = ((x - y).square() + 1).log().mean()
            elif loss == "mae":
                l = (x - y).abs().mean()
            elif loss == "mae_norm":
                l = ((x - y) / (x.abs() + self.norm_factor)).abs().mean()
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
            else:
                raise ValueError(f"unknown loss '{loss}'.")

            values.append(l)

        values = torch.stack(values)

        return torch.vdot(self.weights, values)
