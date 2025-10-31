from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .architectures import Encoder, Decoder


class DiagonalGaussianDistribution:
    """
    Represents a diagonal Gaussian distribution.

    This class is used to model a diagonal Gaussian distribution with a given
    set of parameters. It provides methods to sample from the distribution
    and compute the Kullback–Leibler (KL) divergence.

    Attributes:
        mean (torch.Tensor): The mean of the diagonal Gaussian distribution,
            derived from the input parameters.
        logvar (torch.Tensor): The log variance of the distribution, derived
            from the input parameters and clamped within a specified range
            to maintain numerical stability.
        std (torch.Tensor): The standard deviation of the distribution,
            computed as the exponential of half the log variance.
    """

    @torch.autocast("cuda", torch.float32)
    def __init__(self, parameters: torch.Tensor):
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self) -> torch.Tensor:
        return self.mean + self.std * torch.randn_like(self.std)

    @torch.autocast("cuda", torch.float32)
    def kl(self) -> torch.Tensor:
        kl = torch.pow(self.mean, 2) + torch.exp(self.logvar) - 1.0 - self.logvar
        return 0.5 * torch.sum(kl, dim=tuple(range(1, kl.dim())))


    def __repr__(self):
        return f"DiagonalGaussianDistribution(mean={self.mean}, logvar={self.logvar})"


class AutoencoderKL(nn.Module):
    """
    Represents an autoencoder utilizing a variational KL-divergence approach.

    This class integrates an encoder, decoder, and additional quantization layers
    for transforming and reconstructing data with a focus on dimensionality
    reduction and latent space learning. It is designed for applications requiring
    a probabilistic latent variable framework.

    Attributes:
        encoder (Encoder): The encoder network that processes input tensors and
            produces a latent representation.
        decoder (Decoder): The decoder network that reconstructs data from the
            latent space representation.
        quant_conv (nn.Conv2d): A convolutional layer applied after encoding to
            process latent representations before quantization.
        post_quant_conv (nn.Conv2d): A convolutional layer applied after
            quantization to further process latent representations.
    """

    def __init__(self, config: dict):
        super().__init__()

        num_groups = config.get("num_groups", 32)
        num_heads = config.get("num_heads", 1)
        use_checkpointing = config.get("use_checkpointing", False)

        # Get attention_layers; if not provided, create a default
        attention_layers = config.get("attention_layers")
        if attention_layers is None:
            # Default behavior: apply attention to the last two stages
            num_stages = len(config["channel_multipliers"])
            attention_layers = [i >= num_stages - 2 for i in range(num_stages)]

        encoder_config = {
            "in_channels": config["in_channels"],
            "base_channels": config["base_channels"],
            "channel_multipliers": config["channel_multipliers"],
            "num_res_blocks": config["num_res_blocks"],
            "z_channels": config["z_channels"],
            "num_groups": num_groups,
            "num_heads": num_heads,
            "attention_layers": attention_layers,
            "use_checkpointing": use_checkpointing
        }

        decoder_config = {
            "out_channels": config["out_channels"],
            "base_channels": config["base_channels"],
            "channel_multipliers": config["channel_multipliers"],
            "num_res_blocks": config["num_res_blocks"],
            "z_channels": config["z_channels"],
            "num_groups": num_groups,
            "num_heads": num_heads,
            "attention_layers": attention_layers,
            "use_checkpointing": use_checkpointing
        }

        self.encoder = Encoder(**encoder_config)
        self.decoder = Decoder(**decoder_config)
        self.quant_conv = nn.Conv2d(2 * config['z_channels'], 2 * config['z_channels'], 1)
        self.post_quant_conv = nn.Conv2d(config['z_channels'], config['z_channels'], 1)

    @torch.autocast("cuda", torch.float16)
    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        """
        Encodes the input tensor into a diagonal Gaussian distribution. The function processes the
        given input through an encoder and a quantization convolution layer before creating the
        resulting distribution.

        Args:
            x (torch.Tensor): The input tensor to be encoded.

        Returns:
            DiagonalGaussianDistribution: The diagonal Gaussian distribution generated based on
            the input tensor.
        """
        moments = self.encoder(x)
        moments = self.quant_conv(moments)
        return DiagonalGaussianDistribution(moments)

    @torch.autocast("cuda", torch.float16)
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes the input latent tensor using the post-quantization convolution layer
        and a decoder.

        Args:
            z (torch.Tensor): The latent tensor input to be processed.

        Returns:
            torch.Tensor: The reconstructed output tensor.
        """
        z = self.post_quant_conv(z)
        return self.decoder(z)

    @torch.autocast("cuda", torch.float16)
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forwards the input tensor through the model, performing encoding, sampling, and
        decoding steps. Returns the reconstructed output and the KL divergence from
        the prior.

        :param x: Input tensor to be processed by the model.
        :type x: torch.Tensor
        :returns: A tuple containing the reconstructed tensor and the KL divergence.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        posterior = self.encode(x)
        z = posterior.sample()
        reconstruction = self.decode(z)
        return reconstruction, posterior.kl()


def vae_loss(outputs, targets, epoch, **kwargs):
    """
    VAE loss with KL weight scheduling. Beta is used as the max KL weight.

    Args:
        outputs: Tuple (reconstruction, posterior) as returned from model.
        targets: Input images or ground-truth targets.
        epoch: Current epoch number, determines scheduler value.
        **kwargs: Additional unused arguments (for compatibility).
    Returns:
        total_loss, recon_loss, kl_loss
    """
    if "beta" not in kwargs:
        beta_schedule = 1.0
    else:
        beta_schedule = kl_beta_scheduler(epoch, start=kwargs.get("beta_start", 0), stop=kwargs.get("beta"),
                                          n_epochs=kwargs.get("n_epochs", 1000),
                                          mode=kwargs.get("beta_schedule", "linear"))

    loss_type = kwargs.get("loss_type", "mse")
    loss_fn = F.mse_loss if loss_type == "mse" else F.l1_loss

    reconstruction, posterior_kl = outputs
    recon_loss = loss_fn(reconstruction, targets)
    kl_loss = posterior_kl.mean()
    total_loss = recon_loss + beta_schedule * kl_loss
    return total_loss, recon_loss, kl_loss


def kl_beta_scheduler(epoch, start=0, stop=1, n_epochs=10, mode="linear"):
    """
    Schedule beta from `start` to `stop` over `n_epochs`.
    mode: 'linear' for linear ramp, 'log' for logarithmic ramp.
    """
    if epoch < 0:
        return start
    elif epoch >= n_epochs:
        return stop

    # Avoid division by zero for corner cases
    if n_epochs <= 1:
        return stop

    if mode == "log":
        # Log schedule: ramps up slowly, then faster
        # Map epoch to [1, n_epochs] for logspace, and take log interpolation
        progress = np.log10(1 + 9 * (epoch / n_epochs))  # log10(1) to log10(10)
        weight = start + (stop - start) * (progress / 1.0)  # /1.0 since log10(10)=1
        return weight
    else:
        # Linear
        return start + (stop - start) * (epoch / n_epochs)
