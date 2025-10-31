r"""Deep Compressed Auto-Encoder

References:
    | Lost in Latent Space: An Empirical Study of Latent Diffusion Models for Physics Emulation (Rozet et al., 2025)
    | https://arxiv.org/pdf/2507.02608

    | Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models (Chen et al., 2024)
    | https://arxiv.org/abs/2410.10733v1

    | DC-AE 1.5: Accelerating Diffusion Model Convergence with Structured Latent Space (Chen et al., 2025)
    | https://arxiv.org/pdf/2508.00413
"""

from .autoencoder import AutoEncoder, AutoEncoderLoss
from .autoencoder_KL import AutoEncoderKL, AutoEncoderKLLoss

__all__ = ["AutoEncoder", "AutoEncoderLoss", "AutoEncoderKL", "AutoEncoderKLLoss"]