# Score Models Module

The score_models module in the `nn_handler.model_utils` package provides implementations of score-based generative models, including loss functions, noise schedules, and SDE solvers.

## Overview

Score-based generative models are a class of generative models that learn the score function (gradient of the log probability density) of the data distribution. These models can be used to generate new samples by solving stochastic differential equations (SDEs) or using Langevin dynamics.

The score_models module provides the necessary components for implementing and training score-based models, including:

- Loss functions for score matching
- Noise schedules for diffusion processes
- SDE solvers for sample generation
- Patch-based score models for image generation

## Components

### [Loss Functions](loss_fn.md)

The loss_fn module provides implementations of various loss functions for training score-based models:

- Denoising score matching loss
- Sliced score matching loss
- Variance-preserving (VP) SDE loss
- Variance-exploding (VE) SDE loss

These loss functions are designed to train neural networks to estimate the score function of the data distribution.

### [Noise Schedules](schedules.md)

The schedules module provides implementations of noise schedules for diffusion processes:

- Linear schedules
- Cosine schedules
- Exponential schedules

These schedules control how noise is added to the data during the forward process and how it is removed during the reverse process.

### [SDE Solvers](sde_solver.md)

The sde_solver module provides implementations of stochastic differential equation (SDE) solvers for generating samples from score-based models:

- Euler-Maruyama solver
- Predictor-corrector methods
- Probability flow ODE solver

These solvers are used to generate new samples by reversing the diffusion process.

### [Patch-Based Score Models](patch_score.md)

The patch_score module provides implementations of patch-based score models for image generation:

- Patch-based score networks
- Patch-based sampling methods

These models operate on image patches rather than entire images, which can be more efficient for high-resolution image generation.

## Usage with NNHandler

The score_models module is designed to be used with the NNHandler framework. Here's an example of how to use a score-based model with NNHandler:

```python
import torch
from src.nn_handler import NNHandler
from src.nn_handler.model_utils.score_models.loss_fn import denoising_score_matching_loss
from your_project.models import YourScoreNetwork

# Initialize the handler with a score network
handler = NNHandler(
    model_class=YourScoreNetwork,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Set up the handler with a score matching loss function
handler.set_loss_fn(
    denoising_score_matching_loss,
    pass_epoch_to_loss=True,  # If the loss function needs the current epoch
    sigma_min=0.01,
    sigma_max=50.0,
    schedule="cosine"
)

# Set up optimizer and data loaders
handler.set_optimizer(torch.optim.Adam, lr=1e-4)
handler.set_train_loader(train_dataset, batch_size=32)
handler.set_val_loader(val_dataset, batch_size=32)

# Train the model
handler.train(epochs=1000, validate_every=10)

# Generate samples using an SDE solver
from src.nn_handler.model_utils.score_models.sde_solver import EulerMaruyamaSolver
from src.nn_handler.model_utils.sampler import Sampler

class ScoreSampler(Sampler):
    def __init__(self, sde_solver, **kwargs):
        super().__init__(**kwargs)
        self.sde_solver = sde_solver
    
    def sample(self, N, device, shape=(3, 32, 32), **kwargs):
        return self.sde_solver.sample(N, shape, device, **kwargs)
    
    def save(self):
        return self.sde_solver.get_state_dict()
    
    def load(self, **state_dict):
        self.sde_solver.load_state_dict(**state_dict)
        return self

# Create an SDE solver and sampler
sde_solver = EulerMaruyamaSolver(
    score_fn=handler.model,
    sigma_min=0.01,
    sigma_max=50.0,
    N_steps=1000
)
sampler = ScoreSampler(sde_solver)

# Set the sampler in the handler
handler.set_sampler(sampler)

# Generate samples
samples = handler.sample(N=16)
```

## Advanced Usage

### Custom Score Networks

You can implement custom score networks by subclassing `torch.nn.Module` and implementing the forward method to take a batch of data and a noise level as input:

```python
import torch
import torch.nn as nn

class SimpleScoreNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x, sigma):
        """
        Args:
            x: Input data, shape (batch_size, input_dim)
            sigma: Noise level, shape (batch_size, 1) or scalar
        
        Returns:
            score: Score estimate, shape (batch_size, input_dim)
        """
        # Expand sigma to match x's shape for element-wise operations
        sigma = sigma.view(-1, 1) if sigma.dim() == 1 else sigma
        
        # Compute the score estimate
        return self.net(x) / sigma
```

### Custom SDE Solvers

You can implement custom SDE solvers by subclassing the appropriate base class in the sde_solver module:

```python
from src.nn_handler.model_utils.score_models.sde_solver import SDESolver

class CustomSDESolver(SDESolver):
    def __init__(self, score_fn, sigma_min, sigma_max, N_steps):
        super().__init__(score_fn, sigma_min, sigma_max, N_steps)
    
    def sample(self, N, shape, device, **kwargs):
        # Implement your custom sampling algorithm
        pass
```

## References

For more information on score-based generative models, see the following papers:

1. Song, Y., & Ermon, S. (2019). Generative Modeling by Estimating Gradients of the Data Distribution. NeurIPS.
2. Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). Score-Based Generative Modeling through Stochastic Differential Equations. ICLR.
3. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. NeurIPS.