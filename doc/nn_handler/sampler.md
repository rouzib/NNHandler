# Custom Sampler Integration

The `NNHandler` framework supports integrating custom sampling algorithms, particularly useful for generative models beyond the built-in SDE-based sampling.

## Overview

While `NNHandler` provides a `sample` method specifically designed for **score-based generative models** (using Stochastic Differential Equations configured via `set_sde`), it also allows you to define and use your own sampling logic for other types of generative models (like GANs, VAEs with specific sampling strategies, etc.).

This is achieved by:
1.  Defining a custom sampler class that inherits from the abstract base class `src.nn_handler.sampler.Sampler`.
2.  Implementing the required `sample`, `save`, and `load` methods in your custom class.
3.  Configuring the `NNHandler` to use your custom sampler via `set_sampler`.
4.  Triggering sample generation using `get_samples`.

## The `Sampler` Abstract Base Class

Your custom sampler class **must** inherit from `src.nn_handler.sampler.Sampler` and implement its abstract methods.

```python
# Defined in src/nn_handler/sampler.py
import abc
from typing import Dict, Any, Optional
import torch

class Sampler(abc.ABC):
    @abc.abstractmethod
    def sample(self, N: int, device: Optional[torch.device], **kwargs) -> Any:
        """Generates N samples on the specified device."""
        pass

    @abc.abstractmethod
    def save(self) -> Dict[str, Any]:
        """Returns a dictionary containing the sampler's state."""
        pass

    @abc.abstractmethod
    def load(self, state_dict: Dict[str, Any]):
        """Loads the sampler's state from a dictionary."""
        pass
```

*   **`sample(self, N, device, **kwargs)`**: This method should contain the core logic to generate `N` samples. It receives the desired number of samples (`N`) and the target `device`. Any additional arguments needed for sampling should be handled via `**kwargs` (these might be passed during `set_sampler` or potentially `get_samples`, although passing during initialization is often cleaner).
*   **`save(self)`**: Should return a dictionary containing any internal state of the sampler that needs to be saved and restored (e.g., internal buffers, parameters specific to the sampler). This is used when `NNHandler.save()` is called.
*   **`load(self, state_dict)`**: Should take a dictionary (as returned by `save`) and restore the sampler's internal state. This is used when `NNHandler.load()` is called.

## Configuring the Custom Sampler

Use the `set_sampler` method of the `NNHandler` instance to associate your custom sampler implementation.

```python
def set_sampler(self, sampler_class: type[Sampler], **sampler_kwargs)
```

*   `sampler_class` (type[Sampler]): Your custom class (which inherits from `Sampler`).
*   `**sampler_kwargs`: Keyword arguments passed directly to your custom sampler's `__init__` constructor. This is where you typically pass the model itself or any fixed parameters for the sampler.

## Generating Samples

Once configured, use the `get_samples` method to generate samples using your custom sampler.

```python
def get_samples(self, N, device=None)
```

*   `N` (int): The number of samples to generate.
*   `device` (Optional[torch.device | str], default=None): The device for generation. If `None`, the handler's default device (`self.device`) is used.

This method calls the `sample` method of the configured `Sampler` instance.

**Note:** In DDP mode, `get_samples` is typically intended to be called only on Rank 0. The method itself doesn't prevent calls on other ranks, but the custom `sample` implementation might not be DDP-aware, and only Rank 0 usually needs the generated samples.

## Usage Example

```python
import torch
import abc
from typing import Dict, Any, Optional
from src.nn_handler import NNHandler
from src.nn_handler.sampler import Sampler # Import base class

# --- 1. Define Your Custom Sampler --- 
class MyVAESampler(Sampler):
    def __init__(self, model: torch.nn.Module, temperature: float = 1.0):
        self.model = model # Expects the VAE model (or relevant parts)
        self.temperature = temperature
        # Add any other state needed
        self.internal_state = 0

    def sample(self, N: int, device: Optional[torch.device], **kwargs) -> torch.Tensor:
        # Example: Sampling from a VAE's latent space
        self.model.eval() # Ensure model is in eval mode
        latent_dim = getattr(self.model, 'latent_dim', 64) # Get model's latent dim
        
        with torch.no_grad():
            # Sample random latent vectors (apply temperature if desired)
            z = torch.randn(N, latent_dim, device=device) * self.temperature
            
            # Decode latent vectors using the model's decoder part
            # Assumes model has a 'decode' method or similar structure
            if hasattr(self.model, 'decode'):
                generated_samples = self.model.decode(z)
            else:
                # Adapt if model structure is different
                raise NotImplementedError("Sampler requires model with 'decode' method")
            
            # Update internal state (example)
            self.internal_state += N
            
            return generated_samples

    def save(self) -> Dict[str, Any]:
        # Save relevant internal state
        return {
            'temperature': self.temperature,
            'internal_state': self.internal_state
        }

    def load(self, state_dict: Dict[str, Any]):
        # Load state
        self.temperature = state_dict.get('temperature', 1.0)
        self.internal_state = state_dict.get('internal_state', 0)
        print(f"Loaded sampler state: temp={self.temperature}, internal={self.internal_state}")

# --- 2. Assume you have a trained VAE model and Handler --- 
# Dummy VAE for example
class DummyVAE(torch.nn.Module):
    def __init__(self, latent_dim=64): 
        super().__init__()
        self.latent_dim=latent_dim
        # Dummy decoder part
        self.decoder_fc = torch.nn.Linear(latent_dim, 784)
    def decode(self, z): 
        # Example decoder logic
        img_flat = torch.sigmoid(self.decoder_fc(z))
        return img_flat.view(-1, 1, 28, 28) # Reshape to image
    def forward(self, x): # Dummy forward for training compatibility
        return self.decode(torch.randn(x.shape[0], self.latent_dim, device=x.device))

# Load or create handler with trained model
# handler = NNHandler.load("path/to/your/vae_handler.pth") 
# OR initialize and train:
handler = NNHandler(model_class=DummyVAE, model_kwargs={'latent_dim': 64})
# ... handler.set_optimizer, set_loss_fn, set_train_loader ...
# handler.train(epochs=...) 

# --- 3. Configure the Custom Sampler --- 
handler.set_sampler(
    sampler_class=MyVAESampler,
    # Pass arguments for MyVAESampler.__init__
    model=handler.module, # Pass the unwrapped model
    temperature=0.7 
)

# --- 4. Generate Samples --- 
num_samples_to_generate = 16
if handler._rank == 0: # Typically call on rank 0 in DDP
    generated_samples = handler.get_samples(
        N=num_samples_to_generate,
        device=handler.device 
    )

    if generated_samples is not None:
        print(f"Generated {generated_samples.shape[0]} samples.")
        # Output shape depends on sampler, e.g., torch.Size([16, 1, 28, 28])
```

This example illustrates the workflow: define a sampler inheriting from the base class, implement its methods, configure it within the handler (passing necessary arguments like the model), and then use `get_samples` to trigger generation.