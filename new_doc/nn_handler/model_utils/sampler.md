# Sampler Module

The sampler module in the `nn_handler.model_utils` package provides an abstract base class for implementing various sampling strategies in generative models.

## Overview

Sampling is a crucial component of many generative models, such as diffusion models, score-based models, and other types of generative neural networks. The sampler module defines a consistent interface for implementing different sampling strategies, ensuring that all samplers provide the necessary functionality for:

- Generating samples from a model
- Serializing the sampler state for saving
- Deserializing the sampler state for loading

## Classes

### `Sampler`

```python
class Sampler(abc.ABC):
    def __init__(self, **kwargs)
```

An abstract base class that defines the interface for sampling strategies. This class is intended to be subclassed to implement various sampling algorithms.

#### Methods:

* `sample(N, device, **kwargs)` (abstract):
  * Generates N samples on the specified device.
  * Parameters:
    * `N` (int): Number of samples to generate.
    * `device` (torch.device): Device to generate samples on.
    * `**kwargs`: Additional sampling parameters specific to the implementation.
  * Returns:
    * Implementation-specific, typically a tensor of generated samples.

* `save()` (abstract):
  * Serializes the sampler state for saving.
  * Returns:
    * Dict[str, Any]: A dictionary containing the serialized state.

* `load(**state_dict)` (abstract):
  * Deserializes the sampler state for loading.
  * Parameters:
    * `**state_dict` (Dict[str, Any]): A dictionary containing the serialized state.
  * Returns:
    * Implementation-specific, typically the sampler instance.

## Implementation Example

While the `Sampler` class itself is abstract and cannot be instantiated directly, here's an example of how to implement a concrete sampler:

```python
import torch
from src.nn_handler.model_utils.sampler import Sampler

class GaussianNoiseSampler(Sampler):
    """A simple sampler that generates Gaussian noise."""
    
    def __init__(self, mean=0.0, std=1.0, **kwargs):
        super().__init__(**kwargs)
        self.mean = mean
        self.std = std
    
    def sample(self, N, device, shape=(3, 32, 32), **kwargs):
        """
        Generate N samples of Gaussian noise.
        
        Args:
            N (int): Number of samples to generate.
            device (torch.device): Device to generate samples on.
            shape (tuple): Shape of each sample (excluding batch dimension).
            **kwargs: Additional sampling parameters (unused in this implementation).
            
        Returns:
            torch.Tensor: A tensor of shape (N, *shape) containing the generated samples.
        """
        return torch.normal(
            mean=self.mean,
            std=self.std,
            size=(N, *shape),
            device=device
        )
    
    def save(self):
        """
        Serialize the sampler state.
        
        Returns:
            dict: A dictionary containing the serialized state.
        """
        return {
            'mean': self.mean,
            'std': self.std
        }
    
    def load(self, **state_dict):
        """
        Deserialize the sampler state.
        
        Args:
            **state_dict: A dictionary containing the serialized state.
            
        Returns:
            self: The sampler instance.
        """
        self.mean = state_dict.get('mean', 0.0)
        self.std = state_dict.get('std', 1.0)
        return self
```

## Usage with NNHandler

The `Sampler` class is designed to be used with the NNHandler framework. Here's an example of how to use a custom sampler with NNHandler:

```python
from src.nn_handler import NNHandler
from your_project.samplers import YourCustomSampler

# Initialize the handler
handler = NNHandler(
    model_class=YourModel,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Set up the handler with a custom sampler
sampler = YourCustomSampler(param1=value1, param2=value2)
handler.set_sampler(sampler)

# Later, use the sampler to generate samples
samples = handler.sample(N=16)
```

## Integration with Score-Based Models

The `Sampler` class is particularly useful for score-based generative models, where different sampling strategies (e.g., Langevin dynamics, predictor-corrector methods) can be implemented as subclasses. These can then be used with the SDE solvers provided in the `score_models` module.

For more information on score-based models and SDE solvers, see the [score_models](score_models/README.md) documentation.

## Notes

1. When implementing a custom sampler, ensure that the `save` and `load` methods properly serialize and deserialize all necessary state variables.

2. The `sample` method should be designed to work with batched inputs and outputs, with the first dimension being the batch dimension.

3. Consider implementing additional methods specific to your sampling strategy, but keep the core interface methods (`sample`, `save`, `load`) consistent with the abstract base class.