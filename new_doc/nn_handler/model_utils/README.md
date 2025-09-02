# Model Utilities Module

The model_utils module in the `nn_handler` package provides utilities and components for working with neural network models, with a particular focus on generative models and sampling techniques.

## Overview

The model_utils module offers a collection of tools that extend the core functionality of neural network models in PyTorch. These utilities are designed to:

- Implement sampling strategies for generative models
- Provide components for score-based generative models
- Support advanced model architectures and training techniques

## Components

The model_utils module consists of several components, each addressing specific aspects of model functionality:

### [Sampler](sampler.md)

The sampler module provides an abstract base class for implementing various sampling strategies in generative models:

- `Sampler`: An abstract base class that defines the interface for sampling strategies
- Methods for generating samples, serializing, and deserializing sampler state

Key features:
- Consistent interface for different sampling algorithms
- Integration with the NNHandler framework
- Support for saving and loading sampler state

### [Score Models](score_models/README.md)

The score_models module provides implementations of score-based generative models:

- Loss functions for score matching
- Noise schedules for diffusion processes
- SDE solvers for sample generation
- Patch-based score models for image generation

Key features:
- Complete toolkit for implementing score-based generative models
- Support for various score matching techniques
- Efficient sampling algorithms for generating new samples

## Integration with NNHandler

The model_utils components are designed to work seamlessly with the NNHandler framework:

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

# Use the sampler to generate samples
samples = handler.sample(N=16)
```

## Use Cases

### Generative Modeling

The model_utils module is particularly useful for generative modeling tasks:

1. **Score-Based Models**: Implement and train score-based generative models using the components in the score_models module.

2. **Diffusion Models**: Use the noise schedules and SDE solvers to implement diffusion-based generative models.

3. **Custom Sampling Strategies**: Implement custom sampling strategies by subclassing the Sampler abstract base class.

### Model Deployment

The model_utils module also provides utilities for deploying trained models:

1. **Efficient Sampling**: Generate samples from trained models using optimized sampling algorithms.

2. **State Management**: Save and load sampler state along with model weights for consistent deployment.

## Example Workflow

A typical workflow using the model_utils module might look like:

1. **Define a Model**: Implement a neural network model that outputs the desired quantities (e.g., a score network).

2. **Train the Model**: Use the NNHandler framework to train the model with appropriate loss functions.

3. **Implement a Sampler**: Create a custom sampler that uses the trained model to generate samples.

4. **Generate Samples**: Use the sampler to generate new samples from the model.

```python
import torch
from src.nn_handler import NNHandler
from src.nn_handler.model_utils.score_models.loss_fn import denoising_score_matching_loss
from src.nn_handler.model_utils.sampler import Sampler
from your_project.models import YourScoreNetwork

# 1. Define a model
# (YourScoreNetwork implementation)

# 2. Train the model
handler = NNHandler(
    model_class=YourScoreNetwork,
    device="cuda" if torch.cuda.is_available() else "cpu"
)
handler.set_loss_fn(denoising_score_matching_loss)
handler.set_optimizer(torch.optim.Adam, lr=1e-4)
handler.set_train_loader(train_dataset, batch_size=32)
handler.train(epochs=100)

# 3. Implement a sampler
class YourSampler(Sampler):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model
    
    def sample(self, N, device, **kwargs):
        # Implement your sampling algorithm
        pass
    
    def save(self):
        return {}  # Return serialized state
    
    def load(self, **state_dict):
        return self  # Load serialized state

# 4. Generate samples
sampler = YourSampler(handler.model)
handler.set_sampler(sampler)
samples = handler.sample(N=16)
```

For more detailed information on each component, refer to the individual documentation pages linked above.