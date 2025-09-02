# Dummy Classes

The `dummy_classes` module provides placeholder implementations of classes that might not be available in all environments. These dummy classes maintain the same interface as their real counterparts, allowing code to run seamlessly regardless of whether the actual functionality is available.

## Overview

This module addresses two main use cases:

1. **Automatic Mixed Precision (AMP)**: Provides dummy implementations of PyTorch's `GradScaler` and `autocast` when AMP is not available.
2. **Exponential Moving Average (EMA)**: Provides a dummy implementation of EMA functionality when the actual implementation is not available.

These dummy classes implement the same methods as their real counterparts but with no-op functionality. This allows code that uses these features to run without modification, even in environments where the actual functionality is not available.

## Classes and Functions

### Automatic Mixed Precision (AMP)

#### `GradScaler`

A dummy implementation of PyTorch's `torch.cuda.amp.GradScaler` class.

```python
class GradScaler:
    def __init__(self, enabled=False):
        self._enabled = enabled
```

**Methods:**
- `scale(loss)`: Returns the loss unchanged.
- `step(optimizer)`: Calls the optimizer's step method.
- `update()`: No-op.
- `state_dict()`: Returns an empty dictionary.
- `load_state_dict(state_dict)`: No-op.
- `is_enabled()`: Returns the enabled state.

#### `autocast`

A dummy context manager that mimics PyTorch's `torch.cuda.amp.autocast`.

```python
@contextlib.contextmanager
def autocast(device_type="cpu", enabled=False, **kwargs):
    yield
```

**Usage:**
```python
from src.nn_handler.utils import dummy_classes

# Use the dummy autocast context manager
with dummy_classes.autocast(enabled=True):
    # Code that would normally use mixed precision
    output = model(input)
    loss = loss_fn(output, target)
```

### Exponential Moving Average (EMA)

#### `ExponentialMovingAverage`

A dummy implementation of an Exponential Moving Average for model parameters.

```python
class ExponentialMovingAverage:
    def __init__(self, parameters, decay):
        pass
```

**Methods:**
- `update()`: No-op.
- `average_parameters()`: A context manager that yields without doing anything.
- `copy_to(parameters=None)`: No-op.
- `state_dict()`: Returns an empty dictionary.
- `load_state_dict(state_dict)`: No-op.

**Usage:**
```python
from src.nn_handler.utils import dummy_classes

# Create a dummy EMA instance
ema = dummy_classes.ExponentialMovingAverage(model.parameters(), decay=0.999)

# Update EMA (no-op in dummy implementation)
ema.update()

# Use EMA parameters temporarily
with ema.average_parameters():
    # Model uses EMA parameters in real implementation
    # In dummy implementation, this does nothing
    output = model(input)
```

## Integration with NNHandler

The `NNHandler` class uses these dummy classes as fallbacks when the actual functionality is not available:

```python
# Example of how NNHandler might use these classes
if _amp_available:
    from torch.cuda.amp import GradScaler, autocast
else:
    from .utils.dummy_classes import GradScaler, autocast

if _ema_available:
    from torch_ema import ExponentialMovingAverage
else:
    from .utils.dummy_classes import ExponentialMovingAverage
```

This allows the `NNHandler` to maintain a consistent interface regardless of the environment, with graceful degradation when certain features are not available.

## Usage Notes

- The dummy classes are not meant to be used directly in most cases. They are primarily used as fallbacks within the NNHandler framework.
- When using the dummy implementations, the corresponding functionality (AMP or EMA) will not be active, but code will run without errors.
- The dummy `GradScaler` tracks its enabled state to allow code to check whether AMP is being used.