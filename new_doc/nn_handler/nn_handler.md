# NNHandler Class API

The `NNHandler` class, implemented in `nn_handler_distributed.py`, is the core component of the framework, providing a comprehensive interface for managing PyTorch neural network models through their entire lifecycle.

## Overview

`NNHandler` simplifies training, evaluation, and deployment by handling:

*   Model instantiation and device placement.
*   Optimizer, scheduler, loss function, and data loader setup.
*   Execution of training and validation loops.
*   **Distributed Data Parallel (DDP)** setup and management for multi-GPU/multi-node training.
*   **Automatic Mixed Precision (AMP)** for faster training on compatible hardware.
*   **Gradient Accumulation** to simulate larger batch sizes.
*   **Gradient Clipping** to prevent exploding gradients.
*   **Exponential Moving Average (EMA)** for improved model generalization.
*   A flexible **Callback system** for extensibility.
*   Metric calculation, tracking, and **plotting**.
*   Comprehensive **state saving and loading** for resuming training or deployment.
*   Support for **generative models**, including SDE-based samplers and custom samplers.
*   Integration with **`torch.compile`**.
*   Integrated **logging**.

## Initialization

```python
from src.nn_handler import NNHandler # Or adjust import path
import torch
import torch.nn as nn
import logging
from typing import Union, Optional, Callable, Dict, List, Any, Tuple, Type
from torch.utils.data import DataLoader, Dataset
from enum import Enum

class NNHandler:
    def __init__(self,
                 model_class: type[nn.Module],
                 device: Union[torch.device, str] = "cpu",
                 logger_mode: Optional[NNHandler.LoggingMode] = None,
                 logger_filename: str = "NNHandler.log",
                 logger_level: int = logging.INFO,
                 save_model_code: bool = False,
                 model_type: Union[NNHandler.ModelType, str] = NNHandler.ModelType.CLASSIFICATION,
                 use_distributed: Optional[bool] = None,
                 **model_kwargs):
        """
        Initializes the NNHandler.

        Args:
            model_class (type[nn.Module]): The PyTorch model class to instantiate.
            device (Union[torch.device, str], default="cpu"): The target device ('cpu', 'cuda').
                In DDP mode, this is often overridden by the DDP initialization based on local rank.
            logger_mode (Optional[NNHandler.LoggingMode], default=None): Logging mode
                (CONSOLE, FILE, BOTH). Logging is active only on Rank 0 in DDP.
            logger_filename (str, default="NNHandler.log"): Filename for file logging (Rank 0 only).
            logger_level (int, default=logging.INFO): Logging level (e.g., logging.INFO).
            save_model_code (bool, default=False): If True, attempts to save the model's source code
                within checkpoints (Rank 0 only). Requires model code to be accessible via `inspect`.
            model_type (Union[NNHandler.ModelType, str], default=ModelType.CLASSIFICATION):
                The type of model/task (e.g., CLASSIFICATION, REGRESSION, GENERATIVE, SCORE_BASED).
                Can be passed as an enum member or a string.
            use_distributed (Optional[bool], default=None): Controls DDP usage:
                - `True`: Explicitly attempt to enable DDP. Fails if env vars/torch.distributed not available.
                - `False`: Explicitly disable DDP, even if environment variables are set.
                - `None`: Auto-detect DDP based on environment variables (`RANK`, `LOCAL_RANK`,
                          `WORLD_SIZE` or Slurm vars) and `torch.distributed` availability.
            **model_kwargs: Additional keyword arguments passed directly to the `model_class` constructor.
        """
        # ... implementation details ...
```

*   **DDP Auto-Detection**: If `use_distributed` is `None`, NNHandler checks standard environment variables (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`) or Slurm variables (`SLURM_PROCID`, `SLURM_LOCALID`, `SLURM_NTASKS`) to determine if it should operate in DDP mode.
*   **Device Assignment**: In DDP mode, the device is typically assigned as `cuda:LOCAL_RANK` for GPU training or `cpu` for CPU training. The `device` argument serves as a hint but might be overridden.
*   **Logging**: Logging is configured based on `logger_mode` but only writes output on Rank 0 to prevent clutter.
*   **Model Initialization**: The `model_class` is instantiated with `model_kwargs` and moved to the correct device. In DDP mode, it's automatically wrapped with `torch.nn.parallel.DistributedDataParallel`.

### Enums

```python
class NNHandler.ModelType(Enum):
    CLASSIFICATION = "classification"
    GENERATIVE = "generative"
    REGRESSION = "regression"
    SCORE_BASED = "score_based"

    @classmethod
    def from_string(cls, s: str) -> 'NNHandler.ModelType': ...

class NNHandler.LoggingMode(Enum):
    CONSOLE = "console"
    FILE = "file"
    BOTH = "both"
```

## Key Properties and Setters

These properties allow configuration of the handler *after* initialization.

*   **`handler.device`** (`torch.device`, read-only): The device assigned to the current process/rank. Cannot be set after DDP initialization.
*   **`handler.seed`** (`Optional[int]`): Get or set the random seed for PyTorch and CUDA (applied on all ranks).
    ```python
    handler.seed = 42
    ```
*   **`handler.logger`** (`Optional[logging.Logger]`, read-only): Access the logger instance (returns `None` on non-zero ranks).
*   **`handler.model`** (`Optional[nn.Module]`, read-only): Access the model instance. This might be the raw model, or a `DDP`/`DataParallel` wrapped model. Use `handler.module` for the underlying model.
*   **`handler.module`** (`nn.Module`, read-only): Access the underlying `nn.Module`, automatically unwrapping `DDP` or `DataParallel` if necessary. Crucial for accessing custom methods or attributes of your original model class or when passing the model to functions expecting a plain `nn.Module` (e.g., some loss functions, `torch.compile`).
*   **`handler.optimizer`** (`Optional[torch.optim.Optimizer]`, read-only): Access the optimizer instance. Use `set_optimizer` to configure.
*   **`handler.scheduler`** (`Optional[torch.optim.lr_scheduler.LRScheduler]`, read-only): Access the scheduler instance. Use `set_scheduler` to configure.
*   **`handler.loss_fn`** (`Optional[Callable]`, read-only): Access the loss function. Use `set_loss_fn` to configure.
*   **`handler.pass_epoch_to_loss`** (`bool`): Get or set whether the current epoch number should be passed as a keyword argument (`epoch=...`) to the loss function during calculation.
*   **`handler.loss_fn_kwargs`** (`Dict[str, Any]`): Get or set the fixed keyword arguments passed to the loss function during calculation.
*   **`handler.train_loader`** (`Optional[DataLoader]`, read-only): Access the training data loader. Use `set_train_loader`.
*   **`handler.val_loader`** (`Optional[DataLoader]`, read-only): Access the validation data loader. Use `set_val_loader`.
*   **`handler.metrics`** (`Dict[str, Callable]`, read-only): Access the dictionary of registered metric functions. Use `add_metric`.
*   **`handler.callbacks`** (`List[Callback]`, read-only): Access the list of registered callbacks. Use `add_callback`.
*   **`handler.sde`** (`Optional[Any]`): Get or set the Stochastic Differential Equation (SDE) instance used for score-based models. Can be set via instance or `set_sde`.
*   **`handler.sampler`** (`Optional[Sampler]`): Get or set the custom Sampler instance used for `get_samples`. Can be set via instance or `set_sampler`.
*   **`handler.history`**: Access training history (Rank 0 only).
    *   `handler.train_losses` (`List[float]`)
    *   `handler.val_losses` (`List[float]`)
    *   `handler.train_metrics_history` (`Dict[str, List[float]]`)
    *   `handler.val_metrics_history` (`Dict[str, List[float]]`)
*   **`handler.auto_saver` properties**: Control auto-saving behavior (`save_interval`, `save_path`, `save_model_name`, `overwrite_last_saved`). See [AutoSaver Docs](autosaver.md).

### Configuration Methods

```python
# --- Model ---
handler.set_model(
    model_class: type[nn.Module],
    save_model_code: bool = False,
    model_type: Optional[Union[NNHandler.ModelType, str]] = None,
    **model_kwargs
) # Sets or replaces the model, handles DDP wrapping.

# --- Optimizer ---
handler.set_optimizer(
    optimizer_class: type[torch.optim.Optimizer],
    **optimizer_kwargs
) # Sets the optimizer for handler.model.parameters().

# --- Scheduler ---
handler.set_scheduler(
    scheduler_class: Optional[type[torch.optim.lr_scheduler.LRScheduler]],
    **scheduler_kwargs
) # Sets the LR scheduler, attached to the optimizer.

# --- Loss Function ---
handler.set_loss_fn(
    loss_fn: Callable,
    pass_epoch_to_loss: bool = False, # If True, passes `epoch=...` kwarg to loss_fn if accepted
    **kwargs # Additional fixed keyword arguments for the loss function
)

# --- Data Loaders ---
# In DDP mode, these methods automatically wrap the dataset with a DistributedSampler.
handler.set_train_loader(dataset: Dataset, **loader_kwargs)
handler.set_val_loader(dataset: Dataset, **loader_kwargs)

# --- Metrics & Callbacks ---
handler.add_metric(name: str, metric_fn: Callable)
handler.clear_metrics()
handler.add_callback(callback: Callback) # Add a callback instance

# --- Auto Saving ---
handler.auto_save(
    interval: Optional[int],
    save_path: str = '.',
    name: str = "model_epoch{epoch:02d}",
    overwrite: bool = False
) # Configures periodic state saving (details in autosaver.md)

# --- Generative Model Components ---
handler.set_sde(sde_class: type, **sde_kwargs) # Set SDE by class and args
handler.set_sampler(sampler_class: type[Sampler], **sampler_kwargs) # Set custom Sampler by class/args
```

## Core Methods

### Training

```python
handler.train(
    epochs: int,
    validate_every: int = 1,
    gradient_accumulation_steps: int = 1,
    use_amp: bool = False,
    gradient_clipping_norm: Optional[float] = None,
    ema_decay: float = 0.0,
    seed: Optional[int] = None,
    progress_bar: bool = True,
    debug_print_interval: Optional[int] = None,
    save_on_last_epoch: bool = True,
    epoch_train_and_val_pbar: bool = False
)
```
*   Starts the main training loop.
*   Handles epoch iteration, training steps, validation steps, metric calculation, and aggregation across DDP ranks.
*   Integrates AMP, gradient accumulation/clipping, and EMA if enabled.
*   Runs registered callbacks at appropriate stages.
*   Manages progress bars (rank 0 only).
*   Handles DDP sampler epoch setting for reproducibility.
*   Orchestrates AutoSaver calls (rank 0 only).

### Evaluation & Prediction

```python
handler.eval(activate: bool = True, log: bool = True)
```
*   Sets the model to evaluation (`activate=True`) or training (`activate=False`) mode. Handles modules configured to always stay in eval mode via `keep_eval_on_module`.

```python
handler.predict(data_loader: DataLoader, apply_ema: bool = True) -> Optional[List[Any]]
```
*   Performs inference on the provided `data_loader`.
*   **In DDP mode:** Each rank predicts on its subset of data. Results are gathered on Rank 0 and returned as a list of outputs (batch by batch). Other ranks return `None`. Requires `DistributedSampler(shuffle=False)` for ordered results.
*   Uses EMA weights if `apply_ema=True` and EMA is configured.

### Saving and Loading State

```python
handler.save(path: str)
```
*   Saves the *complete* state of the handler (model weights, optimizer/scheduler state, history, EMA, callbacks, configuration, etc.) to the specified path.
*   **In DDP mode:** Only Rank 0 performs the save operation, while other ranks wait at a barrier.

```python
NNHandler.load(
    path: str,
    device: Optional[Union[str, torch.device]] = None,
    strict_load: bool = False,
    skip_optimizer: bool = False,
    skip_scheduler: bool = False,
    skip_history: bool = False,
    skip_callbacks: bool = False,
    skip_sampler_sde: bool = False,
    skip_ema: bool = False
) -> 'NNHandler'
```
*   Class method to load a handler state from a saved file.
*   Reconstructs the handler with its configuration and restores the state of components.
*   **In DDP mode:** All ranks load the checkpoint, mapping tensors to their assigned device.
*   `skip_*` flags allow selectively ignoring parts of the saved state.
*   Handles potential mismatches between saved state (e.g., DDP) and current environment (e.g., non-DDP) for model weights.

```python
NNHandler.initialize_from_checkpoint(
    checkpoint_path: str,
    model_class: type[nn.Module],
    model_type: Optional[Union[NNHandler.ModelType, str]] = None,
    device: Optional[Union[str, torch.device]] = None,
    strict_load: bool = True,
    **model_kwargs
) -> 'NNHandler'
```
*   Class method to create a *new* handler instance, loading **only the model weights** from a checkpoint.
*   Useful for inference or transfer learning. Does *not* load optimizer, history, etc.
*   Assumes the checkpoint contains either just the model state dict, or a full handler state dict (from which it extracts `model_state_dict`).
*   Handles DDP loading similarly to `load`.

### Generative Model Methods

#### For Custom Samplers (via `set_sampler`)

```python
handler.get_samples(N, device=None) -> Optional[Any]
```
*   Generates `N` samples using the custom `Sampler` instance configured via `set_sampler`.
*   Typically runs on Rank 0 only in DDP mode, returning `None` on other ranks.
*   The output type depends on the custom sampler's implementation.

#### For Score-Based Models (via `set_sde`)

```python
handler.score(t: Tensor, x: Tensor, *args) -> Tensor
```
*   Computes the score function `s(t, x) = ∇ log p_t(x)` using the model and the configured SDE. Assumes the model output is `sigma(t) * s(t, x)`.

```python
handler.sample(
    shape: Tuple[int, ...],
    steps: int,
    condition: Optional[list] = None,
    likelihood_score_fn: Optional[Callable] = None,
    guidance_factor: float = 1.,
    apply_ema: bool = True,
    bar: bool = True,
    stop_on_NaN=True
) -> Optional[Tensor]
```
*   Generates samples using the reverse-time SDE (Euler-Maruyama solver).
*   **In DDP mode:** Intended to run on Rank 0 only; other ranks return `None`.
*   Supports conditional sampling (`condition`) and likelihood guidance (`likelihood_score_fn`, `guidance_factor`).

```python
handler.log_likelihood(
    x: Tensor,
    *args,
    ode_steps: int,
    n_cotangent_vectors: int = 1,
    noise_type: str = "rademacher",
    method: str = "Euler",
    t0: Optional[float] = None,
    t1: Optional[float] = None,
    apply_ema: bool = True,
    pbar: bool = False
) -> Optional[Tensor]
```
*   Estimates the log-likelihood of input data `x` using the Instantaneous Change of Variables formula via ODE integration (probability flow).
*   Uses the Hutchinson trace estimator for divergence calculation.
*   **In DDP mode:** Intended to run on Rank 0 only; other ranks return `None`.

### Other Utility Methods

```python
handler.compile_model(**kwargs)
```
*   Compiles the model using `torch.compile` (if available, PyTorch 2.0+). Pass `torch.compile` options as `kwargs`.

```python
handler.count_parameters(trainable_only: bool = True) -> int
```
*   Counts the number of parameters in the underlying model (`handler.module`).

```python
handler.freeze_module(module: nn.Module, verbose: bool = False) -> None:
```
*   Sets `requires_grad = False` for all parameters within the specified `module`.

```python
handler.keep_eval_on_module(module: nn.Module, activate: bool = True):
```
*   Registers a module to always remain in `eval()` mode, even when `handler.train()` or `handler.eval(activate=False)` is called. Useful for freezing parts like BatchNorm layers after initial training. Set `activate=False` to unregister.

```python
handler.plot_losses(log_y_scale: bool = False, save_path: Optional[str] = None)
```
*   Plots aggregated training and validation losses (requires `matplotlib`). Rank 0 only.

```python
handler.plot_metrics(log_y_scale: bool = False, save_path_prefix: Optional[str] = None)
```
*   Plots aggregated training and validation history for each registered metric (requires `matplotlib`). Rank 0 only.

```python
handler.print(show_model_structure=False) -> str
```
*   Returns a formatted string summarizing the handler's current status and configuration. Rank 0 provides the most complete information (history, etc.).

```python
handler.__call__(*args, **kwargs) -> Any
```
*   Provides a convenient way to perform a forward pass using the handler's model: `output = handler(input_data)`.