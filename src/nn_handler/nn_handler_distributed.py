"""
Author: Nicolas Payot
Date: 08/22/2024

This script provides an extensive python module for handling all the necessary operations related to training,
validating, and managing a PyTorch neural network. This includes the declaration, optimization, and iteration of a
PyTorch model, along with advanced features like metric tracking, a flexible callback system, gradient accumulation,
mixed precision training, distributed training (multi-GPU, multi-node via DDP), and enhanced model checkpointing.
Interfaced functionalities for detailed logging and integration with tools like TensorBoard are implemented as well.
The NNHandler class facilitates model definition, data loading, optimizer/scheduler setup, loss calculation,
and manages the entire training lifecycle, aiming to streamline and accelerate the development and experimentation
process with PyTorch models.
"""
import abc
import contextlib
import os
import logging
import sys
import types
import warnings
from collections import OrderedDict, defaultdict
from datetime import timedelta
from enum import Enum
import inspect
from typing import Callable, Union, Optional, Dict, List, Any, Tuple, TypedDict
import math
import time
import copy

import torch
import torch.nn as nn
from torch import Tensor
from torch.func import vjp
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler, \
    Sampler as TorchSampler  # Added TorchSampler alias
import torch.nn.functional as F

# --- Distributed Training Imports ---
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# --- Local Imports ---
from .autosaver import AutoSaver, load_model_code
from .callbacks.base import Callback
from .sampler import Sampler
from .score_models import get_t_schedule
from .utils import _resolve_device, _initialize_distributed, _should_use_distributed

__version__ = "0.2.6_ddp"

# --- Conditional Imports ---
try:
    from torch.amp import GradScaler, autocast

    _amp_available = True
except ImportError:
    _amp_available = False


    # Define dummy classes if amp is not available
    class GradScaler:
        def __init__(self, enabled=False):
            self._enabled = enabled  # Store enabled state

        def scale(self, loss): return loss

        def step(self, optimizer): optimizer.step()

        def update(self): pass

        def __call__(self, *args, **kwargs): pass  # make it callable for load/save state_dict logic

        def state_dict(self): return {}

        def load_state_dict(self, state_dict): pass

        # Add is_enabled to match real GradScaler
        def is_enabled(self): return self._enabled


    @contextlib.contextmanager
    def autocast(device_type="cpu", enabled=False, **kwargs):
        # Simplified context manager that just yields
        yield

try:
    from torch_ema import ExponentialMovingAverage

    _ema_available = True
except ImportError:
    _ema_available = False


    # Define a dummy EMA class if not available
    class ExponentialMovingAverage:
        def __init__(self, parameters, decay): pass

        def update(self): pass

        @contextlib.contextmanager  # Ensure it's a context manager
        def average_parameters(self): yield  # Dummy context manager

        def copy_to(self, parameters=None): pass

        def state_dict(self): return {}

        def load_state_dict(self, state_dict): pass

try:
    import matplotlib.pyplot as plt

    _matplotlib_available = True
except ImportError:
    _matplotlib_available = False

try:
    from tqdm.auto import tqdm

    _tqdm_available = True
except ImportError:
    _tqdm_available = False


    # Define dummy tqdm if not available
    def tqdm(iterable=None, *args, **kwargs):
        if iterable is not None:
            return iterable

        # Return a dummy object that mimics tqdm's basic interface
        class DummyTqdm:
            def __init__(self, *args, **kwargs): pass

            def update(self, n=1): pass

            def close(self): pass

            def set_description(self, desc=None): pass

            def set_postfix(self, ordered_dict=None, **kwargs): pass

            def set_postfix_str(self, s=''): pass

            def __enter__(self): return self

            def __exit__(self, *args): pass

            # Add a static write method if needed by callbacks etc.
            @staticmethod
            def write(*args, **kwargs):
                print(*args, **kwargs)

        return DummyTqdm(*args, **kwargs)


# --- Helper for DDP metric aggregation ---
def _aggregate_metrics(metrics_dict: Dict[str, float], world_size: int, device: torch.device) -> Dict[str, float]:
    """Aggregates metric values across DDP ranks using average."""
    if world_size <= 1:
        return metrics_dict

    aggregated_metrics = {}
    for name, value in metrics_dict.items():
        # Handle potential NaN values - don't reduce them or use a placeholder?
        # Option 1: Skip NaN reduction (might lead to missing metric on rank 0)
        # Option 2: Reduce count of non-NaNs and sum, then divide. More complex.
        # Option 3: Reduce as is, hoping NaN doesn't propagate badly (risky).
        # Let's stick to simple average, assuming NaNs are infrequent errors handled elsewhere.
        # If a value is NaN on one rank, the average might become NaN.
        metric_tensor = torch.tensor(value, device=device, dtype=torch.float32)
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.AVG)
        aggregated_metrics[name] = metric_tensor.item()
    return aggregated_metrics


def _aggregate_loss(loss_value: float, world_size: int, device: torch.device) -> float:
    """Aggregates loss value across DDP ranks using average."""
    if world_size <= 1 or math.isnan(loss_value):  # Don't aggregate if not distributed or NaN
        return loss_value
    loss_tensor = torch.tensor(loss_value, device=device, dtype=torch.float32)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
    return loss_tensor.item()


# ======================== NNHandler Class Definition ========================
class NNHandler:
    r"""A comprehensive wrapper for PyTorch neural networks enabling streamlined training, validation, and management,
       including support for multi-GPU and multi-node distributed training via DDP.

    NNHandler provides an interface to manage the entire lifecycle of a PyTorch model,
    including setup, training, validation, metric tracking, checkpointing, and inference.
    It supports various model types, custom components (loss, optimizer, scheduler, metrics),
    advanced training techniques (EMA, AMP, gradient accumulation, DDP), and a flexible callback system.

    Attributes:
        _metrics (Dict[str, Callable]): Dictionary of metric functions {name: function}.
        _train_metrics_history (defaultdict[str, List]): History of training metrics per epoch (aggregated on rank 0).
        _val_metrics_history (defaultdict[str, List]): History of validation metrics per epoch (aggregated on rank 0).
        _callbacks (List[Callback]): List of callbacks attached to the handler.
        _stop_training (bool): Flag used by callbacks (like EarlyStopping) to signal training termination.
        _grad_scaler (GradScaler): Gradient scaler for Automatic Mixed Precision (AMP).
        _ema (Optional[ExponentialMovingAverage]): Exponential Moving Average handler.
        _distributed (bool): Flag indicating if DDP is active.
        _rank (int): Process rank in the distributed group (0 if not distributed).
        _local_rank (int): Local rank on the node (-1 if not distributed or using CPU).
        _world_size (int): Total number of processes in the distributed group (1 if not distributed).
        _train_sampler (Optional[DistributedSampler]): Sampler for distributed training data.
        _val_sampler (Optional[DistributedSampler]): Sampler for distributed validation data.

    Args:
        model_class (type[nn.Module]): The PyTorch model class to train.
        device (Union[torch.device, str]): The target device ('cpu', 'cuda'). In DDP mode,
            this is often ignored in favor of the device assigned by the DDP setup based on local rank.
        logger_mode (Optional[NNHandler.LoggingMode]): Logging configuration (only active on rank 0).
        logger_filename (str): Filename for file logging (rank 0 only).
        logger_level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        save_model_code (bool): Attempt to save model source code (rank 0 only).
        model_type (Union[ModelType, str]): Type of model (influences training loop logic).
        use_distributed (bool | None): Explicitly enable/disable DDP. If None, auto-detects
            from environment variables ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'). Defaults to None.
        **model_kwargs: Keyword arguments passed to the model constructor.

    Raises:
        RuntimeError: If DDP initialization fails or essential components are missing during training.
        TypeError: For invalid argument types.
        ValueError: For invalid argument values.
    """

    @property
    def version(self):
        """Returns the version of the package."""
        return __version__

    class ModelType(Enum):
        CLASSIFICATION = "classification"
        GENERATIVE = "generative"
        REGRESSION = "regression"
        SCORE_BASED = "score_based"

        @classmethod
        def from_string(cls, s: str) -> 'NNHandler.ModelType':
            """Converts a string to a ModelType enum member."""
            try:
                return cls(s.lower())
            except ValueError:
                valid_types = [mt.value for mt in cls]
                raise ValueError(f"Invalid model_type string '{s}'. Valid options are: {valid_types}")

    class LoggingMode(Enum):
        CONSOLE = "console"
        FILE = "file"
        BOTH = "both"

    # --- Core Attributes (Initialized in __init__) ---
    _optimizer: Optional[torch.optim.Optimizer]
    _optimizer_kwargs: Dict[str, Any]
    _scheduler: Optional[torch.optim.lr_scheduler.LRScheduler]  # Using base class for broader type hint
    _scheduler_kwargs: Dict[str, Any]
    _loss_fn: Optional[Callable]
    _loss_fn_kwargs: Dict[str, Any]
    _pass_epoch_to_loss: bool
    _train_loader: Optional[DataLoader]
    _train_loader_kwargs: Dict[str, Any]
    _train_dataset: Optional[Dataset]  # Keep track of original dataset
    _val_loader: Optional[DataLoader]
    _val_loader_kwargs: Dict[str, Any]
    _val_dataset: Optional[Dataset]  # Keep track of original dataset
    _model: Optional[nn.Module]
    _model_class: Optional[type[nn.Module]]
    _model_kwargs: Optional[Dict[str, Any]]
    _model_type: ModelType
    _compiled_model: bool
    _sampler: Optional[Sampler]  # Custom sampler
    _sampler_kwargs: Dict[str, Any]
    _sde: Optional[Any]  # Assuming SDE class structure
    _sde_kwargs: Dict[str, Any]
    _device: torch.device  # Resolved device
    _seed: Optional[int]
    _auto_saver: AutoSaver
    _ema: Optional[ExponentialMovingAverage]
    _ema_decay: float
    _train_losses: List[float]  # Aggregated history on rank 0
    _val_losses: List[float]  # Aggregated history on rank 0
    _metrics: Dict[str, Callable]
    _train_metrics_history: Dict[str, List[float]]  # Aggregated history on rank 0
    _val_metrics_history: Dict[str, List[float]]  # Aggregated history on rank 0
    _callbacks: List[Callback]
    _stop_training: bool  # Flag for early stopping, needs broadcast
    _grad_scaler: GradScaler

    # --- DDP Specific Attributes ---
    _distributed: bool
    _rank: int
    _local_rank: int
    _world_size: int
    _train_sampler: Optional[DistributedSampler]  # For setting epoch
    _val_sampler: Optional[DistributedSampler]  # For setting epoch

    __logger: Optional[logging.Logger]  # Logger instance

    def __init__(self,
                 model_class: type[nn.Module],
                 device: Union[torch.device, str] = "cpu",
                 logger_mode: Optional[LoggingMode] = None,
                 logger_filename: str = "NNHandler.log",
                 logger_level: int = logging.INFO,
                 save_model_code: bool = False,
                 model_type: Union[ModelType, str] = ModelType.CLASSIFICATION,
                 use_distributed: Optional[bool] = None,  # DDP control flag
                 **model_kwargs):

        # --- Determine DDP Status and Initialize Process Group ---
        self._distributed = _should_use_distributed(use_distributed)
        if self._distributed:
            self._distributed, self._rank, self._local_rank, self._world_size, self._device = _initialize_distributed()  # Sets self._rank, self._local_rank, self._world_size, self._device
        else:
            self._rank = 0
            self._local_rank = -1  # Convention for non-distributed or CPU
            self._world_size = 1
            self._device = _resolve_device(device)  # Resolve device if not DDP

        # --- Initialize Standard Attributes ---
        self._optimizer = None
        self._optimizer_kwargs = {}
        self._scheduler = None
        self._scheduler_kwargs = {}
        self._loss_fn = None
        self._loss_fn_kwargs = {}
        self._pass_epoch_to_loss = False
        self._train_loader = None
        self._train_loader_kwargs = {}
        self._train_dataset = None
        self._val_loader = None
        self._val_loader_kwargs = {}
        self._val_dataset = None
        self._model = None  # Will be set by set_model
        self._model_class = model_class
        self._model_kwargs = model_kwargs
        self._compiled_model = False
        self._sampler = None
        self._sampler_kwargs = {}
        self._sde = None
        self._sde_kwargs = {}
        self._seed = None
        self._ema = None
        self._ema_decay = 0.0
        self._train_losses = []
        self._val_losses = []
        self._metrics = {}
        # Use defaultdict for simpler accumulation later
        self._train_metrics_history = defaultdict(list)
        self._val_metrics_history = defaultdict(list)
        self._callbacks = []
        self._stop_training = False
        self._grad_scaler = GradScaler(enabled=False)  # Will be enabled in train() if needed
        self._train_sampler = None  # Initialize DDP samplers
        self._val_sampler = None
        self._modules_always_eval = []
        self.__logger = None  # Initialize logger

        # --- Model Type ---
        if isinstance(model_type, str):
            self._model_type = self.ModelType.from_string(model_type)
        elif isinstance(model_type, self.ModelType):
            self._model_type = model_type
        else:
            raise TypeError(f"model_type must be NNHandler.ModelType or str, got {type(model_type)}")

        # --- Logger Initialization (Rank 0 Only) ---
        if logger_mode is not None and self._rank == 0:
            self.initialize_logger(logger_mode, filename=logger_filename, level=logger_level)

        # Log initial status (rank 0)
        if self.__logger:
            self.__logger.info(f"--- NNHandler Initialization (Rank {self._rank}) ---")
            self.__logger.info(f"  Model Class:         {self._model_class.__name__}")
            self.__logger.info(f"  Model Type:          {self._model_type.name}")
            self.__logger.info(f"  Distributed (DDP):   {self._distributed}")
            if self._distributed:
                self.__logger.info(f"  World Size:          {self._world_size}")
                self.__logger.info(f"  Global Rank:         {self._rank}")
                self.__logger.info(f"  Local Rank:          {self._local_rank}")
            self.__logger.info(f"  Target Device:       {self._device}")
            self.__logger.info(f"  AMP Available:       {_amp_available}")
            self.__logger.info(f"  EMA Available:       {_ema_available}")

        # --- Initialize AutoSaver ---
        # Only save code on rank 0
        self._auto_saver = AutoSaver(save_model_code=(save_model_code and self._rank == 0))

        # --- Initialize Model (after DDP setup and logging) ---
        # This needs to happen after device is set, potentially wraps with DDP
        self.set_model(model_class=self._model_class,
                       save_model_code=save_model_code,
                       model_type=self._model_type,  # Pass resolved type
                       **self._model_kwargs)

        # Ensure all ranks are synchronized after initialization
        if self._distributed:
            if self.__logger:
                self.__logger.debug(f"Rank {self._rank} waiting at init barrier.")
            dist.barrier()
            if self.__logger:
                self.__logger.debug(f"Rank {self._rank} passed init barrier.")

    def initialize_logger(self, mode: LoggingMode = LoggingMode.CONSOLE, filename: str = "NNHandler.log",
                          level: int = logging.INFO):
        """Initializes the logger for the NNHandler class (only on rank 0)."""
        # Ensure logger is only configured on rank 0
        if self._rank != 0:
            self.__logger = None  # Explicitly disable logger on non-zero ranks
            return

        logger_name = f"NNHandler_R{self._rank}_{self._model_class.__name__ if self._model_class else 'NoModel'}"

        # Avoid adding multiple handlers if called again on rank 0
        if logging.getLogger(logger_name).hasHandlers():
            self.__logger = logging.getLogger(logger_name)  # Use existing logger
            # Optionally remove old handlers if reconfiguration is desired
            # for handler in self.__logger.handlers[:]: self.__logger.removeHandler(handler)
            return

        self.__logger = logging.getLogger(logger_name)
        self.__logger.setLevel(level)
        # Avoid propagating logs to root logger if handlers are added here
        self.__logger.propagate = False

        formatter = logging.Formatter(
            "[%(levelname)s|%(asctime)s|%(name)s.%(funcName)s:%(lineno)d] %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        handlers_added = 0
        # Console Handler
        if mode in [self.LoggingMode.CONSOLE, self.LoggingMode.BOTH]:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            self.__logger.addHandler(console_handler)
            handlers_added += 1

        # File Handler
        if mode in [self.LoggingMode.FILE, self.LoggingMode.BOTH]:
            try:
                log_dir = os.path.dirname(filename)
                if log_dir:  # Create log directory if it doesn't exist
                    os.makedirs(log_dir, exist_ok=True)
                # Use 'a' mode to append if file exists
                file_handler = logging.FileHandler(filename, mode='a')
                file_handler.setLevel(level)
                file_handler.setFormatter(formatter)
                self.__logger.addHandler(file_handler)
                handlers_added += 1
            except OSError as e:
                # Use print directly as logger might not be fully set up
                print(f"WARNING (Rank {self._rank}): Failed to create log file handler for {filename}: {e}",
                      file=sys.stderr)

        if handlers_added > 0:
            self.__logger.info(f"Logger initialized (mode: {mode.name}, level: {logging.getLevelName(level)}).")
        else:
            print(f"WARNING (Rank {self._rank}): Logger initialization requested but no handlers were added.",
                  file=sys.stderr)
            self.__logger = None  # Disable logger if setup failed

    # --- Properties & Setters (Adjusted for DDP) ---

    @property
    def device(self) -> torch.device:
        """Returns the device assigned to this handler/rank."""
        return self._device

    @device.setter
    def device(self, value: Union[torch.device, str]):
        """Sets the computation device (disabled in DDP mode after init)."""
        if self._distributed:
            resolved_value = _resolve_device(value)
            if resolved_value != self._device:
                # Log warning only on rank 0 to avoid spam
                if self._rank == 0:
                    warnings.warn(
                        f"Cannot change device after DDP initialization. Device remains '{self._device}'. Attempted to set to '{resolved_value}'.",
                        RuntimeWarning)
            # Do not change the device in DDP mode
            return

        # Original logic for non-distributed mode
        resolved_device = _resolve_device(value)
        needs_update = resolved_device != self._device or self._model is None

        if needs_update:
            self._device = resolved_device
            if self.__logger:
                self.__logger.info(f"Device set to '{self._device}'.")
            # Move model if it exists
            if self._model:
                self._model.to(self._device)
            # Re-evaluate DataParallel wrapping (only relevant if not distributed)
            self._wrap_model_dataparallel()

    def _wrap_model_dataparallel(self):
        """Wraps/unwraps model with nn.DataParallel for non-DDP multi-GPU."""
        # This function should only run if not in DDP mode
        if self._distributed or not self._model:
            return

        is_multi_gpu_cuda = self._device.type == 'cuda' and torch.cuda.device_count() > 1
        # Get the raw underlying model (handles if already wrapped)
        raw_model = self.module
        is_already_dp = isinstance(self._model, nn.DataParallel)

        if is_multi_gpu_cuda and not is_already_dp:
            # Wrap with DataParallel
            if self.__logger:
                self.__logger.info(f"Detected {torch.cuda.device_count()} GPUs. Wrapping model with nn.DataParallel.")
            self._model = nn.DataParallel(raw_model)
        elif not is_multi_gpu_cuda and is_already_dp:
            # Unwrap from DataParallel
            if self.__logger:
                self.__logger.info("Device changed or single GPU detected. Unwrapping model from nn.DataParallel.")
            self._model = raw_model  # Assign the unwrapped module back

    @property
    def seed(self) -> Optional[int]:
        return self._seed

    @seed.setter
    def seed(self, seed_value: Optional[int]):
        """Sets the random seed for torch and CUDA (applied on all ranks)."""
        if seed_value is not None:
            if not isinstance(seed_value, int):
                message = f"Seed must be an integer or None, got {type(seed_value)}."
                # Log error only on rank 0
                if self.__logger: self.__logger.error(message)
                raise TypeError(message)

            # Set seed on all ranks to ensure consistent initialization where needed
            # (e.g., model weights before DDP syncs them, random operations in datasets)
            torch.manual_seed(seed_value)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_value)  # Seed all GPUs relevant to this process

            # Log info only on rank 0
            if self.__logger:
                self.__logger.info(f"Global random seed set to: {seed_value} (applied on rank {self._rank})")

        self._seed = seed_value

    @property
    def logger(self) -> logging.Logger:
        return self.__logger

    @property
    def model(self) -> Optional[nn.Module]:
        """Returns the model instance (potentially wrapped by DDP or DataParallel)."""
        return self._model

    @property
    def module(self) -> nn.Module:
        """Returns the underlying nn.Module, unwrapping DDP or DataParallel if necessary."""
        if self._model is None:
            raise RuntimeError("Model has not been set.")
        if isinstance(self._model, (DDP, nn.DataParallel)):
            return self._model.module
        return self._model

    @property
    def modules_always_eval(self):
        return self._modules_always_eval

    @property
    def model_kwargs(self) -> Optional[Dict[str, Any]]:
        return self._model_kwargs

    @property
    def model_code(self) -> Optional[str]:
        # AutoSaver handles rank check internally for saving code
        return self._auto_saver.model_code

    def set_model(self, model_class: type[nn.Module], save_model_code: bool = False,
                  model_type: Optional[Union[ModelType, str]] = None, **model_kwargs):
        """Sets or replaces the model, handling DDP/DataParallel wrapping."""
        if not issubclass(model_class, nn.Module):
            message = f"model_class must be a subclass of torch.nn.Module, got {model_class}."
            if self.__logger: self.__logger.error(message)
            raise TypeError(message)

        # Resolve model type
        if model_type is not None:
            if isinstance(model_type, str):
                self._model_type = self.ModelType.from_string(model_type)
            elif isinstance(model_type, self.ModelType):
                self._model_type = model_type
            # Handle case where it might be an enum member from another definition
            elif issubclass(model_type.__class__, Enum) and hasattr(model_type, 'value') and isinstance(
                    model_type.value, str):
                try:
                    self._model_type = self.ModelType.from_string(model_type.value)
                except ValueError:
                    raise TypeError(f"Invalid model_type enum value: {model_type.value}")
            else:
                raise TypeError(f"model_type must be NNHandler.ModelType or str, got {type(model_type)}")

        self._model_class = model_class
        self._model_kwargs = model_kwargs

        # Instantiate the base model on the correct device for this rank
        base_model = model_class(**model_kwargs).to(self._device)
        self._compiled_model = False  # Reset compiled flag

        # --- Wrap model based on mode ---
        if self._distributed:
            # Wrap with DDP
            # Determine find_unused_parameters based on environment variable or default
            ddp_find_unused = os.environ.get("DDP_FIND_UNUSED_PARAMETERS", "false").lower() == "true"
            if self._device.type == 'cuda':
                # Ensure device_ids is a list containing the local rank
                self._model = DDP(base_model, device_ids=[self._local_rank], output_device=self._local_rank,
                                  find_unused_parameters=ddp_find_unused)
            else:
                # DDP on CPU doesn't use device_ids or output_device
                self._model = DDP(base_model, find_unused_parameters=ddp_find_unused)

            # Log DDP wrapping details (rank 0)
            if self.__logger:
                self.__logger.info(
                    f"Wrapped model with DDP (Rank {self._rank}, Device: {self._device}, find_unused={ddp_find_unused}).")
        else:
            # Not distributed, use original model and potentially wrap with DataParallel later
            self._model = base_model
            self._wrap_model_dataparallel()  # Check if DataParallel needed for non-DDP multi-GPU

        # Handle auto-saving code (rank 0 only)
        self._auto_saver.save_model_code = save_model_code and self._rank == 0
        if self._auto_saver.save_model_code:  # Condition already checks rank
            self._auto_saver.try_save_model_code(self._model_class, self.__logger)  # logger is None on non-zero ranks

        # Re-initialize optimizer and scheduler if they were already set
        if self._optimizer is not None:
            if self.__logger: self.__logger.warning("Model replaced. Re-initializing optimizer with previous settings.")
            # Use saved class and kwargs
            self.set_optimizer(self._optimizer.__class__, **self._optimizer_kwargs)
            # Scheduler is re-initialized within set_optimizer if it was set

        # Log final status (rank 0)
        if self.__logger:
            param_count = self.count_parameters()
            self.__logger.info(f"Model set to {model_class.__name__} (Type: {self._model_type.name}).")
            self.__logger.info(f"Model contains {param_count:,} trainable parameters.")

    @property
    def optimizer(self) -> Optional[torch.optim.Optimizer]:
        return self._optimizer

    def set_optimizer(self, optimizer_class: type[torch.optim.Optimizer], **optimizer_kwargs):
        """Sets the optimizer and re-initializes the scheduler if present."""
        if not issubclass(optimizer_class, torch.optim.Optimizer):
            message = f"optimizer_class must be a subclass of torch.optim.Optimizer, got {optimizer_class}."
            if self.__logger: self.__logger.error(message)
            raise TypeError(message)
        if self._model is None:
            message = "Model must be set before setting the optimizer."
            if self.__logger: self.__logger.error(message)
            raise RuntimeError(message)

        self._optimizer_kwargs = optimizer_kwargs
        # Pass the parameters of the potentially wrapped model
        # DDP/DataParallel correctly handle requests for .parameters()
        self._optimizer = optimizer_class(self.model.parameters(), **optimizer_kwargs)

        # Log on rank 0
        if self.__logger:
            self.__logger.info(f"Optimizer set to {optimizer_class.__name__} with kwargs: {optimizer_kwargs}")

        # Re-initialize scheduler with the new optimizer if it exists
        if self._scheduler is not None:
            if self.__logger:
                self.__logger.info("Re-initializing scheduler with new optimizer.")
            scheduler_class = self._scheduler.__class__  # Get class before overwriting
            self.set_scheduler(scheduler_class, **self._scheduler_kwargs)

    @property
    def scheduler(self) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        return self._scheduler

    def set_scheduler(self, scheduler_class: Optional[type[torch.optim.lr_scheduler.LRScheduler]], **scheduler_kwargs):
        """Sets the learning rate scheduler."""
        if scheduler_class is None:
            self._scheduler = None
            self._scheduler_kwargs = {}
            if self.__logger:  # Log on rank 0
                self.__logger.info("Scheduler removed.")
            return

        # Check if it's a valid scheduler type (_LRScheduler or ReduceLROnPlateau or new LRScheduler base)
        # Using LRScheduler base class covers most modern schedulers
        if not issubclass(scheduler_class, torch.optim.lr_scheduler.LRScheduler):
            # Keep specific checks for older common types just in case
            if not (issubclass(scheduler_class, torch.optim.lr_scheduler._LRScheduler) or
                    issubclass(scheduler_class, torch.optim.lr_scheduler.ReduceLROnPlateau)):
                message = (f"scheduler_class {scheduler_class} must be a subclass of "
                           f"torch.optim.lr_scheduler.LRScheduler (or older base classes).")
                if self.__logger: self.__logger.error(message)
                raise TypeError(message)

        if self._optimizer is None:
            message = "Optimizer must be set before setting the scheduler."
            if self.__logger: self.__logger.error(message)
            raise ValueError(message)  # Changed to ValueError for consistency

        self._scheduler_kwargs = scheduler_kwargs
        self._scheduler = scheduler_class(self._optimizer, **scheduler_kwargs)

        # Log on rank 0
        if self.__logger:
            self.__logger.info(f"Scheduler set to {scheduler_class.__name__} with kwargs: {scheduler_kwargs}")

    # --- SDE/Sampler Properties ---
    # No changes needed specifically for DDP in setters, but usage might need care.
    @property
    def sde(self) -> Optional[Any]:
        return self._sde

    @sde.setter
    def sde(self, sde_instance: Any):
        """Sets the SDE instance directly."""
        # Basic check if it has expected methods/attributes
        expected_attrs = ['prior', 'drift', 'diffusion', 'sigma', 'T', 'epsilon']
        if not all(hasattr(sde_instance, attr) for attr in expected_attrs):
            warnings.warn("Provided SDE instance might be missing expected attributes/methods "
                          f"(e.g., {expected_attrs}).", RuntimeWarning)

        if self._model_type != self.ModelType.SCORE_BASED:
            if self.__logger:  # Log on rank 0
                self.__logger.warning(f"Model Type was {self._model_type.name}. Changed to SCORE_BASED as SDE was set.")
            self._model_type = self.ModelType.SCORE_BASED

        self._sde = sde_instance
        self._sde_kwargs = {}  # Clear kwargs if instance is set directly
        if self.__logger:  # Log on rank 0
            self.__logger.info(f"SDE instance set to: {sde_instance}")

    def set_sde(self, sde_class: type, **sde_kwargs):
        """Sets the SDE by providing the class and keyword arguments."""
        # Could add more checks on sde_class if an SDE ABC/protocol exists
        if self._model_type != self.ModelType.SCORE_BASED:
            if self.__logger:  # Log on rank 0
                self.__logger.warning(f"Model Type was {self._model_type.name}. Changed to SCORE_BASED as SDE was set.")
            self._model_type = self.ModelType.SCORE_BASED

        try:
            self._sde = sde_class(**sde_kwargs)
            self._sde_kwargs = sde_kwargs
            if self.__logger:  # Log on rank 0
                self.__logger.info(f"SDE set to {sde_class.__name__} with kwargs: {sde_kwargs}")
        except Exception as e:
            message = f"Failed to instantiate SDE class {sde_class.__name__} with kwargs {sde_kwargs}: {e}"
            if self.__logger: self.__logger.error(message, exc_info=True)
            raise RuntimeError(message) from e

    @property
    def sampler(self) -> Optional[Sampler]:  # Custom Sampler base class
        return self._sampler

    @sampler.setter
    def sampler(self, sampler_instance: Sampler):
        """Sets the Sampler instance directly."""
        # Check against the imported custom Sampler base class
        if not isinstance(sampler_instance, Sampler):
            message = "sampler must be an instance of a class inheriting from the custom Sampler base class."
            if self.__logger: self.__logger.error(message)
            raise TypeError(message)
        self._sampler = sampler_instance
        self._sampler_kwargs = {}  # Clear kwargs
        if self.__logger:  # Log on rank 0
            self.__logger.info(f"Custom sampler instance set to: {sampler_instance}")

    def set_sampler(self, sampler_class: type[Sampler], **sampler_kwargs):
        """Sets the Sampler by providing the class and keyword arguments."""
        # Check against the imported custom Sampler base class
        if not issubclass(sampler_class, Sampler):
            message = f"sampler_class {sampler_class.__name__} must be a subclass of the custom Sampler base class."
            if self.__logger: self.__logger.error(message)
            raise TypeError(message)

        try:
            self._sampler = sampler_class(**sampler_kwargs)
            self._sampler_kwargs = sampler_kwargs
            if self.__logger:  # Log on rank 0
                self.__logger.info(f"Custom sampler set to {sampler_class.__name__} with kwargs: {sampler_kwargs}")
        except Exception as e:
            message = f"Failed to instantiate Sampler class {sampler_class.__name__} with kwargs {sampler_kwargs}: {e}"
            if self.__logger: self.__logger.error(message, exc_info=True)
            raise RuntimeError(message) from e

    def get_samples(self, N, device=None):
        """Generates samples using the custom sampler (intended for rank 0)."""
        target_device = self.device if device is None else _resolve_device(device)
        if self._sampler is None:
            raise RuntimeError("Custom Sampler has not been set.")

        # Sample generation is typically done on rank 0 to avoid redundant computation
        # unless the sampler itself is designed for distributed operation.
        if self._distributed and self._rank != 0:
            # Non-zero ranks do nothing and return None
            # A barrier might be needed later if rank 0 needs to wait for others
            # before proceeding, but for just getting samples, maybe not.
            return None

        if self.__logger:
            self.__logger.info(
                f"Generating {N} samples using {type(self._sampler).__name__} on device {target_device} (Rank {self._rank}).")

        # Perform sampling only on rank 0 (or all ranks if sampler handles it)
        samples = self._sampler.sample(N, device=target_device)
        return samples

    # --- Loss Function Properties --- (No DDP specific changes needed)
    @property
    def loss_fn(self) -> Optional[Callable]:
        return self._loss_fn

    @loss_fn.setter
    def loss_fn(self, loss_function: Callable):
        """Sets the loss function. Use set_loss_fn for kwargs."""
        if not callable(loss_function):
            message = "loss_fn must be a callable function."
            if self.__logger: self.__logger.error(message)
            raise TypeError(message)
        self._loss_fn = loss_function
        self._loss_fn_kwargs = {}  # Reset kwargs when set directly
        self._pass_epoch_to_loss = False  # Reset flag
        if self.__logger:  # Log on rank 0
            fn_name = getattr(loss_function, '__name__', repr(loss_function))
            self.__logger.info(f"Loss function set to {fn_name}.")

    def set_loss_fn(self, loss_fn: Callable, pass_epoch_to_loss: bool = False, **kwargs):
        """Sets the loss function with optional kwargs and epoch passing flag."""
        if not callable(loss_fn):
            message = "loss_fn must be a callable function."
            if self.__logger: self.__logger.error(message)
            raise TypeError(message)
        self._loss_fn = loss_fn
        self._pass_epoch_to_loss = pass_epoch_to_loss
        self._loss_fn_kwargs = kwargs or {}
        if self.__logger:  # Log on rank 0
            fn_name = getattr(loss_fn, '__name__', repr(loss_fn))
            self.__logger.info(f"Loss function set to {fn_name} with kwargs: {self._loss_fn_kwargs}.")
            if pass_epoch_to_loss:
                self.__logger.info("Current epoch will be passed to the loss function if it accepts 'epoch' kwarg.")

    @property
    def pass_epoch_to_loss(self) -> bool:
        return self._pass_epoch_to_loss

    @pass_epoch_to_loss.setter
    def pass_epoch_to_loss(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("pass_epoch_to_loss must be a boolean.")
        self._pass_epoch_to_loss = value
        if self.__logger:  # Log on rank 0
            self.__logger.info(f"pass_epoch_to_loss set to {value}.")

    @property
    def loss_fn_kwargs(self) -> Dict[str, Any]:
        return self._loss_fn_kwargs

    @loss_fn_kwargs.setter
    def loss_fn_kwargs(self, value: Dict[str, Any]):
        if not isinstance(value, dict):
            raise TypeError("loss_fn_kwargs must be a dictionary.")
        self._loss_fn_kwargs = value or {}
        if self.__logger:  # Log on rank 0
            self.__logger.info(f"Loss function kwargs updated to: {self._loss_fn_kwargs}")

    # --- Data Loaders (Modified for DDP) ---
    @property
    def train_loader(self) -> Optional[DataLoader]:
        return self._train_loader

    @property
    def train_loader_kwargs(self) -> Dict[str, Any]:
        return self._train_loader_kwargs

    def _create_distributed_loader(self, dataset: Dataset, loader_kwargs: Dict[str, Any], is_eval: bool = False) -> \
            Tuple[DataLoader, DistributedSampler]:
        """Creates a DataLoader with a DistributedSampler for DDP."""
        if not self._distributed:
            # This should not be called in non-distributed mode, but check defensively
            raise RuntimeError("Internal Error: _create_distributed_loader called in non-DDP mode.")

        # Determine shuffle and drop_last for the sampler
        # Training: usually shuffle=True, drop_last=True (recommended for DDP)
        # Eval: shuffle=False, drop_last=False (usually)
        default_shuffle = not is_eval
        shuffle = loader_kwargs.get('shuffle', default_shuffle)  # User can override default

        # DDP often benefits from drop_last=True during training to avoid hangs
        # if the last batch is smaller and syncs incorrectly. For eval, usually False.
        default_drop_last = not is_eval
        drop_last = loader_kwargs.get('drop_last', default_drop_last)  # User can override

        # Create the DistributedSampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=self._world_size,
            rank=self._rank,
            shuffle=shuffle,
            drop_last=drop_last
        )

        # Prepare DataLoader kwargs
        new_loader_kwargs = loader_kwargs.copy()
        # Sampler replaces shuffle and batch_sampler
        new_loader_kwargs['sampler'] = sampler
        new_loader_kwargs['shuffle'] = False  # Sampler handles shuffling
        if 'batch_sampler' in new_loader_kwargs:
            del new_loader_kwargs['batch_sampler']  # Sampler is mutually exclusive
        new_loader_kwargs['drop_last'] = drop_last  # Ensure DataLoader knows drop_last

        # Set sensible defaults if not provided
        new_loader_kwargs.setdefault('batch_size', 1)  # Need a default batch size
        # Default num_workers based on SLURM env var or 0
        new_loader_kwargs.setdefault('num_workers', int(os.environ.get("SLURM_CPUS_PER_TASK", 0)))
        # Enable pin_memory if using CUDA device
        new_loader_kwargs.setdefault('pin_memory', self._device.type == 'cuda')
        # Set persistent_workers based on num_workers
        new_loader_kwargs.setdefault('persistent_workers', new_loader_kwargs['num_workers'] > 0)

        # Create the DataLoader
        new_loader = DataLoader(dataset, **new_loader_kwargs)

        if self.__logger:  # Log on rank 0
            self.__logger.info(
                f"Created DDP DataLoader ({'Eval' if is_eval else 'Train'}) for {type(dataset).__name__}: "
                f"Shuffle={shuffle}, DropLast={drop_last}, BatchSize={loader_kwargs['batch_size']}, "
                f"Workers={new_loader_kwargs['num_workers']}, PinMem={new_loader_kwargs['pin_memory']}, "
                f"PersistWrk={new_loader_kwargs['persistent_workers']}")

        return new_loader, sampler

    def set_train_loader(self, dataset: Dataset, **loader_kwargs):
        """Sets the training data loader, using DistributedSampler in DDP mode."""
        if not isinstance(dataset, Dataset):
            raise TypeError(f"dataset must be an instance of torch.utils.data.Dataset, got {type(dataset)}")

        self._train_dataset = dataset  # Store original dataset
        self._train_loader_kwargs = loader_kwargs
        self._train_sampler = None  # Reset sampler

        if self._distributed:
            # Create loader with DistributedSampler
            self._train_loader, self._train_sampler = self._create_distributed_loader(dataset, loader_kwargs,
                                                                                      is_eval=False)
        else:
            # Standard non-distributed loader
            # Set defaults for non-DDP case
            loader_kwargs.setdefault('shuffle', True)  # Usually shuffle training data
            loader_kwargs.setdefault('num_workers', 0)  # Default to 0 if not set
            loader_kwargs.setdefault('pin_memory', self._device.type == 'cuda')
            loader_kwargs.setdefault('persistent_workers', loader_kwargs['num_workers'] > 0)

            self._train_loader = DataLoader(dataset, **loader_kwargs)

        # Log on rank 0
        if self.__logger:
            mode = 'DDP' if self._distributed else 'Standard'
            self.__logger.info(f"Train DataLoader ({mode}) set for {type(dataset).__name__}.")

    @property
    def val_loader(self) -> Optional[DataLoader]:
        return self._val_loader

    @property
    def val_loader_kwargs(self) -> Dict[str, Any]:
        return self._val_loader_kwargs

    def set_val_loader(self, dataset: Dataset, **loader_kwargs):
        """Sets the validation data loader, using DistributedSampler in DDP mode."""
        if not isinstance(dataset, Dataset):
            raise TypeError(f"dataset must be an instance of torch.utils.data.Dataset, got {type(dataset)}")

        self._val_dataset = dataset  # Store original dataset
        self._val_loader_kwargs = loader_kwargs
        self._val_sampler = None  # Reset sampler

        if self._distributed:
            # Create loader with DistributedSampler for evaluation
            # Typically shuffle=False, drop_last=False for validation
            loader_kwargs.setdefault('shuffle', False)
            loader_kwargs.setdefault('drop_last', False)
            self._val_loader, self._val_sampler = self._create_distributed_loader(dataset, loader_kwargs, is_eval=True)
        else:
            # Standard non-distributed loader
            loader_kwargs.setdefault('shuffle', False)  # No shuffle for validation
            loader_kwargs.setdefault('num_workers', 0)
            loader_kwargs.setdefault('pin_memory', self._device.type == 'cuda')
            loader_kwargs.setdefault('persistent_workers', loader_kwargs['num_workers'] > 0)

            self._val_loader = DataLoader(dataset, **loader_kwargs)

        # Log on rank 0
        if self.__logger:
            mode = 'DDP' if self._distributed else 'Standard'
            self.__logger.info(f"Validation DataLoader ({mode}) set for {type(dataset).__name__}.")

    # --- Metrics --- (Aggregation happens in train loop)
    @property
    def metrics(self) -> Dict[str, Callable]:
        return self._metrics

    def add_metric(self, name: str, metric_fn: Callable):
        """Adds a metric function to be tracked during training and validation."""
        if not callable(metric_fn):
            raise TypeError("metric_fn must be callable.")
        if not isinstance(name, str) or not name:
            raise ValueError("Metric name must be a non-empty string.")

        self._metrics[name] = metric_fn
        # Initialize history lists (will store aggregated results on rank 0)
        self._train_metrics_history[name] = []
        self._val_metrics_history[name] = []
        if self.__logger:  # Log on rank 0
            self.__logger.info(f"Added metric '{name}'.")

    def clear_metrics(self):
        """Removes all tracked metrics."""
        self._metrics.clear()
        self._train_metrics_history.clear()
        self._val_metrics_history.clear()
        if self.__logger:  # Log on rank 0
            self.__logger.info("All metrics cleared.")

    # --- History Properties (Return Rank 0 Aggregated History) ---
    @property
    def train_losses(self) -> List[float]:
        """Returns the history of aggregated training losses (rank 0 only)."""
        # Consider adding a check or warning if accessed by non-zero rank in DDP?
        # if self._distributed and self._rank != 0:
        #    warnings.warn("Accessing train_losses on non-zero rank in DDP mode. History is only stored on rank 0.")
        return self._train_losses

    @property
    def val_losses(self) -> List[float]:
        """Returns the history of aggregated validation losses (rank 0 only)."""
        return self._val_losses

    @property
    def train_metrics_history(self) -> Dict[str, List[float]]:
        """Returns the history of aggregated training metrics (rank 0 only)."""
        return dict(self._train_metrics_history)  # Return copy

    @property
    def val_metrics_history(self) -> Dict[str, List[float]]:
        """Returns the history of aggregated validation metrics (rank 0 only)."""
        return dict(self._val_metrics_history)  # Return copy

    # --- Auto Saving Properties (Delegate, saving done by rank 0) ---
    @property
    def save_interval(self) -> Optional[int]:
        return self._auto_saver.save_interval

    @save_interval.setter
    def save_interval(self, interval: Optional[int]):
        try:
            # Allow setting on all ranks, but saving only happens on rank 0
            self._auto_saver.save_interval = interval
            if self.__logger:  # Log on rank 0
                self.__logger.info(f"Auto-save interval set to {interval} epochs.")
        except (TypeError, ValueError) as e:
            if self.__logger:  # Log error on rank 0
                self.__logger.error(f"Failed to set save_interval: {e}")
            raise e

    @property
    def save_path(self) -> Optional[str]:
        return self._auto_saver.save_path

    @save_path.setter
    def save_path(self, path: Optional[str]):
        try:
            # Create directory only on rank 0 to avoid race conditions
            if path is not None and self._rank == 0:
                # Attempt to create the directory
                try:
                    os.makedirs(path, exist_ok=True)
                    if self.__logger:
                        self.__logger.info(f"Ensured auto-save directory exists: {path}")
                except OSError as e:
                    message = f"Save path '{path}' is not a valid directory and could not be created: {e}"
                    if self.__logger:
                        self.__logger.error(message)
                    raise ValueError(message) from e

            # Set path on all ranks (state needs to be consistent if saved/loaded)
            self._auto_saver.save_path = path
            if self.__logger:  # Log on rank 0
                self.__logger.info(f"Auto-save path set to '{path}'.")
        except TypeError as e:
            if self.__logger:  # Log error on rank 0
                self.__logger.error(f"Failed to set save_path: {e}")
            raise e

    @property
    def save_model_name(self) -> str:
        return self._auto_saver.save_model_name

    @save_model_name.setter
    def save_model_name(self, name: str):
        try:
            # Set on all ranks for consistency
            self._auto_saver.save_model_name = name
            if self.__logger:  # Log on rank 0
                self.__logger.info(f"Auto-save model name format set to '{name}'.")
        except TypeError as e:
            if self.__logger:  # Log error on rank 0
                self.__logger.error(f"Failed to set save_model_name: {e}")
            raise e

    @property
    def overwrite_last_saved(self) -> bool:
        return self._auto_saver.overwrite_last_saved

    @overwrite_last_saved.setter
    def overwrite_last_saved(self, overwrite: bool):
        try:
            # Set on all ranks for consistency
            self._auto_saver.overwrite_last_saved = overwrite
            if self.__logger:  # Log on rank 0
                self.__logger.info(f"Auto-save overwrite set to {overwrite}.")
        except TypeError as e:
            if self.__logger:  # Log error on rank 0
                self.__logger.error(f"Failed to set overwrite_last_saved: {e}")
            raise e

    def auto_save(self, interval: Optional[int], save_path: str = '.', name: str = "model_epoch{epoch:02d}",
                  overwrite: bool = False):
        """Configures periodic model saving (saving performed by rank 0)."""
        try:
            # Configure settings on all ranks
            self.save_interval = interval
            self.save_path = save_path
            self.save_model_name = name
            self.overwrite_last_saved = overwrite

            # Log configuration on rank 0
            if self.__logger:
                if interval is None or interval == 0:
                    self.__logger.info("Auto-save disabled.")
                else:
                    self.__logger.info(
                        f"Auto-save configured: Interval={interval}, Path='{save_path}', Name='{name}', Overwrite={overwrite}")
        except (TypeError, ValueError) as e:
            if self.__logger:  # Log error on rank 0
                self.__logger.error(f"Failed to configure auto_save: {e}")
            # Avoid raising here, just log error

    # --- Callbacks ---
    @property
    def callbacks(self) -> List[Callback]:
        return self._callbacks

    def add_callback(self, callback: Callback):
        """Adds a callback instance to the handler (all ranks)."""
        if not isinstance(callback, Callback):
            raise TypeError("callback must be an instance of the Callback base class.")

        # Callbacks should ideally be DDP-aware internally if they perform rank-specific actions
        callback.set_handler(self)  # Link handler to callback
        self._callbacks.append(callback)
        if self.__logger:  # Log on rank 0
            self.__logger.info(f"Added callback: {type(callback).__name__}")

    def _run_callbacks(self, method_name: str, *args, **kwargs):
        """Helper to run a specific method on all registered callbacks."""
        # Callbacks are run on all ranks unless they have internal rank checks
        for callback in self._callbacks:
            method = getattr(callback, method_name, None)
            if callable(method):
                try:
                    method(*args, **kwargs)
                except Exception as e:
                    # Log error only on rank 0 to avoid excessive output
                    if self._rank == 0:
                        err_msg = f"ERROR in callback {type(callback).__name__}.{method_name} on Rank {self._rank}: {e}"
                        if self.__logger:
                            self.__logger.error(err_msg, exc_info=True)  # Include traceback in log file
                        else:
                            print(err_msg, file=sys.stderr)
                            import traceback
                            traceback.print_exc()  # Print traceback to stderr if no logger

    # --- Training Loop Logic ---
    # _prepare_batch, _calculate_loss, _calculate_metrics operate locally
    # _train_step, _val_step operate locally and return local results

    def _prepare_batch(self, batch: Any) -> dict[str, Any | None]:
        """
        Moves batch to the rank's assigned device and formats it based on model type.
        Handles inputs, targets, and additional forward parameters.
        """
        inputs: Any
        targets: Optional[Any] = None
        additional_params = {}

        if self._model is None:
            raise RuntimeError("Model is not set.")

        # Get signature from the underlying module
        model_to_inspect = self.module
        try:
            model_sig = inspect.signature(model_to_inspect.forward)
            valid_param_names = list(model_sig.parameters.keys())[1:]  # Skip 'self', get names of forward args
        except (TypeError, ValueError):
            # Handle models without standard forward signature (e.g., functional) gracefully
            valid_param_names = []
            if self.__logger and self._rank == 0:  # Log warning once on rank 0
                self.__logger.debug(
                    f"Could not inspect forward signature of {type(model_to_inspect).__name__}. Extra batch items may not be passed correctly.")

        # --- Helper to move data recursively ---
        def _to_device(data):
            if isinstance(data, Tensor):
                return data.to(self._device, non_blocking=True)  # Use non_blocking for potential speedup
            elif isinstance(data, (list, tuple)):
                return type(data)(_to_device(d) for d in data)  # Preserve original type (list/tuple)
            elif isinstance(data, dict):
                return {k: _to_device(v) for k, v in data.items()}
            else:
                return data  # Keep non-tensor data as is

        # --- Process Batch based on structure and model type ---
        if isinstance(batch, (list, tuple)):
            if not batch:
                raise ValueError("Batch is empty.")

            inputs = _to_device(batch[0])
            extra_items_idx_start = 1  # Index of items after input

            # Determine targets based on model type
            if self._model_type in [self.ModelType.CLASSIFICATION, self.ModelType.REGRESSION]:
                if len(batch) < 2:
                    raise ValueError(
                        f"{self._model_type.name} model type expects batch to be a sequence (inputs, targets, ...). Got sequence of length {len(batch)}.")
                targets = _to_device(batch[1])
                extra_items_idx_start = 2
            elif self._model_type == self.ModelType.GENERATIVE:
                # Assumes input is also the target (e.g., autoencoder)
                targets = inputs  # Target is a reference to the input tensor on the device
            elif self._model_type == self.ModelType.SCORE_BASED:
                # Score-based models often don't need explicit targets here; loss handles it
                targets = None
            else:
                # Should be unreachable if model_type validation is correct
                raise ValueError(f"Unsupported ModelType: {self._model_type}")

            # Process remaining items in the batch as additional forward parameters
            extra_batch_items = batch[extra_items_idx_start:]
            num_extra_params_needed = len(valid_param_names)
            num_extra_items_given = len(extra_batch_items)

            if num_extra_items_given > num_extra_params_needed:
                # Log warning only on rank 0
                if self._rank == 0:
                    warnings.warn(
                        f"Batch contains {num_extra_items_given} extra items, but model forward() expects {num_extra_params_needed} after input. Ignoring excess items.",
                        RuntimeWarning)
                # Trim excess items
                extra_batch_items = extra_batch_items[:num_extra_params_needed]
                num_extra_items_given = len(extra_batch_items)  # Update count

            # Check if enough extra items were provided for required params
            # This check might be too strict if some params are optional with defaults
            # if num_extra_items_given < num_extra_params_needed:
            #    missing_params = valid_param_names[num_extra_items_given:]
            #    warnings.warn(f"Model forward() expects parameters {missing_params} which were not found in the batch.", RuntimeWarning)

            # Assign extra items to parameter names in order
            for i, item in enumerate(extra_batch_items):
                param_name = valid_param_names[i]
                additional_params[param_name] = _to_device(item)

        else:  # Assume batch is a single tensor (input)
            inputs = _to_device(batch)
            if self._model_type == self.ModelType.GENERATIVE:
                targets = inputs
            elif self._model_type in [self.ModelType.CLASSIFICATION, self.ModelType.REGRESSION]:
                raise ValueError(
                    f"{self._model_type.name} model type expects batch to be a sequence (inputs, targets, ...). Got a single tensor.")
            # Score-based handles single tensor input correctly (targets=None)

        return {"inputs": inputs, "targets": targets, **additional_params}

    def _calculate_loss(self, model_output: Any, targets: Optional[Any], inputs: Any, current_epoch: int) -> Union[
        Tensor, Tuple[Tensor, List[Tensor]]]:
        """Calculates the loss based on ModelType using local data."""
        if self._loss_fn is None:
            raise RuntimeError("Loss function not set.")

        loss_args = []
        # Start with configured kwargs, potentially add epoch later
        loss_kwargs = self._loss_fn_kwargs.copy()

        if self._model_type in [self.ModelType.CLASSIFICATION, self.ModelType.REGRESSION, self.ModelType.GENERATIVE]:
            if targets is None:  # Should not happen if _prepare_batch is correct
                raise RuntimeError(f"{self._model_type.name} requires targets for loss calculation.")
            loss_args = [model_output, targets]
        elif self._model_type == self.ModelType.SCORE_BASED:
            if self._sde is None:
                raise RuntimeError("Score-based model requires an SDE to be set for loss calculation.")
            # Assumed signature: loss(data, sde, model, device, **kwargs)
            # Pass the unwrapped model for score-based loss functions
            loss_args = [inputs, self._sde, self.module, self._device]
        else:
            raise ValueError(f"Unsupported ModelType for loss: {self._model_type}")

        # Add epoch if required and loss function signature allows it
        if self._pass_epoch_to_loss:
            try:
                # Attempt to bind 'epoch' as a keyword argument
                inspect.signature(self._loss_fn).bind_partial(**{'epoch': current_epoch})
                # If bind doesn't raise TypeError, add 'epoch' to kwargs
                loss_kwargs['epoch'] = current_epoch
            except TypeError:
                # Signature doesn't accept 'epoch' kwarg, issue warning on rank 0 once
                if self.__logger and current_epoch == 0:  # Log only once per training run
                    fn_name = getattr(self._loss_fn, '__name__', repr(self._loss_fn))
                    self.__logger.warning(f"Loss function {fn_name} requested epoch passing (pass_epoch_to_loss=True), "
                                          "but it does not accept 'epoch' as a keyword argument. Epoch will not be passed.")
            except Exception as e:  # Catch other potential signature errors
                if self.__logger and current_epoch == 0:
                    fn_name = getattr(self._loss_fn, '__name__', repr(self._loss_fn))
                    self.__logger.error(f"Error inspecting signature of loss function {fn_name}: {e}")

        # Call the loss function
        loss = self._loss_fn(*loss_args, **loss_kwargs)
        return loss

    def _calculate_metrics(self, model_output: Any, targets: Optional[Any]) -> Dict[str, float]:
        """Calculates all defined metrics on the local batch."""
        batch_metrics = {}
        if not self._metrics:
            return batch_metrics
        # Cannot calculate metrics requiring targets if targets are None (e.g., score-based validation)
        if targets is None and self._model_type in [self.ModelType.CLASSIFICATION, self.ModelType.REGRESSION]:
            return batch_metrics

        with torch.no_grad():  # Ensure metrics don't track gradients
            for name, metric_fn in self._metrics.items():
                try:
                    # Assume metric_fn takes (output, target)
                    value = metric_fn(model_output, targets)
                    # Ensure result is a float number
                    if isinstance(value, Tensor):
                        batch_metrics[name] = value.item()
                    else:
                        batch_metrics[name] = float(value)
                except Exception as e:
                    # Log error on rank 0
                    if self.__logger:
                        self.__logger.error(f"Rank {self._rank}: Error calculating metric '{name}': {e}",
                                            exc_info=False)  # Avoid too much logging
                    batch_metrics[name] = math.nan  # Indicate error
        return batch_metrics

    def _train_step(self, batch: Any, current_epoch: int, accumulation_steps: int, use_amp: bool) -> tuple[
        float, Dict[str, float]]:
        """Performs a single training step on the local batch (forward, loss, backward).
           Returns the local loss item (un-normalized) and local metrics dict.
        """
        if self._model is None or self._optimizer is None or self._loss_fn is None:
            raise RuntimeError("Model, optimizer, and loss function must be set for training.")

        self._model.train()  # Ensure model is in training mode

        # Prepare batch data for the current device
        batch_data = self._prepare_batch(batch)
        inputs = batch_data["inputs"]
        targets = batch_data["targets"]
        additional_params = {k: v for k, v in batch_data.items() if k not in ["inputs", "targets"]}

        # Mixed Precision Context
        # autocast needs device type ('cuda' or 'cpu')
        with autocast(device_type=self._device.type, enabled=use_amp):
            # Forward pass - use self.model to handle DDP/DP wrapping automatically
            if self._model_type == self.ModelType.SCORE_BASED:
                # Score-based loss often calls the model internally
                model_output = None  # Placeholder
            else:
                model_output = self.model(inputs, **additional_params)

            # Loss calculation
            loss_val = self._calculate_loss(model_output, targets, inputs, current_epoch)

            # Handle tuple loss (optional secondary losses) - grab primary loss
            # We don't aggregate secondary losses across ranks currently
            if isinstance(loss_val, tuple):
                loss = loss_val[0]
                # Store or log secondary losses locally if needed? For now, just use primary.
            else:
                loss = loss_val

            # Check for NaNs/Infs in the primary loss
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                # Log warning on rank 0
                if self._rank == 0:
                    warnings.warn(
                        f"NaN or Inf detected in training loss on Rank {self._rank} (Batch). Skipping backward pass for this batch.",
                        RuntimeWarning)
                # Return NaN loss and empty metrics to indicate skip
                return math.nan, {}

            # Normalize loss for gradient accumulation *before* scaling/backward
            loss = loss / accumulation_steps

        # Backward pass & Gradient Scaling (if AMP)
        # DDP automatically synchronizes gradients during backward
        self._grad_scaler.scale(loss).backward()

        # Metrics calculation (using local model_output and targets)
        # Skip metrics if score-based (or if model_output is None)
        batch_metrics = {}
        if self._model_type != self.ModelType.SCORE_BASED and model_output is not None:
            batch_metrics = self._calculate_metrics(model_output, targets)

        # Return the un-normalized loss item for accumulation and local metrics
        # Multiply by accumulation_steps to get the effective loss for this batch before normalization
        return loss.item() * accumulation_steps, batch_metrics

    def _val_step(self, batch: Any, current_epoch: int) -> tuple[float, Dict[str, float]]:
        """Performs a single validation step on the local batch.
           Returns the local loss item and local metrics dict.
        """
        if self._model is None or self._loss_fn is None:
            raise RuntimeError("Model and loss function must be set for validation.")

        self.eval(activate=True, log=False)  # Ensure model is in evaluation mode

        batch_data = self._prepare_batch(batch)
        inputs = batch_data["inputs"]
        targets = batch_data["targets"]
        additional_params = {k: v for k, v in batch_data.items() if k not in ["inputs", "targets"]}

        with torch.no_grad():  # No gradients needed for validation
            # Forward pass (similar to train step)
            if self._model_type == self.ModelType.SCORE_BASED:
                model_output = None  # Loss handles model call
            else:
                # Use self.model for forward pass (handles DDP/DP wrapper)
                model_output = self.model(inputs, **additional_params)

            # Loss calculation
            loss_val = self._calculate_loss(model_output, targets, inputs, current_epoch)
            # Handle tuple loss
            if isinstance(loss_val, tuple):
                loss = loss_val[0]
            else:
                loss = loss_val

            # Metrics calculation
            batch_metrics = {}
            if self._model_type != self.ModelType.SCORE_BASED and model_output is not None:
                batch_metrics = self._calculate_metrics(model_output, targets)

        # Return loss item and metrics dict
        # Handle potential NaN/Inf in validation loss
        loss_item = loss.item() if not (torch.isnan(loss).any() or torch.isinf(loss).any()) else math.nan
        return loss_item, batch_metrics

    # --- Main Training Method (Handles DDP Aggregation) ---
    def train(self,
              epochs: int,
              validate_every: int = 1,
              gradient_accumulation_steps: int = 1,
              use_amp: bool = False,
              gradient_clipping_norm: Optional[float] = None,
              ema_decay: float = 0.0,
              seed: Optional[int] = None,
              progress_bar: bool = True,
              debug_print_interval: Optional[int] = None,  # Print local batch logs every N steps (rank 0)
              save_on_last_epoch: bool = True,
              epoch_train_and_val_pbar: bool = False):  # Inner progress bars (rank 0)
        """
        Starts the training process, handling DDP aggregation, logging, and checkpointing.

        Args:
            epochs (int): Total number of epochs to train for.
            validate_every (int): Run validation every N epochs (uses aggregated metrics).
                                  Set to 0 or None to disable validation.
            gradient_accumulation_steps (int): Accumulate gradients over N steps.
            use_amp (bool): Enable Automatic Mixed Precision (requires CUDA and torch.amp).
            gradient_clipping_norm (Optional[float]): Max norm for gradient clipping.
            ema_decay (float): Decay factor for Exponential Moving Average (requires torch_ema). 0 disables EMA.
            seed (Optional[int]): Seed for this training run (applied to all ranks). Overrides handler seed.
            progress_bar (bool): Display tqdm progress bar for epochs (rank 0 only).
            debug_print_interval (Optional[int]): Print local batch info every N steps (rank 0 only).
            save_on_last_epoch (bool): Ensure AutoSaver saves on the final epoch if enabled.
            epoch_train_and_val_pbar (bool): Display inner tqdm bars for train/val batches (rank 0 only).
        """
        # --- Pre-Training Checks (Rank 0 performs checks, others wait) ---
        if self._rank == 0:
            if self._model is None or self._optimizer is None or self._train_loader is None or self._loss_fn is None:
                message = "Model, optimizer, training loader, and loss function must be set before training."
                if self.__logger: self.__logger.error(message)
                raise RuntimeError(message)
            if (validate_every is not None and validate_every > 0) and self._val_loader is None:
                message = "Validation requested (validate_every > 0), but validation loader is not set."
                if self.__logger: self.__logger.error(message)
                raise ValueError(message)  # Use ValueError for config issues
            if use_amp and not _amp_available:
                warnings.warn("AMP requested but torch.amp not available. Disabling AMP.", RuntimeWarning)
            if use_amp and self._device.type != 'cuda':
                warnings.warn("AMP requested but device is not CUDA. Disabling AMP.", RuntimeWarning)
            if ema_decay > 0 and not _ema_available:
                warnings.warn("EMA requested but torch_ema not available. Disabling EMA.", RuntimeWarning)

        # Synchronize after checks to ensure all ranks proceed or fail together
        if self._distributed:
            dist.barrier()

        # Determine effective settings after warnings
        effective_use_amp = use_amp and _amp_available and self._device.type == 'cuda'
        effective_ema_decay = ema_decay if _ema_available and ema_decay > 0 else 0.0

        # Apply seed if provided (applies on all ranks via setter)
        if seed is not None:
            self.seed = seed  # Use setter

        start_epoch = len(self._train_losses)  # Resume from last epoch (based on rank 0 history)
        total_epochs = start_epoch + epochs

        # --- Setup (Logging, EMA, AMP Scaler) ---
        if self.__logger:  # Log setup details on rank 0
            self.__logger.info(f"--- Starting Training Run ---")
            self.__logger.info(f"  Epochs:              {start_epoch + 1} -> {total_epochs}")
            self.__logger.info(f"  Distributed:         {self._distributed} (World Size: {self._world_size})")
            self.__logger.info(f"  AMP Enabled:         {effective_use_amp}")
            self.__logger.info(f"  EMA Enabled:         {effective_ema_decay > 0} (Decay: {effective_ema_decay:.4f})")
            self.__logger.info(f"  Grad Accumulation:   {gradient_accumulation_steps}")
            self.__logger.info(f"  Grad Clipping Norm:  {gradient_clipping_norm}")
            self.__logger.info(f"  Validate Every:      {validate_every}")
            self.__logger.info(f"  Seed:                {self._seed}")

        # Initialize EMA if needed (on all ranks, state loaded later if applicable)
        self._ema = None
        self._ema_decay = effective_ema_decay
        if self._ema_decay > 0:
            try:
                # Pass the parameters of the potentially wrapped model
                self._ema = ExponentialMovingAverage(self.model.parameters(), decay=self._ema_decay)
                # Load EMA state if resuming (handled in load method)
                if self.__logger:  # Log on rank 0
                    self.__logger.info(f"Initialized Exponential Moving Average with decay {self._ema_decay}.")
            except Exception as e:
                if self.__logger:  # Log error on rank 0
                    self.__logger.error(f"Failed to initialize EMA: {e}. Disabling EMA.")
                self._ema = None
                self._ema_decay = 0.0

        # Initialize GradScaler for AMP (on all ranks, state loaded later)
        self._grad_scaler = GradScaler(enabled=effective_use_amp)
        if effective_use_amp and self.__logger:  # Log on rank 0
            self.__logger.info("Automatic Mixed Precision (AMP) GradScaler enabled.")

        # Progress bar setup (only on rank 0)
        # Outer loop pbar
        pbar_outer = None
        if progress_bar and self._rank == 0:
            if _tqdm_available:
                pbar_outer = tqdm(range(start_epoch, total_epochs), desc="Epochs", unit="epoch", dynamic_ncols=True)
            else:  # Log warning if tqdm not available
                if self.__logger:
                    self.__logger.warning("tqdm not found. Progress bar disabled. Install with 'pip install tqdm'")

        # --- Callback: on_train_begin ---
        self._stop_training = False  # Reset stop flag
        train_begin_logs = {'start_epoch': start_epoch, 'total_epochs': total_epochs, 'world_size': self._world_size}
        # Run on all ranks, callbacks should handle rank internally if needed
        self._run_callbacks('on_train_begin', logs=train_begin_logs)

        # =========================== Main Training Loop ===========================
        # Initialize variable for final epoch logs outside the loop
        final_epoch_logs_agg = {}

        for epoch in range(start_epoch, total_epochs):
            epoch_start_time = time.time()
            current_epoch_1_based = epoch + 1
            # Logs for this epoch, start empty, populated by train/val/callbacks
            # Aggregated logs will be stored on rank 0
            epoch_logs = {'epoch': current_epoch_1_based}

            # --- Set Sampler Epoch (Important for DDP reproducibility) ---
            if self._distributed:
                if self._train_sampler:
                    self._train_sampler.set_epoch(epoch)
                if self._val_sampler:
                    # Set epoch for val sampler too, important if shuffle=True for val
                    self._val_sampler.set_epoch(epoch)

            # --- Callback: on_epoch_begin ---
            # Run on all ranks
            self._run_callbacks('on_epoch_begin', epoch=epoch, logs=epoch_logs)

            # ================== Training Phase ==================
            self.eval(activate=False, log=False)  # Set model to train mode
            # Accumulators for local results on this rank
            train_loss_accum_local = 0.0
            train_metrics_accum_local = defaultdict(float)
            train_batches_processed_local = 0  # Count successful batches
            batches_in_epoch = 0  # Determine batches per rank
            if self._train_loader:
                try:
                    batches_in_epoch = len(self._train_loader)
                except TypeError:
                    # Handle iterable datasets with no __len__
                    batches_in_epoch = -1  # Indicate unknown length
                    if self.__logger and epoch == start_epoch:  # Log once
                        self.__logger.warning("Train DataLoader has no length. Inner progress bar may be inaccurate.")

            # Setup inner progress bar for training (rank 0 only)
            train_iterator = enumerate(self._train_loader) if self._train_loader else []
            pbar_inner_train = None
            if epoch_train_and_val_pbar and self._rank == 0 and _tqdm_available and batches_in_epoch > 0:
                pbar_inner_train = tqdm(total=batches_in_epoch, desc=f"E{current_epoch_1_based} Train", leave=False,
                                        unit="batch", dynamic_ncols=True)

                # Wrap iterator with pbar update
                def _train_pbar_update_iterator(iterator, pbar):
                    for idx, data in iterator:
                        yield idx, data
                        pbar.update(1)

                train_iterator = _train_pbar_update_iterator(train_iterator, pbar_inner_train)

            # --- Training Batch Loop ---
            for batch_idx, batch_data in train_iterator:
                # Determine batch size for logging if possible
                batch_size = -1
                try:
                    if isinstance(batch_data, (list, tuple)) and batch_data and isinstance(batch_data[0], Tensor):
                        batch_size = batch_data[0].size(0)
                    elif isinstance(batch_data, Tensor):
                        batch_size = batch_data.size(0)
                except:
                    pass  # Ignore errors getting batch size

                batch_logs = {'batch': batch_idx, 'size': batch_size}
                # --- Callback: on_train_batch_begin --- (All ranks)
                self._run_callbacks('on_train_batch_begin', batch=batch_idx, logs=batch_logs)

                # --- Train Step (Forward, Loss, Backward) ---
                local_loss_item, local_metrics_items = self._train_step(batch_data, epoch,
                                                                        gradient_accumulation_steps, effective_use_amp)

                # --- Accumulate Local Results ---
                if not math.isnan(local_loss_item):
                    train_loss_accum_local += local_loss_item
                    train_batches_processed_local += 1
                    for name, value in local_metrics_items.items():
                        # Accumulate metrics, skip NaNs from metric calculation
                        if not math.isnan(value):
                            train_metrics_accum_local[name] += value
                        # Consider counting non-NaNs per metric for accurate averaging later?
                        # For simplicity, average over processed batches count for now.

                # --- Optimizer Step ---
                is_accumulation_step = (batch_idx + 1) % gradient_accumulation_steps == 0
                # Handle last batch correctly if epoch length not divisible by accum steps
                is_last_batch = (batch_idx + 1) == batches_in_epoch if batches_in_epoch > 0 else False
                # If length is unknown, step on every accumulation interval
                # TODO: How to handle last batch if length is unknown? Need a signal?

                if (is_accumulation_step or is_last_batch) and not math.isnan(local_loss_item):
                    # Gradient Clipping (optional) - Unscale first if using AMP
                    if gradient_clipping_norm is not None:
                        if effective_use_amp:
                            self._grad_scaler.unscale_(self._optimizer)  # Unscale inplace before clipping
                        # Clip gradients of the model parameters
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=gradient_clipping_norm)

                    # Optimizer Step (using scaler if AMP enabled)
                    self._grad_scaler.step(self._optimizer)
                    # Update GradScaler's scale for next iteration
                    self._grad_scaler.update()
                    # Zero gradients *after* stepping scaler and optimizer
                    self._optimizer.zero_grad()

                    # EMA Update (after optimizer step)
                    if self._ema:
                        self._ema.update()

                # --- Callbacks & Debug Logging ---
                # Add local results to batch logs for callbacks
                batch_logs['loss'] = local_loss_item  # Local loss for this batch
                batch_logs.update(local_metrics_items)  # Local metrics for this batch
                # --- Callback: on_train_batch_end --- (All ranks)
                self._run_callbacks('on_train_batch_end', batch=batch_idx, logs=batch_logs)

                # Debug print local info (rank 0 only)
                if self._rank == 0 and debug_print_interval and (batch_idx + 1) % debug_print_interval == 0:
                    debug_str = f"[R{self._rank} E{current_epoch_1_based} B{batch_idx + 1}] Local L:{local_loss_item:.3e}"
                    debug_str += " Metrics: " + " ".join([f"{k}:{v:.3f}" for k, v in local_metrics_items.items()])
                    if self.__logger:
                        self.__logger.debug(debug_str)
                    # Update inner pbar postfix if exists
                    if pbar_inner_train:
                        pbar_inner_train.set_postfix_str(f"Local: {debug_str}", refresh=False)

            # Close inner training progress bar if used
            if pbar_inner_train:
                pbar_inner_train.close()

            # --- Aggregate Training Results Across Ranks ---
            # Ensure all ranks finish local training loop
            if self._distributed:
                dist.barrier()

            # Calculate average local loss and metrics
            avg_train_loss_local = train_loss_accum_local / train_batches_processed_local if train_batches_processed_local > 0 else math.nan
            avg_train_metrics_local = {
                name: total / train_batches_processed_local
                for name, total in train_metrics_accum_local.items()
            } if train_batches_processed_local > 0 else {name: math.nan for name in self._metrics}

            # Aggregate averages across ranks
            avg_train_loss_agg = _aggregate_loss(avg_train_loss_local, self._world_size, self._device)
            avg_train_metrics_agg = _aggregate_metrics(avg_train_metrics_local, self._world_size, self._device)

            # Store aggregated results on Rank 0
            if self._rank == 0:
                self._train_losses.append(avg_train_loss_agg)
                epoch_logs['loss'] = avg_train_loss_agg  # Add aggregated loss to epoch logs
                for name, value in avg_train_metrics_agg.items():
                    self._train_metrics_history[name].append(value)
                    epoch_logs[name] = value  # Add aggregated train metrics
                # Log aggregated train results
                if self.__logger:
                    train_log_msg = f"Epoch {current_epoch_1_based} Train Aggregated: Loss={avg_train_loss_agg:.4e}"
                    train_log_msg += " Metrics: " + ", ".join(
                        [f"{k}={v:.4f}" for k, v in avg_train_metrics_agg.items()])
                    self.__logger.debug(train_log_msg)

            # ================= Validation Phase ==================
            run_validation = (
                    validate_every is not None and validate_every > 0 and current_epoch_1_based % validate_every == 0)
            avg_val_loss_agg = math.nan  # Initialize aggregated results
            avg_val_metrics_agg = {name: math.nan for name in self._metrics}

            if run_validation:
                # --- Callback: on_val_begin --- (All ranks)
                self._run_callbacks('on_val_begin', logs=epoch_logs)  # Pass current epoch logs

                # Accumulators for local validation results
                val_loss_accum_local = 0.0
                val_metrics_accum_local = defaultdict(float)
                val_batches_processed_local = 0
                val_batches_in_epoch = 0
                if self._val_loader:
                    try:
                        val_batches_in_epoch = len(self._val_loader)
                    except TypeError:
                        val_batches_in_epoch = -1  # Unknown length
                        if self.__logger and epoch == start_epoch:
                            self.__logger.warning(
                                "Validation DataLoader has no length. Inner progress bar may be inaccurate.")

                # Setup inner progress bar for validation (rank 0 only)
                val_iterator = enumerate(self._val_loader) if self._val_loader else []
                pbar_inner_val = None
                if epoch_train_and_val_pbar and self._rank == 0 and _tqdm_available and val_batches_in_epoch > 0:
                    pbar_inner_val = tqdm(total=val_batches_in_epoch, desc=f"E{current_epoch_1_based} Val", leave=False,
                                          unit="batch", dynamic_ncols=True)

                    # Wrap iterator with pbar update
                    def _val_pbar_update_iterator(iterator, pbar):
                        for idx, data in iterator:
                            yield idx, data
                            pbar.update(1)

                    val_iterator = _val_pbar_update_iterator(val_iterator, pbar_inner_val)

                # Apply EMA weights for validation if enabled
                # Use context manager to automatically restore weights after
                ema_context = self._ema.average_parameters() if self._ema else contextlib.nullcontext()
                with ema_context:
                    self.eval(activate=True, log=False)  # Ensure eval mode inside EMA context if needed
                    for val_batch_idx, val_batch_data in val_iterator:
                        # Determine batch size for logging
                        batch_size = -1
                        try:
                            if isinstance(val_batch_data, (list, tuple)) and val_batch_data and isinstance(
                                    val_batch_data[0], Tensor):
                                batch_size = val_batch_data[0].size(0)
                            elif isinstance(val_batch_data, Tensor):
                                batch_size = val_batch_data.size(0)
                        except:
                            pass

                        batch_logs = {'batch': val_batch_idx, 'size': batch_size}
                        # --- Callback: on_val_batch_begin --- (All ranks)
                        self._run_callbacks('on_val_batch_begin', batch=val_batch_idx, logs=batch_logs)

                        # --- Val Step ---
                        local_loss_item, local_metrics_items = self._val_step(val_batch_data, epoch)

                        # --- Accumulate Local Validation Results ---
                        if not math.isnan(local_loss_item):
                            val_loss_accum_local += local_loss_item
                            val_batches_processed_local += 1
                            for name, value in local_metrics_items.items():
                                if not math.isnan(value):
                                    val_metrics_accum_local[name] += value

                        # --- Callbacks ---
                        batch_logs['val_loss'] = local_loss_item  # Local val loss
                        # Add local val metrics with prefix
                        batch_logs.update({f'val_{k}': v for k, v in local_metrics_items.items()})
                        # --- Callback: on_val_batch_end --- (All ranks)
                        self._run_callbacks('on_val_batch_end', batch=val_batch_idx, logs=batch_logs)

                        # Update inner pbar postfix if exists (rank 0)
                        if pbar_inner_val:
                            debug_str = f"Local L:{local_loss_item:.3e} "
                            debug_str += " ".join([f"{k}:{v:.3f}" for k, v in local_metrics_items.items()])
                            pbar_inner_val.set_postfix_str(debug_str, refresh=False)

                # Close inner validation progress bar if used
                if pbar_inner_val:
                    pbar_inner_val.close()

                # --- Aggregate Validation Results Across Ranks ---
                if self._distributed:
                    dist.barrier()

                # Calculate average local validation results
                avg_val_loss_local = val_loss_accum_local / val_batches_processed_local if val_batches_processed_local > 0 else math.nan
                avg_val_metrics_local = {
                    name: total / val_batches_processed_local
                    for name, total in val_metrics_accum_local.items()
                } if val_batches_processed_local > 0 else {name: math.nan for name in self._metrics}

                # Aggregate averages across ranks
                avg_val_loss_agg = _aggregate_loss(avg_val_loss_local, self._world_size, self._device)
                avg_val_metrics_agg = _aggregate_metrics(avg_val_metrics_local, self._world_size, self._device)

                # Store aggregated results on Rank 0
                if self._rank == 0:
                    self._val_losses.append(avg_val_loss_agg)
                    epoch_logs['val_loss'] = avg_val_loss_agg  # Add aggregated val loss
                    for name, value in avg_val_metrics_agg.items():
                        self._val_metrics_history[name].append(value)
                        epoch_logs[f'val_{name}'] = value  # Add aggregated val metrics with prefix

                    if self.__logger:
                        val_log_msg = f"Epoch {current_epoch_1_based} Val Aggregated: Loss={avg_val_loss_agg:.4e}"
                        val_log_msg += " Metrics: " + ", ".join(
                            [f"{k}={v:.4f}" for k, v in avg_val_metrics_agg.items()])
                        self.__logger.debug(val_log_msg)

                # --- Callback: on_val_end --- (All ranks, logs contain aggregated results if rank 0)
                # Need to broadcast logs from rank 0 if callbacks on other ranks need aggregated results
                logs_list_for_broadcast = [epoch_logs if self._rank == 0 else None]
                if self._distributed:
                    dist.broadcast_object_list(logs_list_for_broadcast, src=0)
                received_val_end_logs = logs_list_for_broadcast[0]
                self._run_callbacks('on_val_end', logs=received_val_end_logs)

            elif self._rank == 0:  # No validation run, append NaN on rank 0 for consistent history length
                self._val_losses.append(math.nan)
                epoch_logs['val_loss'] = math.nan
                for name in self._metrics.keys():
                    self._val_metrics_history[name].append(math.nan)
                    epoch_logs[f'val_{name}'] = math.nan

            # --- Scheduler Step ---
            # Step scheduler on all ranks based on aggregated metric if needed
            if self._scheduler:
                if isinstance(self._scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # Requires aggregated validation metric (use avg_val_loss_agg)
                    if run_validation:  # Only step if validation ran
                        # Ensure the aggregated loss is valid before stepping
                        if not math.isnan(avg_val_loss_agg):
                            self._scheduler.step(avg_val_loss_agg)
                            # Log LR change possibility on rank 0
                            if self.__logger and epoch == start_epoch:  # Log once
                                self.__logger.debug(
                                    f"Stepped ReduceLROnPlateau scheduler with aggregated val_loss: {avg_val_loss_agg:.4e}")
                        elif self._rank == 0:  # Warn on rank 0 if metric is NaN
                            warnings.warn(
                                f"Epoch {current_epoch_1_based}: ReduceLROnPlateau requires a valid aggregated validation metric (e.g., val_loss) "
                                "to step, but received NaN. Scheduler not stepped.", RuntimeWarning)
                    # else: Do nothing if validation didn't run
                else:
                    # Standard schedulers usually step every epoch
                    self._scheduler.step()
                    # Log scheduler step debug on rank 0 (once)
                    if self.__logger and epoch == start_epoch:
                        self.__logger.debug(f"Stepped scheduler {type(self._scheduler).__name__}.")

            # --- Log LR & Epoch Summary (Rank 0) ---
            epoch_time = time.time() - epoch_start_time
            # Store epoch time on rank 0 logs before broadcasting
            if self._rank == 0:
                epoch_logs['epoch_time'] = epoch_time
                # Add current learning rate(s) to logs (rank 0)
                if self._optimizer:
                    for i, pg in enumerate(self._optimizer.param_groups):
                        lr_key = f'lr_group_{i}'
                        epoch_logs[lr_key] = pg['lr']
                        if i == 0:
                            epoch_logs['lr'] = pg['lr']  # Common key for first group LR

                # Format log message using aggregated results from epoch_logs
                log_msg = f"E{current_epoch_1_based}/{total_epochs} [{epoch_time:.2f}s]"
                log_msg += f" Train Loss: {epoch_logs.get('loss', math.nan):.4e}"
                train_metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in avg_train_metrics_agg.items()])
                if train_metrics_str: log_msg += f" Train Metrics: [{train_metrics_str}]"

                if run_validation:
                    log_msg += f" | Val Loss: {epoch_logs.get('val_loss', math.nan):.4e}"
                    val_metrics_str = ", ".join([f"val_{k}={v:.4f}" for k, v in avg_val_metrics_agg.items()])
                    if val_metrics_str: log_msg += f" Val Metrics: [{val_metrics_str}]"

                log_msg += f" | LR: {epoch_logs.get('lr', math.nan):.2e}"

                if self.__logger:
                    self.__logger.info(log_msg)
                # Update outer progress bar postfix (rank 0)
                if pbar_outer:
                    # Extract summary part for postfix
                    summary_postfix = log_msg[log_msg.find("Train Loss:"):]
                    pbar_outer.set_postfix_str(summary_postfix)

            # --- Callback: on_epoch_end ---
            # Broadcast logs from Rank 0 to all other ranks so callbacks have consistent info
            logs_list_broadcast = [epoch_logs if self._rank == 0 else None]
            if self._distributed:
                dist.broadcast_object_list(logs_list_broadcast, src=0)
            # All ranks receive the aggregated logs from rank 0
            final_epoch_logs_agg = logs_list_broadcast[0]

            # Run on_epoch_end on all ranks with the *same* aggregated logs
            self._run_callbacks('on_epoch_end', epoch=epoch, logs=final_epoch_logs_agg)

            # --- Auto Saving (Rank 0 Only) ---
            # Pass aggregated logs for filename formatting
            if self._rank == 0:
                self._auto_save_epoch(epoch, total_epochs, save_on_last_epoch, final_epoch_logs_agg)

            # --- Check for Early Stopping Signal ---
            # The flag _stop_training might be set by a callback (e.g., EarlyStopping on rank 0)
            # We need to broadcast this decision from rank 0 to all ranks
            stop_tensor = torch.tensor(int(self._stop_training), device=self._device, dtype=torch.int)
            if self._distributed:
                dist.broadcast(stop_tensor, src=0)  # Broadcast the decision from rank 0
            self._stop_training = bool(stop_tensor.item())  # Update flag on all ranks

            if self._stop_training:
                if self.__logger:  # Log on rank 0
                    self.__logger.info(f"Early stopping triggered after epoch {current_epoch_1_based}.")
                # Ensure all ranks know about stopping before breaking
                if self._distributed:
                    dist.barrier()
                break  # Exit the main training loop

            # Update outer progress bar (rank 0)
            if pbar_outer:
                pbar_outer.update(1)

            # Final barrier at end of epoch loop for synchronization
            if self._distributed:
                dist.barrier()

        # =========================== End of Training ===========================
        if pbar_outer:
            pbar_outer.close()

        # Apply final EMA weights if used (on all ranks)
        if self._ema:
            if self.__logger:  # Log on rank 0
                self.__logger.info("Applying final EMA weights to the model.")
            try:
                # Ensure EMA update happens on the parameters of the wrapped model
                self._ema.copy_to(self.model.parameters())
            except Exception as e:
                if self.__logger:  # Log error on rank 0
                    self.__logger.error(f"Failed to apply final EMA weights: {e}")

        # --- Callback: on_train_end ---
        # Pass the final aggregated logs from the last completed epoch
        final_logs = {'final_epoch': epoch + 1, 'world_size': self._world_size}
        # final_epoch_logs_agg should contain the aggregated logs from the last epoch
        final_logs.update(final_epoch_logs_agg if final_epoch_logs_agg else {})
        # Run on all ranks with consistent final logs
        self._run_callbacks('on_train_end', logs=final_logs)

        if self.__logger:  # Log on rank 0
            self.__logger.info("--- Training Run Finished ---")

        # Final barrier to ensure all processes finish cleanly
        if self._distributed:
            dist.barrier()

    def _auto_save_epoch(self, epoch: int, total_epochs: int, save_on_last_epoch: bool, logs: Dict[str, Any]):
        """Handles the logic for auto-saving the model state (only executed on rank 0)."""
        # Explicit rank check, though this method is called within rank 0 block in train()
        if self._auto_saver.save_interval is None or self._auto_saver.save_path is None:
            return  # Auto-save disabled

        current_epoch_1_based = epoch + 1
        should_save = False

        # Interval saving
        if self._auto_saver.save_interval > 0 and (current_epoch_1_based % self._auto_saver.save_interval == 0):
            should_save = True

        # Save on last epoch requested
        is_last_epoch = (current_epoch_1_based == total_epochs)
        if save_on_last_epoch and is_last_epoch:
            should_save = True

        # Save if interval is -1 (only last epoch)
        if self._auto_saver.save_interval == -1 and is_last_epoch:
            should_save = True

        if should_save:
            # Use logs dict for formatting filename keys (e.g., val_loss, val_accuracy)
            format_dict = logs.copy()
            format_dict['epoch'] = current_epoch_1_based  # Ensure epoch is available
            try:
                # Add .pth extension if not included in the format name
                base_filename = self._auto_saver.save_model_name.format(**format_dict)
                filename = base_filename if base_filename.endswith(".pth") else f"{base_filename}.pth"
            except KeyError as e:
                # Fallback if format key is missing in logs (e.g., no validation ran)
                filename = f"{self._auto_saver.save_model_name}_epoch{current_epoch_1_based}.pth"
                if self.__logger:
                    self.__logger.warning(
                        f"Auto-save filename formatting failed (KeyError: {e}). Using fallback: {filename}")
            except Exception as e:
                filename = f"{self._auto_saver.save_model_name}_epoch{current_epoch_1_based}.pth"
                if self.__logger:
                    self.__logger.error(
                        f"Auto-save filename formatting failed unexpectedly ({type(e).__name__}: {e}). Using fallback: {filename}")

            save_path = os.path.join(self._auto_saver.save_path, filename)

            if self.__logger:
                self.__logger.info(f"Auto-saving handler state to: {save_path}")

            # Perform save using the main save method
            self.save(save_path)  # save() is rank 0 aware

            if self._rank != 0:
                return

            # Handle overwriting previous auto-save file
            # Check if overwrite enabled, if a previous file exists, and if it's different from the current save
            if self._auto_saver.overwrite_last_saved and \
                    self._auto_saver.last_saved_model and \
                    self._auto_saver.last_saved_model != save_path:
                try:
                    os.remove(self._auto_saver.last_saved_model)
                    if self.__logger:
                        self.__logger.debug(f"Removed previous auto-saved state: {self._auto_saver.last_saved_model}")
                except OSError as e:
                    # Warn if removal fails but continue
                    warn_msg = f"Could not remove previous auto-saved state '{self._auto_saver.last_saved_model}': {e}"
                    warnings.warn(warn_msg, RuntimeWarning)
                    if self.__logger:
                        self.__logger.warning(warn_msg)

            # Update the last saved model path
            self._auto_saver.last_saved_model = save_path

    def monitor_keys(self) -> List[str]:
        """
        Returns a list of key metric names typically available in the `logs` dictionary
        passed to callbacks during training (based on aggregated results).

        This includes standard losses, time, learning rate, and any custom metrics added.
        """
        keys = ['loss', 'val_loss', 'epoch_time', 'lr']
        # Add keys for all defined custom metrics
        keys.extend(self._metrics.keys())
        # Add keys for validation versions of custom metrics (prefixed)
        keys.extend([f'val_{k}' for k in self._metrics.keys()])
        # Add keys for multiple learning rate groups if applicable
        # (Callbacks need to handle potential missing keys if optimizer changes)
        # keys.extend([f'lr_group_{i}' for i in range(num_param_groups)]) # Need num_param_groups at runtime
        return list(set(keys))  # Return unique keys

    def save(self, path: str):
        """Saves the complete handler state to a file (executed by rank 0 only).
           Other ranks wait at a barrier. State includes model, optimizer, scheduler,
           history (rank 0), EMA, scaler, config, etc.
        """
        # --- Rank 0 performs the save ---
        if self._rank == 0:
            if self._model is None:
                warnings.warn("Attempting to save handler state, but model is missing. Skipping save.", RuntimeWarning)
                # Ensure barrier is still hit even if save is skipped
                return

            # Ensure directory exists
            try:
                save_dir = os.path.dirname(path)
                if save_dir:  # Only create if path includes a directory
                    os.makedirs(save_dir, exist_ok=True)
            except OSError as e:
                if self.__logger:
                    self.__logger.error(f"Could not create directory for saving state to {path}: {e}")
                raise e

            # --- Prepare State Dictionary ---
            state = {
                # Core Components State Dicts
                "model_state_dict": self.module.state_dict(),  # *** Save unwrapped model state ***
                "optimizer_state_dict": self._optimizer.state_dict() if self._optimizer else None,
                "scheduler_state_dict": self._scheduler.state_dict() if self._scheduler else None,
                "grad_scaler_state_dict": self._grad_scaler.state_dict() if _amp_available and self._grad_scaler else None,

                # Configuration (Classes and Kwargs)
                "model_class": self._model_class,  # Needs to be unpicklable or handled carefully during load
                "model_kwargs": self._model_kwargs,
                "model_type": self._model_type.value,  # Save enum value as string
                "optimizer_class": self._optimizer.__class__ if self._optimizer else None,
                "optimizer_kwargs": self._optimizer_kwargs,
                "scheduler_class": self._scheduler.__class__ if self._scheduler else None,
                "scheduler_kwargs": self._scheduler_kwargs,
                "loss_fn": self._loss_fn,  # Caution: pickling functions can be fragile
                "loss_fn_kwargs": self._loss_fn_kwargs,
                "pass_epoch_to_loss": self._pass_epoch_to_loss,
                "metrics": self._metrics,  # Caution: pickling metric functions
                "seed": self._seed,
                # Device is not saved, as it's determined at load time by rank/environment

                # Data Loader Kwargs (Dataset itself is not saved)
                "train_loader_kwargs": self._train_loader_kwargs,
                "val_loader_kwargs": self._val_loader_kwargs,

                # History (Aggregated on rank 0)
                "train_losses": self._train_losses,
                "val_losses": self._val_losses,
                "train_metrics_history": dict(self._train_metrics_history),  # Convert defaultdict
                "val_metrics_history": dict(self._val_metrics_history),

                # AutoSaver State (Rank 0 only relevant parts, but save full state)
                "auto_saver_state": self._auto_saver.state_dict(),

                # EMA State
                "ema_state_dict": self._ema.state_dict() if self._ema else None,
                "ema_decay": self._ema_decay,

                # SDE / Custom Sampler State
                "sde_class": self._sde.__class__ if self._sde else None,
                "sde_kwargs": self._sde_kwargs,

                "sampler_class": self._sampler.__class__ if self._sampler else None,
                "sampler_kwargs": self._sampler_kwargs,
                "sampler_state_dict": self._sampler.save() if self._sampler and hasattr(self._sampler,
                                                                                        'save') else None,

                # Callback States (Callbacks should implement state_dict/load_state_dict)
                # Save states from callbacks present on rank 0
                "callback_classes": {cb.__class__.__name__: cb.__class__ for cb in self._callbacks},
                "callback_states": {cb.__class__.__name__: cb.state_dict() for cb in self._callbacks},

                # Versioning / Metadata
                "nn_handler_version": __version__,  # Mark version for DDP compatibility
                "pytorch_version": torch.__version__,
            }

            # --- Perform Save ---
            try:
                torch.save(state, path)
                if self.__logger:
                    self.__logger.info(f"NNHandler state saved successfully by Rank {self._rank} to: {path}")
            except Exception as e:
                if self.__logger:
                    self.__logger.error(f"Rank {self._rank} failed to save NNHandler state to {path}: {e}",
                                        exc_info=True)
                raise e  # Re-raise the exception after logging

    @staticmethod
    def load(path: str,
             device: Optional[Union[str, torch.device]] = None,  # Target device (overrides DDP default)
             strict_load: bool = False,  # For model state_dict loading
             # Skip loading options
             skip_optimizer: bool = False,
             skip_scheduler: bool = False,
             skip_history: bool = False,
             skip_callbacks: bool = False,
             skip_sampler_sde: bool = False,
             skip_ema: bool = False
             ) -> 'NNHandler':
        """
        Loads a saved NNHandler state from a file.

        In DDP mode, all ranks load the checkpoint state, mapping tensors to their
        assigned local device. The handler is initialized within the DDP environment
        before loading the state dictionary components.

        Args:
            path (str): Path to the saved state file.
            device (Optional[Union[str, torch.device]]): Explicitly specify the device
                to load onto. If None, uses the DDP-assigned device (cuda:local_rank or cpu)
                or defaults for non-DDP mode.
            strict_load (bool): Passed to model.load_state_dict. If True, keys must match exactly.
            skip_optimizer (bool): Don't load optimizer state.
            skip_scheduler (bool): Don't load scheduler state.
            skip_history (bool): Don't load loss/metric history (rank 0).
            skip_callbacks (bool): Don't load callback states.
            skip_sampler_sde (bool): Don't load sampler/SDE state.
            skip_ema (bool): Don't load EMA state.

        Returns:
            NNHandler: An instance loaded with the saved state.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        # --- Determine Device and Map Location for Loading ---
        # Check if running in a DDP environment *currently*
        is_distributed_load = dist.is_available() and dist.is_initialized()
        current_rank = dist.get_rank() if is_distributed_load else 0

        if device is None:
            # Auto-detect map location based on current environment
            if is_distributed_load:
                local_rank = int(os.environ.get('LOCAL_RANK', -1))  # Get local rank
                if torch.cuda.is_available() and local_rank >= 0 and local_rank < torch.cuda.device_count():
                    # Map to the GPU assigned to this rank
                    map_location = f'cuda:{local_rank}'
                else:
                    # DDP on CPU or invalid local_rank
                    map_location = 'cpu'
            else:
                # Non-distributed: load to CUDA if available, else CPU
                map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            # User specified a device override
            # Resolve the user's request (e.g., "cuda", "cpu", torch.device object)
            # Note: This overrides the DDP default device, which might be unintended.
            # NNHandler's __init__ will still use the DDP device, but loading happens here.
            map_location_device = _resolve_device(device)  # Use static method context
            map_location = str(map_location_device)  # torch.load needs string or torch.device

        resolved_map_location_device = _resolve_device(map_location)

        print(
            f"INFO (Rank {current_rank}): Loading checkpoint '{os.path.basename(path)}' to map_location='{map_location}' (Resolved: {resolved_map_location_device})")

        # --- Load State Dictionary ---
        # weights_only=False is required to load optimizer, scheduler, etc.
        # If all skips are True, we could use weights_only=True, but safer to keep False.
        try:
            state = torch.load(path, map_location=map_location)  # Load onto the target device map
        except Exception as e:
            # Add rank info to error
            raise RuntimeError(
                f"Rank {current_rank}: Failed to load checkpoint from {path} using map_location '{map_location}': {e}") from e

        # --- Extract Configuration Needed for Handler Init ---
        model_class = state.get("model_class")
        model_kwargs = state.get("model_kwargs", {})
        model_type_str = state.get("model_type", NNHandler.ModelType.CLASSIFICATION.value)  # Get string value
        try:
            # Convert saved string back to enum member
            model_type = NNHandler.ModelType(model_type_str)
        except ValueError:
            # Handle case where saved value is invalid
            warnings.warn(
                f"Rank {current_rank}: Invalid model_type '{model_type_str}' found in checkpoint. Defaulting to CLASSIFICATION.",
                RuntimeWarning)
            model_type = NNHandler.ModelType.CLASSIFICATION

        if model_class is None:
            raise ValueError(
                f"Rank {current_rank}: Saved state from '{path}' is missing 'model_class'. Cannot reconstruct handler.")

        # --- Instantiate Handler ---
        # The handler will initialize DDP based on the *current* environment,
        # regardless of how the checkpoint was saved. The `device` argument here
        # tells the handler which device *this rank* should use (determined by DDP init).
        # The `map_location` used in `torch.load` ensures tensors land correctly.
        # Pass the device determined by DDP init (or user override if provided)
        handler = NNHandler(model_class=model_class,
                            device=resolved_map_location_device,  # Use the device for this rank
                            model_type=model_type,
                            # Determine DDP status based on current env, not checkpoint state
                            use_distributed=is_distributed_load,
                            **model_kwargs)
        # handler._model is now initialized and potentially DDP-wrapped

        # --- Load Model Weights ---
        model_state_dict = state.get("model_state_dict")
        if not model_state_dict:
            raise ValueError(f"Rank {current_rank}: Checkpoint '{path}' missing 'model_state_dict'.")

        # Adjust keys for DDP/DP mismatch if necessary
        # Check if the saved state dict has 'module.' prefix (saved from DDP/DP)
        saved_parallel = any(k.startswith('module.') for k in model_state_dict.keys())
        # Check if the *current* handler's model is DDP/DP wrapped
        current_parallel = isinstance(handler._model, (DDP, nn.DataParallel))

        load_state_dict = model_state_dict  # Start with original
        if saved_parallel and not current_parallel:
            # Saved with wrapper, loading into raw model -> strip 'module.'
            # This case is less common if loading into a DDP setup correctly.
            load_state_dict = OrderedDict((k[len("module."):], v) for k, v in model_state_dict.items())
            if handler.__logger: handler.__logger.debug(
                " Stripped 'module.' prefix from saved model state_dict for loading into non-parallel model.")
        elif not saved_parallel and current_parallel:
            # Saved raw, loading into wrapped model -> add 'module.'
            load_state_dict = OrderedDict(('module.' + k, v) for k, v in model_state_dict.items())
            if handler.__logger: handler.__logger.debug(
                " Added 'module.' prefix to saved model state_dict for loading into parallel model.")
        # Else: keys match (both parallel or both raw), use load_state_dict as is

        # Load into the underlying module
        try:
            missing_keys, unexpected_keys = handler.module.load_state_dict(load_state_dict, strict=strict_load)
            # Log results only on rank 0
            if handler.__logger:
                if missing_keys: handler.__logger.warning(
                    f" Missing keys when loading model state_dict: {missing_keys}")
                if unexpected_keys: handler.__logger.warning(
                    f" Unexpected keys when loading model state_dict: {unexpected_keys}")
                handler.__logger.info(" Model state_dict loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Rank {current_rank}: Failed to load model state_dict: {e}") from e

        # Model is already on the correct device due to handler init and map_location

        # --- Load Other Components (Optional) ---

        # Optimizer
        opt_class = state.get("optimizer_class")
        opt_kwargs = state.get("optimizer_kwargs", {})
        opt_state = state.get("optimizer_state_dict")
        if not skip_optimizer and opt_class and opt_state:
            try:
                # Recreate optimizer with model parameters (already wrapped if DDP)
                handler.set_optimizer(opt_class, **opt_kwargs)
                handler._optimizer.load_state_dict(opt_state)
                if handler.__logger: handler.__logger.info(" Optimizer state loaded.")
            except Exception as e:
                # Warn on all ranks, but maybe log error only on rank 0?
                warnings.warn(f"Rank {current_rank}: Failed to load optimizer state: {e}. Optimizer reset.",
                              RuntimeWarning)
                handler._optimizer = None
                handler._scheduler = None  # Also reset scheduler if optimizer failed

        # Scheduler (requires optimizer)
        sched_class = state.get("scheduler_class")
        sched_kwargs = state.get("scheduler_kwargs", {})
        sched_state = state.get("scheduler_state_dict")
        if not skip_scheduler and sched_class and sched_state and handler._optimizer:
            try:
                # Recreate scheduler attached to the loaded optimizer
                handler.set_scheduler(sched_class, **sched_kwargs)
                handler._scheduler.load_state_dict(sched_state)
                if handler.__logger: handler.__logger.info(" Scheduler state loaded.")
            except Exception as e:
                warnings.warn(f"Rank {current_rank}: Failed to load scheduler state: {e}. Scheduler reset.",
                              RuntimeWarning)
                handler._scheduler = None

        # GradScaler (AMP)
        scaler_state = state.get("grad_scaler_state_dict")
        # Check if AMP is available and scaler state exists in checkpoint
        if _amp_available and scaler_state and handler._grad_scaler:
            try:
                # Load state into the existing scaler (created in __init__)
                handler._grad_scaler.load_state_dict(scaler_state)
                # Check if loaded state implies scaler should be enabled
                # Note: dummy scaler doesn't have is_enabled, real one does.
                enabled_info = f"(Enabled: {handler._grad_scaler.is_enabled()})" if hasattr(handler._grad_scaler,
                                                                                            'is_enabled') else ""
                if handler.__logger: handler.__logger.info(f" GradScaler state loaded {enabled_info}.")
            except Exception as e:
                warnings.warn(f"Rank {current_rank}: Failed to load GradScaler state: {e}.", RuntimeWarning)

        # Loss Function & Metrics (Assume functions are available in environment)
        # Assigning function objects directly can be fragile if environment changes.
        try:
            handler._loss_fn = state.get("loss_fn")
            handler._metrics = state.get("metrics", {})  # Re-assign metrics dict
        except Exception as e:
            warnings.warn(f"Rank {current_rank}: Failed to load loss/metric functions (may need manual setting): {e}",
                          RuntimeWarning)
        handler._loss_fn_kwargs = state.get("loss_fn_kwargs", {})
        handler._pass_epoch_to_loss = state.get("pass_epoch_to_loss", False)

        # History (Load only on rank 0)
        if not skip_history:
            if handler._rank == 0:
                handler._train_losses = state.get("train_losses", [])
                handler._val_losses = state.get("val_losses", [])
                # Restore defaultdict structure if needed, default to empty dict
                handler._train_metrics_history = defaultdict(list, state.get("train_metrics_history", {}))
                handler._val_metrics_history = defaultdict(list, state.get("val_metrics_history", {}))
                if handler.__logger: handler.__logger.info(" Training history loaded (rank 0).")
            else:
                # Ensure history is empty on non-zero ranks
                handler._train_losses = []
                handler._val_losses = []
                handler._train_metrics_history = defaultdict(list)
                handler._val_metrics_history = defaultdict(list)

        # AutoSaver State (Load on all ranks for consistency, but only rank 0 uses it actively)
        auto_saver_state = state.get("auto_saver_state")
        if auto_saver_state:
            try:
                handler._auto_saver.load_state_dict(auto_saver_state)
                # Code saving only relevant for rank 0
                handler._auto_saver.save_model_code = handler._auto_saver.save_model_code and (handler._rank == 0)
                if handler.__logger: handler.__logger.info(" AutoSaver state loaded.")
            except Exception as e:
                warnings.warn(f"Rank {current_rank}: Failed to load AutoSaver state: {e}", RuntimeWarning)

        # EMA State
        handler._ema_decay = state.get("ema_decay", 0.0)
        ema_state = state.get("ema_state_dict")
        if not skip_ema and handler._ema_decay > 0 and ema_state and _ema_available:
            try:
                # Re-initialize EMA attached to the current model parameters
                handler._ema = ExponentialMovingAverage(handler.model.parameters(), decay=handler._ema_decay)
                handler._ema.load_state_dict(ema_state)
                if handler.__logger: handler.__logger.info(" EMA state loaded.")
            except Exception as e:
                warnings.warn(f"Rank {current_rank}: Failed to load EMA state: {e}. EMA reset.", RuntimeWarning)
                handler._ema = None
                handler._ema_decay = 0.0

        # SDE / Custom Sampler
        sde_class = state.get("sde_class")
        sde_kwargs = state.get("sde_kwargs", {})
        sampler_class = state.get("sampler_class")
        sampler_kwargs = state.get("sampler_kwargs", {})
        sampler_state = state.get("sampler_state_dict")
        if not skip_sampler_sde:
            if sde_class:
                try:
                    handler.set_sde(sde_class, **sde_kwargs)  # Re-initializes SDE
                    # Load SDE internal state if applicable (needs sde.load_state_dict)
                    # if hasattr(handler._sde, 'load_state_dict') and 'sde_state_dict' in state:
                    #    handler._sde.load_state_dict(state['sde_state_dict'])
                    if handler.__logger: handler.__logger.info(" SDE re-initialized from checkpoint info.")
                except Exception as e:
                    warnings.warn(f"Rank {current_rank}: Failed to load/re-init SDE: {e}", RuntimeWarning)
            if sampler_class:
                try:
                    handler.set_sampler(sampler_class, **sampler_kwargs)  # Re-initializes Sampler
                    # Load sampler state if available
                    if sampler_state and handler._sampler and hasattr(handler._sampler, 'load'):
                        handler._sampler.load(sampler_state)  # Use sampler's load method
                    if handler.__logger: handler.__logger.info(
                        " Sampler re-initialized and state loaded from checkpoint.")
                except Exception as e:
                    warnings.warn(f"Rank {current_rank}: Failed to load/re-init Sampler: {e}", RuntimeWarning)

        # Callbacks State (Load state into *existing* callbacks)
        # Assumes user adds the *same* callbacks *before* calling load.
        callback_classes = state.get("callback_classes", {})
        callback_states = state.get("callback_states", {})
        if not skip_callbacks:
            if callback_classes and callback_states:
                loaded_cb_names = set()
                for cb_name, cb_class in callback_classes.items():  # Iterate through callbacks already added to the handler
                    if cb_name in callback_states:
                        try:
                            cb = cb_class()
                            cb.load_state_dict(callback_states[cb_name])
                            loaded_cb_names.add(cb_name)
                            handler.add_callback(cb)
                            if handler.__logger: handler.__logger.info(f" Callback '{cb_name}' state loaded.")
                            # Log loaded state on rank 0 only
                            if handler.__logger:
                                handler.__logger.debug(f" Loaded state for callback '{cb_name}'.")
                        except Exception as e:
                            warnings.warn(f"Rank {current_rank}: Failed to load state for callback '{cb_name}': {e}",
                                          RuntimeWarning)

                # Warn about unmatched states (rank 0 only)
                if handler._rank == 0:
                    unmatched_states = set(callback_states.keys()) - loaded_cb_names
                    if unmatched_states:
                        warnings.warn(
                            f"Callback states found in checkpoint but not loaded (no matching callback class existed in the saved file): {unmatched_states}",
                            RuntimeWarning)
            elif callback_states and handler._rank == 0:  # Check if states exist but no callbacks added
                warnings.warn(
                    "Callback states found in checkpoint, but no callbacks class currently saved in the handler instance. States not loaded.",
                    RuntimeWarning)

        # Other config loaded on all ranks
        handler._seed = state.get("seed")  # Restore seed used for the original training run
        handler._train_loader_kwargs = state.get("train_loader_kwargs", {})  # Store for reference/re-creation
        handler._val_loader_kwargs = state.get("val_loader_kwargs", {})

        if handler.__logger:  # Rank 0 log
            handler.__logger.info(
                f"NNHandler loaded successfully by Rank {current_rank} from: {os.path.basename(path)}")

        # Barrier to ensure all ranks finish loading before returning control
        if is_distributed_load:
            if handler.__logger: handler.__logger.debug(f"Rank {current_rank} waiting at load barrier.")
            dist.barrier()
            if handler.__logger: handler.__logger.debug(f"Rank {current_rank} passed load barrier.")

        return handler

    @staticmethod
    def initialize(**kwargs):
        """DEPRECATED: Initializes NNHandler using keyword arguments. Use direct __init__ and setters."""
        warnings.warn("NNHandler.initialize is deprecated. Use direct __init__ and setters.", DeprecationWarning,
                      stacklevel=2)
        # Basic translation, may lack DDP awareness or newer features
        required = ['model_class', 'optimizer_class', 'loss_fn', 'train_data']
        missing = [k for k in required if k not in kwargs]
        if missing:
            raise ValueError(f"Missing required arguments for deprecated NNHandler.initialize: {missing}")

        mc = kwargs.pop('model_class')
        dev = kwargs.pop('device', 'cpu')
        logm = kwargs.pop('logger_mode', None)
        logf = kwargs.pop('logger_filename', 'NNHandler.log')
        logl = kwargs.pop('logger_level', logging.INFO)
        smc = kwargs.pop('save_model_code', False)
        mt = kwargs.pop('model_type', NNHandler.ModelType.CLASSIFICATION)
        # DDP flag not handled by this deprecated method
        mkw = kwargs.pop('model_kwargs', {})

        # Instantiate handler (will auto-detect DDP if env vars set)
        h = NNHandler(mc, device=dev, logger_mode=logm, logger_filename=logf, logger_level=logl,
                      save_model_code=smc, model_type=mt, **mkw)

        h.set_optimizer(kwargs.pop('optimizer_class'), **kwargs.pop('optimizer_kwargs', {}))
        h.set_scheduler(kwargs.pop('scheduler_class', None), **kwargs.pop('scheduler_kwargs', {}))
        # Data loaders will use DDP sampler if handler is in distributed mode
        h.set_train_loader(kwargs.pop('train_data'), **kwargs.pop('train_loader_kwargs', {}))
        h.set_loss_fn(kwargs.pop('loss_fn'), kwargs.pop('pass_epoch_to_loss', False),
                      **kwargs.pop('loss_fn_kwargs', {}))

        if 'val_data' in kwargs:
            h.set_val_loader(kwargs.pop('val_data'), **kwargs.pop('val_loader_kwargs', {}))
        if 'sde_class' in kwargs:
            h.set_sde(kwargs.pop('sde_class'), **kwargs.pop('sde_kwargs', {}))
        h.seed = kwargs.pop('seed', None)  # Use seed setter

        # Auto save config
        if 'auto_save_interval' in kwargs:
            h.auto_save(
                interval=kwargs.pop('auto_save_interval'),
                save_path=kwargs.pop('auto_save_path', '.'),
                name=kwargs.pop('auto_save_name', 'model_epoch{epoch:02d}'),
                overwrite=kwargs.pop('auto_save_overwrite', False)
            )

        if kwargs:  # Check for unused arguments
            warnings.warn(f"Unused arguments passed to deprecated NNHandler.initialize: {list(kwargs.keys())}",
                          RuntimeWarning)
        return h

    @staticmethod
    def initialize_from_checkpoint(checkpoint_path: str,
                                   model_class: type[nn.Module],  # Require model class for instantiation
                                   model_type: Optional[Union[ModelType, str]] = None,
                                   device: Optional[Union[str, torch.device]] = None,
                                   strict_load: bool = True,  # Default strict for loading weights
                                   **model_kwargs) -> 'NNHandler':
        """
        Initializes a new NNHandler instance loading ONLY model weights from a checkpoint file.

        This is useful for inference or transfer learning. Optimizer, scheduler, history, etc.,
        are *not* loaded. Assumes the checkpoint file contains *only* the model's state_dict,
        or a dictionary where the weights are under the key "model_state_dict".

        In DDP mode, all ranks perform this initialization and load the weights onto their
        assigned device.

        Args:
            checkpoint_path (str): Path to the checkpoint file (model state_dict).
            model_class (type[nn.Module]): The model class to instantiate.
            model_type (Optional[Union[ModelType, str]]): The type of the model. Defaults to CLASSIFICATION.
            device (Optional[Union[str, torch.device]]): Target device. If None, uses DDP default or auto-detect.
            strict_load (bool): Whether to strictly enforce state_dict key matching. Defaults to True.
            **model_kwargs: Keyword arguments for the model constructor.

        Returns:
            NNHandler: An instance with the specified model and loaded weights.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        # --- Determine Device and Map Location ---
        is_distributed_load = dist.is_available() and dist.is_initialized()
        current_rank = dist.get_rank() if is_distributed_load else 0

        if device is None:
            if is_distributed_load:
                local_rank = int(os.environ.get('LOCAL_RANK', -1))
                map_location = f'cuda:{local_rank}' if torch.cuda.is_available() and local_rank >= 0 else 'cpu'
            else:
                map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            # Use static method context for resolving device string/object
            resolved_map_location_device = _resolve_device(device)
            map_location = str(resolved_map_location_device)

        print(
            f"INFO (Rank {current_rank}): Initializing handler from weights checkpoint '{os.path.basename(checkpoint_path)}' with map_location='{map_location}'")

        # --- Load State Dict ---
        try:
            # Use weights_only=True if possible, but load full dict first to check structure
            loaded_data = torch.load(checkpoint_path, map_location=map_location)
        except Exception as e:
            raise RuntimeError(
                f"Rank {current_rank}: Failed to load checkpoint from {checkpoint_path} using map_location '{map_location}': {e}") from e

        # Check if it's a full handler state dict or just model weights
        if isinstance(loaded_data, dict) and "model_state_dict" in loaded_data:
            # Checkpoint contains full handler state, extract model weights
            if current_rank == 0:  # Warn only once
                warnings.warn(
                    "Checkpoint seems to contain full handler state, but initialize_from_checkpoint loads *only* model weights.",
                    RuntimeWarning)
            state_dict = loaded_data["model_state_dict"]
        elif isinstance(loaded_data, dict):
            # Assume it's just the model state_dict
            state_dict = loaded_data
        else:
            raise TypeError(
                f"Expected checkpoint '{checkpoint_path}' to contain a state_dict (dict), but got {type(loaded_data)}")

        # --- Instantiate Handler ---
        eff_model_type = model_type if model_type is not None else NNHandler.ModelType.CLASSIFICATION
        # Handler init sets up DDP env and device correctly
        handler = NNHandler(model_class=model_class,
                            device=map_location,  # Pass the determined device
                            model_type=eff_model_type,
                            use_distributed=is_distributed_load,  # Match current env
                            **model_kwargs)
        # handler._model is now initialized and potentially wrapped

        # --- Load Weights into Model ---
        # Adjust keys for DDP/DP mismatch
        saved_parallel = any(k.startswith('module.') for k in state_dict.keys())
        current_parallel = isinstance(handler._model, (DDP, nn.DataParallel))
        load_state_dict = state_dict
        if saved_parallel and not current_parallel:
            load_state_dict = OrderedDict((k[len("module."):], v) for k, v in state_dict.items())
        elif not saved_parallel and current_parallel:
            load_state_dict = OrderedDict(('module.' + k, v) for k, v in state_dict.items())

        # Load into the underlying module
        try:
            missing_keys, unexpected_keys = handler.module.load_state_dict(load_state_dict, strict=strict_load)
            if handler.__logger:  # Log on rank 0
                if missing_keys: handler.__logger.warning(f" Missing keys when loading weights: {missing_keys}")
                if unexpected_keys: handler.__logger.warning(
                    f" Unexpected keys when loading weights: {unexpected_keys}")
                handler.__logger.info(f"Model weights loaded successfully from {os.path.basename(checkpoint_path)}.")
                handler.__logger.warning(
                    "Initialized from weights checkpoint: Optimizer, scheduler, history, etc., are NOT loaded.")
        except Exception as e:
            raise RuntimeError(
                f"Rank {current_rank}: Failed to load model weights from checkpoint state_dict: {e}") from e

        # Barrier to ensure all ranks finish loading weights
        if is_distributed_load:
            dist.barrier()

        return handler

    def __str__(self) -> str:
        """Provides a string representation of the handler's status (rank 0 oriented)."""
        # Call print method to generate the string
        return self.print(show_model_structure=False)

    def print(self, show_model_structure=False) -> str:
        """
        Prints or returns a detailed status summary of the handler.
        In DDP mode, Rank 0 provides the most complete info (history, etc.).

        Args:
            show_model_structure (bool): If True, includes the model structure string (rank 0 only).

        Returns:
            str: Formatted string containing the handler status.
        """
        # Gather info, potentially different on ranks but only rank 0 prints comprehensive details
        model_cls_name = self._model_class.__name__ if self._model_class else 'None'
        model_type_name = self._model_type.name if self._model_type else 'N/A'
        try:
            # Count parameters of the base model
            num_params = f"{self.count_parameters():,}" if self._model else "N/A"
        except RuntimeError:
            num_params = "Error counting"

        dev_str = str(self._device)
        # History length based on rank 0's stored history
        num_epochs = len(self._train_losses) if self._rank == 0 else 'N/A (Rank 0)'

        opt_str = self._optimizer.__class__.__name__ if self._optimizer else "None"
        sched_str = self._scheduler.__class__.__name__ if self._scheduler else "None"
        loss_fn_str = getattr(self._loss_fn, '__name__', str(self._loss_fn)) if self._loss_fn else "None"
        metrics_str = ", ".join(self._metrics.keys()) if self._metrics else "None"
        cbs_str = ", ".join(cb.__class__.__name__ for cb in self._callbacks) if self._callbacks else "None"
        train_loader_set = 'Set' if self._train_loader else 'Not Set'
        val_loader_set = 'Set' if self._val_loader else 'Not Set'

        # AutoSaver details (consistent across ranks, but action on rank 0)
        auto_save_interval = self._auto_saver.save_interval
        auto_save_enabled = 'Enabled' if auto_save_interval is not None and auto_save_interval != 0 else 'Disabled'
        auto_save_details = f" (Interval:{auto_save_interval}, Path:'{self._auto_saver.save_path}', Name:'{self._auto_saver.save_model_name}')" if auto_save_enabled == 'Enabled' else ''

        # EMA / AMP status
        ema_enabled = self._ema is not None
        ema_decay_str = f"{self._ema_decay:.4f}" if ema_enabled else 'N/A'
        amp_enabled = self._grad_scaler.is_enabled() if hasattr(self._grad_scaler, 'is_enabled') else False

        # --- Build String ---
        repr_str = f"--- NNHandler Status (Rank {self._rank}) ---\n"
        repr_str += f"  Distributed:      {self._distributed} (World Size: {self._world_size})\n"
        repr_str += f"  Device:           {dev_str}\n"
        repr_str += f"  Model Class:      {model_cls_name}\n"
        repr_str += f"  Model Type:       {model_type_name}\n"
        repr_str += f"  Trainable Params: {num_params}\n"
        repr_str += f"  Trained Epochs:   {num_epochs}\n"  # Clarify history source
        repr_str += f"  Optimizer:        {opt_str}\n"
        repr_str += f"  Scheduler:        {sched_str}\n"
        repr_str += f"  Loss Function:    {loss_fn_str}\n"
        repr_str += f"  Metrics:          {metrics_str}\n"
        repr_str += f"  Callbacks:        {cbs_str}\n"
        repr_str += f"  Train Loader:     {train_loader_set}\n"
        repr_str += f"  Val Loader:       {val_loader_set}\n"
        repr_str += f"  Auto Saving:      {auto_save_enabled}{auto_save_details}\n"
        repr_str += f"  EMA Enabled:      {ema_enabled} (Decay: {ema_decay_str})\n"
        repr_str += f"  AMP Enabled:      {amp_enabled}\n"
        repr_str += f"  Model Compiled:   {self._compiled_model}\n"  # Added compiled status

        # Show model structure only on rank 0 if requested
        if show_model_structure and self._rank == 0:
            try:
                # Get string representation of the underlying module
                model_structure_str = str(self.module)
            except Exception as e:
                model_structure_str = f"Error getting model structure: {e}"
            # Basic indentation for readability
            indented_model_str = "    " + model_structure_str.replace('\n', '\n    ')
            repr_str += f"  Model Structure (Rank 0):\n{indented_model_str}\n"

        return repr_str

    # --- Model Interaction ---

    def __call__(self, *args, **kwargs) -> Any:
        """
        Performs a forward pass using the underlying model.
        Handles DDP/DataParallel wrapping automatically.
        For score-based models, this typically computes the score.
        """
        if self._model is None:
            raise RuntimeError("Model has not been set.")

        # Special handling for score-based models if __call__ should invoke score()
        # This assumes score() is the primary inference method for these models.
        if self._model_type == self.ModelType.SCORE_BASED:
            # Delegate to the score method if it exists
            if hasattr(self, 'score'):
                # score() expects (t, x, *args) - need to match args/kwargs
                # This might require adaptation based on how __call__ is used for score models
                # Assuming args[0]=t, args[1]=x for now, which is fragile.
                # A dedicated predict_score method might be better.
                # Let's default to model forward for now, assuming score() is called explicitly when needed.
                # return self.score(*args, **kwargs) # <-- Potential issue with arg matching
                pass  # Fall through to model forward, user should call score() explicitly

        # self.model() correctly calls the forward method of the wrapped/unwrapped model
        return self.model(*args, **kwargs)

    def freeze_module(self, module: nn.Module) -> None:
        """
        Freezes the parameters of a given PyTorch module.

        Args:
            module: The PyTorch module (e.g., a layer or the entire model) to freeze.
        """
        if self.__logger:
            self.__logger.info(f"Freezing parameters for module: {module.__class__.__name__}")
        for param in module.parameters():
            param.requires_grad = False
        if self.__logger:
            self.__logger.info(
                f"Module parameters frozen. Model now contains {self.count_parameters(True)} trainable parameters.")

    @torch.no_grad()
    def predict(self, data_loader: DataLoader, apply_ema: bool = True) -> Optional[List[Any]]:
        """
        Performs inference on a DataLoader.
        In DDP mode, each rank predicts on its data subset, and results are
        gathered and returned only on Rank 0.

        Args:
            data_loader (DataLoader): DataLoader for prediction. Must use
                DistributedSampler with shuffle=False in DDP mode for ordered results.
            apply_ema (bool): If True and EMA is enabled, use EMA weights for prediction.

        Returns:
            Optional[List[Any]]: A list containing the outputs for each batch, aggregated
                                 in order on Rank 0. Returns None on non-zero ranks in DDP mode.
                                 The structure depends on the model's output.
        """
        if self._model is None:
            raise RuntimeError("Model has not been set.")
        if not isinstance(data_loader, DataLoader):
            raise TypeError("predict requires a torch.utils.data.DataLoader instance.")

        # Check if the dataloader sampler is appropriate for DDP prediction
        if self._distributed:
            if not isinstance(data_loader.sampler, DistributedSampler):
                # Warn only on rank 0
                if self._rank == 0:
                    warnings.warn(
                        "Predicting in DDP mode, but DataLoader does not use DistributedSampler. Results might be incorrect or duplicated.",
                        RuntimeWarning)
            elif data_loader.sampler.shuffle:
                if self._rank == 0:
                    warnings.warn(
                        "Predicting in DDP mode with a shuffled DistributedSampler. Gathered results might not be in the original dataset order.",
                        RuntimeWarning)

        self.eval(activate=True, log=False)  # Set model to evaluation mode
        local_predictions = []  # Predictions collected on this rank

        # Progress bar only on rank 0
        pbar_predict = None
        is_rank_0 = self._rank == 0
        if is_rank_0 and _tqdm_available:
            try:
                pbar_predict = tqdm(total=len(data_loader), desc="Predicting", leave=False, dynamic_ncols=True)
            except TypeError:  # Handle dataloaders without __len__
                pbar_predict = tqdm(desc="Predicting", leave=False, dynamic_ncols=True)

        # Create iterator, wrap with pbar update if pbar exists
        predict_iterator = enumerate(data_loader)
        if pbar_predict:
            def _predict_pbar_update_iterator(iterator, pbar):
                for idx, data in iterator:
                    yield idx, data
                    pbar.update(1)

            predict_iterator = _predict_pbar_update_iterator(predict_iterator, pbar_predict)

        # Apply EMA weights context manager if enabled
        ema_context = self._ema.average_parameters() if (self._ema and apply_ema) else contextlib.nullcontext()

        with ema_context:
            for batch_idx, batch_data in predict_iterator:
                # Prepare batch - assuming prediction uses standard input format
                # Use _prepare_batch for device placement and basic handling
                # We typically only need 'inputs' for prediction.
                try:
                    # Pass only the input part of the batch if possible?
                    # _prepare_batch expects the full batch structure usually.
                    prepared_batch = self._prepare_batch(batch_data)
                    inputs = prepared_batch["inputs"]
                    additional_params = {k: v for k, v in prepared_batch.items() if k not in ["inputs", "targets"]}
                except ValueError as e:
                    # Fallback for simple cases if _prepare_batch fails
                    if isinstance(batch_data, (list, tuple)) and batch_data:
                        inputs = batch_data[0].to(self._device, non_blocking=True)
                        additional_params = {}
                    elif isinstance(batch_data, Tensor):
                        inputs = batch_data.to(self._device, non_blocking=True)
                        additional_params = {}
                    else:
                        # Cannot determine input, re-raise error with context
                        raise ValueError(
                            f"Cannot determine input from batch type {type(batch_data)} for predict: {e}") from e

                # Forward pass using the model (handles DDP/DP)
                # Use __call__ or directly self.model? Let's use self.model for clarity.
                outputs = self.model(inputs, **additional_params)

                # Store local predictions (move to CPU to save GPU memory and for gathering)
                # Detach from computation graph
                if isinstance(outputs, Tensor):
                    local_predictions.append(outputs.cpu().detach())
                elif isinstance(outputs, (list, tuple)):  # Handle multiple tensor outputs
                    local_predictions.append([o.cpu().detach() if isinstance(o, Tensor) else o for o in outputs])
                elif isinstance(outputs, dict):  # Handle dictionary outputs
                    local_predictions.append(
                        {k: v.cpu().detach() if isinstance(v, Tensor) else v for k, v in outputs.items()})
                else:
                    local_predictions.append(outputs)  # Append non-tensor outputs as is

        # Close prediction progress bar
        if pbar_predict:
            pbar_predict.close()

        # --- Gather results on Rank 0 ---
        all_predictions = None
        if self._distributed:
            # Barrier to ensure all ranks finish prediction loop before gathering
            if self.__logger: self.__logger.debug(f"Rank {self._rank} finished predict loop, waiting at barrier.")
            dist.barrier()

            # Use gather_object for general Python objects (like lists of tensors/dicts)
            # Note: This can be slow and memory-intensive on rank 0 for large datasets.
            # Consider torch.distributed.gather for tensor outputs if possible and order is guaranteed.
            gathered_list = [None] * self._world_size
            if self.__logger: self.__logger.debug(f"Rank {self._rank} starting gather_object.")
            try:
                dist.gather_object(
                    local_predictions,
                    gathered_list if is_rank_0 else None,
                    dst=0
                )
            except Exception as e:
                # Log error on rank 0
                if self.__logger: self.__logger.error(f"Failed during dist.gather_object: {e}", exc_info=True)
                # Ensure barrier is still hit
                dist.barrier()
                raise RuntimeError(f"Rank {self._rank}: Failed during prediction gathering: {e}") from e

            if self.__logger: self.__logger.debug(f"Rank {self._rank} finished gather_object.")

            if is_rank_0:
                # Concatenate results from all ranks IN ORDER.
                # Assumes DistributedSampler(shuffle=False) was used.
                # `gathered_list` contains one list of batch outputs per rank.
                # We need to interleave these based on the original dataset order.
                # Simplified approach: Just concatenate the lists.
                # This assumes batch sizes are constant and world size divides dataset size,
                # or that drop_last=True was used in the sampler. Otherwise order might be wrong.
                all_predictions = []
                # Crude concatenation - might not preserve order perfectly if batches uneven
                for rank_preds in gathered_list:
                    if rank_preds is not None:  # Ensure the gathered object is not None
                        all_predictions.extend(rank_preds)

                if self.__logger:
                    self.__logger.info(f"Gathered predictions from {self._world_size} ranks on Rank 0.")
                    # Log shapes/types for debugging?
                    # if all_predictions:
                    #    first_item = all_predictions[0]
                    #    if isinstance(first_item, Tensor): shape_info = tuple(first_item.shape)
                    #    elif isinstance(first_item, list): shape_info = f"list[{len(first_item)} items]"
                    #    elif isinstance(first_item, dict): shape_info = f"dict[{list(first_item.keys())}]"
                    #    else: shape_info = type(first_item)
                    #    self.__logger.debug(f" Shape/Type of first gathered prediction item: {shape_info}")

            # Barrier after gathering and processing on rank 0
            if self.__logger: self.__logger.debug(f"Rank {self._rank} waiting at barrier after gather.")
            dist.barrier()

        else:
            # Non-distributed: local predictions are all predictions
            all_predictions = local_predictions

        # Restore model to train mode? Optional, depends on expected usage after predict.
        # self._model.train()

        # Return collected predictions only on Rank 0
        return all_predictions if is_rank_0 else None

    def eval(self, activate: bool = True, log:bool = True):
        """Sets model to evaluation or training mode (all ranks)."""
        if self._model is None:
            return
        if activate:
            self._model.eval()
            # Log only on rank 0
            if self.__logger and log:
                self.__logger.info(f"Model set to eval() mode (Rank {self._rank}).")
        else:
            self._model.train()
            for module in self._modules_always_eval:
                module.eval()
            # Log only on rank 0
            if self.__logger and log:
                self.__logger.info(f"Model set to train() mode (Rank {self._rank}).")

    def keep_eval_on_module(self, module: nn.Module, activate: bool = True):
        """
        Manages a module's evaluation state by ensuring it always remains in evaluation mode or returns
        to training mode when specified. The method modifies an internal list that tracks modules set
        to always remain in evaluation mode.

        Args:
            module (nn.Module): The neural network module whose evaluation state needs
                to be managed.
            activate (bool): A flag indicating whether to keep the module in evaluation
                mode (True) or allow it to return to training mode (False). Defaults to True.
        """
        if activate and module not in self._modules_always_eval:
            self._modules_always_eval.append(module)
            module.eval()
        elif activate is False and module in self._modules_always_eval:
            self._modules_always_eval.remove(module)
            module.train()
        else:
            print(f"keep_eval_on_module: Module {module} is already in {self._modules_always_eval}.")

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Counts model parameters (uses the underlying unwrapped module)."""
        if self._model is None:
            return 0
        # Count parameters of the raw model, even if wrapped
        model_to_count = self.module
        params = model_to_count.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        else:
            return sum(p.numel() for p in params)

    # --- Plotting Methods (Rank 0 Only) ---
    def plot_losses(self, log_y_scale: bool = False, save_path: Optional[str] = None):
        """Plots training and validation losses (rank 0 only)."""
        # Only Rank 0 performs plotting
        if self._rank != 0:
            return

        if not _matplotlib_available:
            warnings.warn("Matplotlib not found. Cannot plot losses. Install with 'pip install matplotlib'",
                          RuntimeWarning)
            return

        # Use aggregated history stored on rank 0
        train_loss_history = self.train_losses
        val_loss_history = self.val_losses

        if not train_loss_history and not val_loss_history:
            print("INFO (Rank 0): No loss history recorded to plot.")
            return

        epochs = range(1, len(train_loss_history) + 1)  # Use train loss length for x-axis
        plt.figure(figsize=(5, 3))

        # Plot training loss if available
        if train_loss_history:
            plt.plot(epochs, train_loss_history, label='Training Loss', marker='.', linestyle='-', ms=4, lw=1)

        # Plot validation loss only where it's not NaN
        valid_val_indices = [i for i, loss in enumerate(val_loss_history) if not math.isnan(loss)]
        if valid_val_indices:
            valid_epochs = [epochs[i] for i in valid_val_indices]
            valid_val_data = [val_loss_history[i] for i in valid_val_indices]
            plt.plot(valid_epochs, valid_val_data, label='Validation Loss', marker='x', linestyle='--', ms=4, lw=1)

        if log_y_scale:
            plt.yscale("log")

        plt.title("Training and Validation Loss (Aggregated)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(frameon=False)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()  # Adjust layout

        if save_path:
            try:
                # Ensure directory exists before saving
                save_dir = os.path.dirname(save_path)
                if save_dir: os.makedirs(save_dir, exist_ok=True)
                plt.savefig(save_path)
                if self.__logger:
                    self.__logger.info(f"Loss plot saved by Rank 0 to: {save_path}")
                plt.close()  # Close the plot figure after saving
            except Exception as e:
                # Log error if saving fails
                if self.__logger:
                    self.__logger.error(f"Rank 0 failed to save loss plot to {save_path}: {e}", exc_info=True)
                # Optionally show plot if saving failed? Or just raise error?
                # plt.show() # Show if save fails
                plt.close()  # Close plot even if saving failed
                print(f"ERROR (Rank 0): Could not save loss plot to {save_path}. See logs for details.",
                      file=sys.stderr)
        else:
            # Show plot if no save path provided
            plt.show()
            plt.close()  # Close plot after showing

    def plot_metrics(self, log_y_scale: bool = False, save_path_prefix: Optional[str] = None):
        """Plots training and validation metrics (rank 0 only)."""
        # Only Rank 0 performs plotting
        if self._rank != 0:
            return

        if not _matplotlib_available:
            warnings.warn("Matplotlib not found. Cannot plot metrics. Install with 'pip install matplotlib'",
                          RuntimeWarning)
            return

        if not self._metrics:
            print("INFO (Rank 0): No metrics configured to plot.")
            return

        # Use aggregated history stored on rank 0
        train_metrics_hist = self.train_metrics_history
        val_metrics_hist = self.val_metrics_history

        # Use length of train_losses as reference for total epochs recorded
        num_epochs_recorded = len(self.train_losses)
        if num_epochs_recorded == 0:
            print("INFO (Rank 0): No training history found to plot metrics against.")
            return
        epochs_axis = range(1, num_epochs_recorded + 1)

        # Plot each metric in a separate figure
        for name in self._metrics.keys():
            train_metric_data = train_metrics_hist.get(name, [])
            val_metric_data = val_metrics_hist.get(name, [])

            # Ensure data length matches recorded epochs (can happen if loaded partial history)
            train_metric_data = train_metric_data[:num_epochs_recorded]
            val_metric_data = val_metric_data[:num_epochs_recorded]

            if not train_metric_data and not any(not math.isnan(m) for m in val_metric_data):
                # Skip plotting if no data exists for this metric
                if self.__logger: self.__logger.debug(f"Skipping plot for metric '{name}': No data recorded.")
                continue

            plt.figure(figsize=(5, 3))
            has_plot_data = False

            # Plot training metric if available
            if train_metric_data:
                plt.plot(epochs_axis[:len(train_metric_data)], train_metric_data, label=f'Train {name}', marker='.',
                         linestyle='-', ms=4, lw=1)
                has_plot_data = True

            # Plot validation metric only where not NaN
            valid_val_indices = [i for i, metric in enumerate(val_metric_data) if not math.isnan(metric)]
            if valid_val_indices:
                valid_epochs = [epochs_axis[i] for i in valid_val_indices]
                valid_val_metric = [val_metric_data[i] for i in valid_val_indices]
                plt.plot(valid_epochs, valid_val_metric, label=f'Validation {name}', marker='x', linestyle='--', ms=4,
                         lw=1)
                has_plot_data = True

            if not has_plot_data:
                plt.close()  # Close figure if nothing was plotted
                continue

            if log_y_scale:
                plt.yscale("log")

            plt.title(f"Training and Validation Metric: {name} (Aggregated)")
            plt.xlabel("Epoch")
            plt.ylabel(name)
            plt.legend(frameon=False)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.tight_layout()

            if save_path_prefix:
                # Sanitize metric name for filename
                safe_metric_name = ''.join(c if c.isalnum() else '_' for c in name)
                filepath = f"{save_path_prefix}_metric_{safe_metric_name}.png"
                try:
                    save_dir = os.path.dirname(filepath)
                    if save_dir: os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(filepath)
                    if self.__logger:
                        self.__logger.info(f"Metric plot for '{name}' saved by Rank 0 to: {filepath}")
                    plt.close()  # Close plot after saving
                except Exception as e:
                    if self.__logger:
                        self.__logger.error(f"Rank 0 failed to save metric plot '{name}' to {filepath}: {e}",
                                            exc_info=True)
                    plt.close()  # Close plot even if save fails
                    print(f"ERROR (Rank 0): Could not save metric plot '{name}' to {filepath}. See logs.",
                          file=sys.stderr)
            else:
                plt.show()
                plt.close()  # Close plot after showing

    def compile_model(self, **kwargs):
        """
        Compiles the model using torch.compile (if available).
        In DDP mode, compilation happens on the underlying module on each rank.
        """
        # Check PyTorch version compatibility first
        if not hasattr(torch, 'compile'):
            # Warn only on rank 0
            if self._rank == 0:
                warnings.warn(
                    "torch.compile not available in this PyTorch version (requires 2.0+). Skipping compilation.",
                    RuntimeWarning)
            return

        if self._model is None:
            raise RuntimeError("Model must be set before compiling.")

        if self._compiled_model:
            if self.__logger:  # Log on rank 0
                self.__logger.warning("Model is already compiled. Skipping recompilation.")
            return

        try:
            if self.__logger:  # Log on rank 0
                self.__logger.info(f"Compiling model with torch.compile (Rank {self._rank})... Options: {kwargs}")

            start_time = time.time()
            # Compile the underlying module on each rank
            # DDP should handle the compiled module correctly.
            compiled_module = torch.compile(self.module, **kwargs)

            # If using DDP/DP, the wrapper holds the original module. We need to replace it.
            if isinstance(self._model, (DDP, nn.DataParallel)):
                self._model.module = compiled_module
            else:
                # If not wrapped, the model itself is the module
                self._model = compiled_module

            self._compiled_model = True
            end_time = time.time()

            if self.__logger:  # Log on rank 0
                self.__logger.info(
                    f"Model compiled successfully on Rank {self._rank} in {end_time - start_time:.2f} seconds.")

        except Exception as e:
            if self.__logger:  # Log error on rank 0
                self.__logger.error(f"Failed to compile model on Rank {self._rank}: {e}", exc_info=True)
            # Should we revert? Compilation failure might leave model in unusable state.
            # Re-raising might be safer to halt execution.
            raise e  # Re-raise the exception

    # --- Score-Based Methods ---
    def score(self, t: Tensor, x: Tensor, *args) -> Tensor:
        """Computes the score function s(t, x) = ∇ log p_t(x)."""
        if self._model_type != self.ModelType.SCORE_BASED:
            raise NotImplementedError("Score function is only supported for SCORE_BASED models.")
        if self._sde is None:
            raise RuntimeError("SDE must be set to compute the score.")
        if self._model is None:
            raise RuntimeError("Model must be set to compute the score.")

        # Ensure t and x are tensors on the correct device for this rank
        # Convert time t to tensor if it's not already
        if not isinstance(t, Tensor):
            t = torch.tensor(t, device=self._device)
        # Ensure t is broadcastable to x's batch dimension
        t_dev = t.expand(x.shape[0]).to(self._device)
        x_dev = x.to(self._device)

        # Get model output using the underlying module
        # Pass through any additional args
        model_output = self.module(t_dev, x_dev, *args)

        # Get sigma(t) from SDE
        sigma_t = self._sde.sigma(t_dev)  # Assume sigma(t) returns tensor of shape (B,)

        # Ensure sigma_t has correct shape for broadcasting division (B, 1, 1, ...)
        _, *D_x = x_dev.shape  # Get spatial/feature dimensions
        sigma_t_reshaped = sigma_t.view(-1, *[1] * len(D_x))

        # Calculate score: model_output / sigma(t)
        # Add small epsilon for numerical stability if sigma can be zero
        score_val = model_output / (sigma_t_reshaped + 1e-8)

        return score_val

    def make_pos_grid(self, x0: int, y0: int, p: int, H: int):
        """Return a (1,2,p,p) tensor with x/y in [-1,1]."""
        xs = torch.arange(x0, x0 + p, device=self.device)
        ys = torch.arange(y0, y0 + p, device=self.device)
        xg = ((xs.view(1, -1).repeat(p, 1) / (H - 1)) - 0.5) * 2
        yg = ((ys.view(-1, 1).repeat(1, p) / (H - 1)) - 0.5) * 2
        return torch.stack([xg, yg], 0).unsqueeze(0)  # (1,2,p,p)

    def patch_score_vectorized(self, t, x, patch_size=16, stride=6, patch_chunk=None):
        """
        Vectorized patch_score: computes score for all overlapping patches and averages overlaps.

        Args:
            t: (B,) tensor, batch of scalar times
            x: (B,C,H,W) input images
            patch_size: (int) size of square patch
            stride: (int) stride between patches
            patch_chunk: (int or None) number of *patches* (not images) to score per chunk.
                         If None, computes in a single pass.

        Returns:
            (B, C, H, W) score image
        """
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype

        # Number of sliding-window positions
        patch_H = (H - patch_size) // stride + 1
        patch_W = (W - patch_size) // stride + 1
        num_patches = patch_H * patch_W

        # Unfold image into sliding windows (view-like layout)
        # x_unf: (B, C*patch_size*patch_size, num_patches)
        x_unf = F.unfold(x, kernel_size=patch_size, stride=stride)

        # Helper: top-left coordinates (y, x) for each patch index [0..num_patches)
        rows = torch.arange(0, H - patch_size + 1, stride, device=device)
        cols = torch.arange(0, W - patch_size + 1, stride, device=device)
        ii, jj = torch.meshgrid(rows, cols, indexing='ij')  # (patch_H, patch_W)
        top_y = ii.reshape(-1)  # (num_patches,)
        left_x = jj.reshape(-1)  # (num_patches,)

        if patch_chunk is None:
            # --- Single pass ---
            # Build all patch tensors
            x_patches = x_unf.permute(0, 2, 1).reshape(B * num_patches, C, patch_size, patch_size)

            # Build all position grids
            pos_list = [
                self.make_pos_grid(int(x0), int(y0), patch_size, H)  # (1,2,p,p)
                for y0, x0 in zip(top_y.tolist(), left_x.tolist())
            ]
            pos_grids = torch.cat(pos_list, dim=0)  # (num_patches, 2, p, p)
            pos_grids = pos_grids.unsqueeze(0).repeat(B, 1, 1, 1, 1).reshape(
                B * num_patches, 2, patch_size, patch_size
            )

            # Repeat t for all patches
            t_exp = t.unsqueeze(1).expand(B, num_patches).reshape(-1)

            # Model prediction: (B*num_patches, C, p, p)
            s = self.score(t_exp, x_patches, pos_grids)

            # Prepare for fold: (B, C*p*p, num_patches)
            s_cols = s.reshape(B, num_patches, C * patch_size * patch_size).transpose(1, 2)

        else:
            # --- Chunked pass ---
            patch_chunk = max(1, int(patch_chunk))

            # We accumulate per-patch columns and do a single fold at the end.
            s_cols = torch.empty(
                B, C * patch_size * patch_size, num_patches, device=device, dtype=dtype
            )

            # Pre-expand time in a streaming-friendly way
            # (we materialize only per chunk below)
            for start in range(0, num_patches, patch_chunk):
                end = min(start + patch_chunk, num_patches)
                npi = end - start  # number of patches in this chunk

                # Take the corresponding patch columns (view), then reshape to (B*npi, C, p, p)
                x_chunk = x_unf[:, :, start:end].permute(0, 2, 1).reshape(
                    B * npi, C, patch_size, patch_size
                )

                # Build only the needed position grids for this chunk
                pos_list = [
                    self.make_pos_grid(int(x0), int(y0), patch_size, H)
                    for y0, x0 in zip(top_y[start:end].tolist(), left_x[start:end].tolist())
                ]
                pos_chunk = torch.cat(pos_list, dim=0)  # (npi, 2, p, p)
                pos_chunk = pos_chunk.unsqueeze(0).repeat(B, 1, 1, 1, 1).reshape(
                    B * npi, 2, patch_size, patch_size
                )

                # Time for this chunk
                t_chunk = t.unsqueeze(1).expand(B, npi).reshape(-1)

                # Score this chunk
                s_chunk = self.score(t_chunk, x_chunk, pos_chunk)  # (B*npi, C, p, p)

                # Write into the correct columns so we can fold once at the end
                s_chunk_cols = s_chunk.reshape(B, npi, C * patch_size * patch_size).transpose(1, 2).contiguous()
                s_cols[:, :, start:end] = s_chunk_cols

        # Fold all patches back to image space (linear; equivalent to summing overlaps)
        score = F.fold(
            s_cols,
            output_size=(H, W),
            kernel_size=patch_size,
            stride=stride,
        )

        # Overlap normalization (how many times each pixel was covered)
        ones = torch.ones((B, C, H, W), device=device, dtype=dtype)
        overlap = F.fold(
            F.unfold(ones, kernel_size=patch_size, stride=stride),
            output_size=(H, W),
            kernel_size=patch_size,
            stride=stride,
        )

        return score / overlap

    @torch.no_grad()
    def sample(self, shape: Tuple[int, ...], steps: int, condition: Optional[list] = None,
               likelihood_score_fn: Optional[Callable] = None, guidance_factor: float = 1.,
               apply_ema: bool = True, bar: bool = True, stop_on_NaN=True, patch_size=None, stride=None,
               patch_chunk=None) -> Optional[
        Tensor]:
        """
        Performs sampling using the reverse SDE (Euler-Maruyama).
        Intended to be run on Rank 0 primarily, returns None on other ranks.
        """
        # Sampling is typically done on rank 0 to avoid redundant computation and gathering issues.
        if self._distributed and self._rank != 0:
            # Add barrier? Depends if rank 0 needs to wait. Assume not for typical sampling.
            return None

        if self._model_type != self.ModelType.SCORE_BASED:
            raise NotImplementedError("Sampling is only supported for SCORE_BASED models.")
        if self._sde is None: raise RuntimeError("SDE must be set for sampling.")
        if self._model is None: raise RuntimeError("Model must be set for sampling.")

        if condition is None: condition = []  # Ensure condition is a list
        self.eval(activate=True, log=False)  # Set model to evaluation mode

        if (patch_size is None) != (stride is None):
            raise ValueError("Both patch_size and stride must be specified together (either both set, or both None).")
        elif (patch_size is not None) and (stride is not None):
            patch_diffusion_mode = True
        else:
            patch_diffusion_mode = False

        # TQDM setup (rank 0 only)
        tqdm_module = None
        if bar and _tqdm_available:
            try:
                from tqdm.auto import tqdm as tqdm_auto
                tqdm_module = tqdm_auto
            except ImportError:
                bar = False  # Disable if import fails despite check

        B, *D = shape  # Batch size and data dimensions
        sampling_from = "prior" if likelihood_score_fn is None else "posterior"
        if self.__logger: self.__logger.info(
            f"Starting sampling (Rank 0): Shape={shape}, Steps={steps}, Source='{sampling_from}'")

        # Define a zero likelihood function if none provided
        if likelihood_score_fn is None:
            def zero_likelihood_score(t, x): return torch.zeros_like(x)

            likelihood_score_fn = zero_likelihood_score

        # Initial sample from prior distribution (on the correct device)
        try:
            x = self._sde.prior(D).sample([B]).to(self._device)
        except Exception as e:
            raise RuntimeError(f"Rank 0 failed to sample from SDE prior: {e}") from e

        # Time schedule
        # Check if SDE class name suggests cosine schedule (avoid direct import)
        is_vpsde = hasattr(self._sde, '__class__') and self._sde.__class__.__name__ == "VPSDE"
        time_schedule = torch.linspace(self._sde.T, self._sde.epsilon, steps + 1, device=self._device)

        if is_vpsde:  # Use cosine schedule for VPSDE based on name match
            t_schedule = torch.tensor([
                self._sde.epsilon + 0.5 * (self._sde.T - self._sde.epsilon) * (1 + math.cos(math.pi * i / steps))
                for i in range(steps + 1)
            ], device=self._device)
            if self.__logger: self.__logger.debug("Using custom cosine time schedule (VPSDE detected).")
        else:
            t_schedule = time_schedule  # Linear schedule otherwise
            if self.__logger: self.__logger.debug("Using linear time schedule.")

        # Prepare iterator with progress bar (rank 0 only)
        pbar_iterator = range(steps)
        if bar and tqdm_module:
            pbar_iterator = tqdm_module(pbar_iterator, desc=f"Sampling ({sampling_from})", dynamic_ncols=True)

        # Apply EMA weights context manager if enabled
        ema_context = self._ema.average_parameters() if (self._ema and apply_ema) else contextlib.nullcontext()
        x_mean = torch.zeros_like(x)  # Initialize x_mean

        with ema_context:
            for i in pbar_iterator:
                t_current = t_schedule[i]
                t_next = t_schedule[i + 1]
                step_dt = t_next - t_current  # dt for this step (should be negative)

                # Ensure t is broadcastable tensor on correct device
                t_batch = torch.ones(B, device=self._device) * t_current

                if t_current.item() <= self._sde.epsilon:  # Check against current t
                    if self.__logger: self.__logger.warning(
                        f"Reached time epsilon ({self._sde.epsilon:.4f}) early at step {i}. Stopping sampling.")
                    break  # Stop if time goes below epsilon

                # Get SDE components g(t) and f(t,x)
                g = self._sde.diffusion(t_batch, x)
                f = self._sde.drift(t_batch, x)

                # Calculate score s(t, x) using the handler's method
                if patch_diffusion_mode:
                    score_val = self.patch_score_vectorized(t_batch, x, patch_size, stride, patch_chunk)
                else:
                    score_val = self.score(t_batch, x, *condition)

                # Calculate likelihood score (user provided or zero)
                likelihood_score_val = likelihood_score_fn(t_batch, x)
                # Combine scores for guided sampling
                combined_score = score_val + guidance_factor * likelihood_score_val

                # Calculate reverse SDE drift: f(t,x) - g(t)^2 * combined_score
                # Ensure g^2 broadcasting works: g likely (B,), needs (B, 1, 1...)
                g_squared_expanded = (g ** 2).view(-1, *[1] * (x.ndim - 1))
                drift = f - g_squared_expanded * combined_score

                # Noise term for Euler-Maruyama step
                # dw ~ N(0, dt) -> sqrt(-dt) * Z where Z ~ N(0, I) (dt is negative)
                noise = torch.randn_like(x)
                dw = noise * torch.sqrt(-step_dt)  # Make sure step_dt is negative

                # Euler-Maruyama step: x_{t-1} = x_t + f_reverse(t, x_t) * dt + g(t) * dw
                x_mean = x + drift * step_dt  # Mean update (deterministic part)
                # Need to ensure g matches noise shape for element-wise multiplication
                # g is likely (B,), noise is (B, C, H, W), need g expanded
                g_expanded = g.view(-1, *[1] * (x.ndim - 1))
                x = x_mean + g_expanded * dw  # Add noise scaled by diffusion

                # Check for numerical issues
                if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
                    error_msg = f"NaN or Inf detected in sampling at step {i + 1} (t={t_current.item():.4f}). Stopping."
                    if stop_on_NaN:
                        raise RuntimeError(error_msg)
                    else:
                        warnings.warn(error_msg, RuntimeWarning)
                        if self.__logger: self.__logger.error(error_msg)
                        # Return the last valid mean if possible, otherwise the corrupted x
                        last_valid_x = x_mean if not (
                                torch.any(torch.isnan(x_mean)) or torch.any(torch.isinf(x_mean))) else x
                        return last_valid_x

                # Update progress bar postfix (rank 0)
                if bar and tqdm_module:
                    postfix_dict = {
                        "t": f"{t_current.item():.3f}",
                        "sigma": f"{g[0].item():.2e}",
                        "|x|": f"{x.abs().mean().item():.2e}"
                    }
                    pbar_iterator.set_postfix(postfix_dict, refresh=False)

        # Return the mean of the last step (often considered cleaner)
        if self.__logger: self.__logger.info(f"Sampling finished on Rank 0.")
        return x_mean

    @torch.no_grad()
    def log_likelihood(self, x: Tensor, *args, ode_steps: int,
                       n_cotangent_vectors: int = 1, noise_type: str = "rademacher",
                       method: str = "Euler", t0: Optional[float] = None, t1: Optional[float] = None,
                       apply_ema: bool = True, pbar: bool = False) -> Optional[Tensor]:
        """
        Estimates log-likelihood via ODE method (Instantaneous Change of Variables).
        Intended to be run on Rank 0 primarily, returns None on other ranks.
        """
        # Log likelihood calculation is typically done on rank 0
        if self._distributed and self._rank != 0:
            return None

        if self._model_type != self.ModelType.SCORE_BASED:
            raise NotImplementedError("Log-likelihood estimation is only supported for SCORE_BASED models.")
        if self._sde is None: raise RuntimeError("SDE must be set for log-likelihood estimation.")
        if self._model is None: raise RuntimeError("Model must be set for log-likelihood estimation.")

        self.eval(activate=True, log=False)  # Ensure model is in eval mode

        # Set default integration times if not provided
        t_start = t0 if t0 is not None else self._sde.epsilon
        t_end = t1 if t1 is not None else self._sde.T

        if not (t_start >= self._sde.epsilon and t_end <= self._sde.T and t_start < t_end):
            raise ValueError(
                f"Invalid time range [{t_start}, {t_end}]. Must be within SDE bounds [{self._sde.epsilon}, {self._sde.T}].")

        if self.__logger:
            self.__logger.info(
                f"Starting log-likelihood estimation (Rank 0): Steps={ode_steps}, Method='{method}', Time=[{t_start:.4f}, {t_end:.4f}]")

        # Ensure input tensor x is on the correct device
        x = x.to(self._device)
        B, *D = x.shape  # Batch size and data dimensions

        # --- Define ODE Drift and Divergence Functions ---
        # Use the underlying model (self.module) for score calculation
        target_model = self.module

        def ode_drift_func(t: Tensor, x_in: Tensor, *drift_args) -> Tensor:
            """Calculates the drift term f_tilde(t, x) for the probability flow ODE."""
            # Ensure t is a tensor broadcastable to x_in's batch dimension
            if not isinstance(t, Tensor): t = torch.tensor(t, device=self._device)
            t_batch = t.expand(x_in.shape[0]).to(self._device)

            f = self._sde.drift(t_batch, x_in)
            g = self._sde.diffusion(t_batch, x_in)

            # Calculate score s(t, x) = model(t, x) / sigma(t)
            sigma_t = self._sde.sigma(t_batch)
            sigma_t_reshaped = sigma_t.view(-1, *[1] * len(D))
            score_val = target_model(t_batch, x_in, *drift_args) / (sigma_t_reshaped + 1e-8)

            # Probability flow ODE drift: f(t,x) - 0.5 * g(t)^2 * s(t,x)
            g_squared_reshaped = (g ** 2).view(-1, *[1] * len(D))
            f_tilde = f - 0.5 * g_squared_reshaped * score_val
            return f_tilde

        # Divergence calculation using Hutchinson trace estimator
        def divergence_func(t: Tensor, x_in: Tensor, *div_args) -> Tensor:
            """Estimates the divergence of the ODE drift using Hutchinson."""
            # Prepare inputs for vjp: repeat inputs for multiple cotangent vectors
            samples = x_in.repeat_interleave(n_cotangent_vectors, dim=0)
            # Ensure t is tensor and repeat
            if not isinstance(t, Tensor): t = torch.tensor(t, device=self._device)
            t_repeated = t.expand(samples.shape[0]).to(self._device)  # Match expanded batch

            # Sample cotangent vectors (noise)
            if noise_type == 'rademacher':
                vectors = torch.randint(low=0, high=2, size=samples.shape, device=self._device).float() * 2 - 1
            elif noise_type == 'gaussian':
                vectors = torch.randn_like(samples)
            else:
                raise ValueError(f"Unknown noise type for Hutchinson estimator: {noise_type}")

            # Define the function whose Jacobian-vector product we need: drift(t, x, *args)
            f_vjp = lambda x_vjp: ode_drift_func(t_repeated, x_vjp, *div_args)

            # Compute vector-Jacobian product (vjp)
            # Need gradients through the drift function calculation
            with torch.enable_grad():
                samples.requires_grad_(True)
                drift_output = f_vjp(samples)
                # vjp computes (vector^T @ Jacobian)
                vjp_product = torch.autograd.grad(drift_output, samples, grad_outputs=vectors, create_graph=False)[0]
                samples.requires_grad_(False)  # Turn off grad requirement after use

            # Calculate divergence estimate: dot product(vectors, vjp_product)
            # Flatten spatial/channel dimensions to compute dot product easily
            vjp_product_flat = vjp_product.flatten(start_dim=1)
            vectors_flat = vectors.flatten(start_dim=1)

            # Sum over feature dimensions: gives one value per expanded sample (B * n_cotangent,)
            div_expanded = torch.sum(vectors_flat * vjp_product_flat, dim=1)

            # Average over Hutchinson samples for each original batch item
            # Reshape to (B, n_cotangent) and average over the n_cotangent dimension
            div_avg = div_expanded.view(B, n_cotangent_vectors).mean(dim=1)
            return div_avg

        # --- ODE Integration Setup ---
        tqdm_module = None
        if pbar and _tqdm_available:
            try:
                from tqdm.auto import tqdm as tqdm_auto
                tqdm_module = tqdm_auto
            except ImportError:
                pbar = False

        log_p = torch.zeros(B, device=self._device)  # Log likelihood accumulator (delta log p)
        current_t = torch.ones(B, device=self._device) * t_start
        dt = (t_end - t_start) / ode_steps  # Step size (positive)

        # Prepare iterator with progress bar (rank 0 only)
        ode_iterator = range(ode_steps)
        if pbar and tqdm_module:
            ode_iterator = tqdm_module(ode_iterator, desc="Log-Likelihood ODE", dynamic_ncols=True)

        # Apply EMA weights context manager if enabled
        ema_context = self._ema.average_parameters() if (self._ema and apply_ema) else contextlib.nullcontext()

        with ema_context:
            for i in ode_iterator:
                step_start_time = time.time()

                # Calculate drift and divergence at current time t with current x
                current_drift = ode_drift_func(current_t, x, *args)
                current_div = divergence_func(current_t, x, *args)

                # --- Perform ODE Step ---
                if method == "Euler":
                    # Update x: x_{t+dt} = x_t + f_tilde(t, x_t) * dt
                    x = x + current_drift * dt
                    # Update log_p: log_p += div(t, x_t) * dt
                    # Divergence is based on x *before* the update for this step's contribution
                    log_p += current_div * dt
                    # Update time
                    current_t += dt

                elif method == "Heun":
                    # Predictor step for x at t+dt
                    x_pred = x + current_drift * dt
                    t_next = current_t + dt
                    # Calculate drift and divergence at t+dt using predicted x
                    next_drift = ode_drift_func(t_next, x_pred, *args)
                    next_div = divergence_func(t_next, x_pred, *args)
                    # Corrector step for x
                    x = x + 0.5 * (current_drift + next_drift) * dt
                    # Update log_p using average of divergences
                    log_p += 0.5 * (current_div + next_div) * dt
                    # Update time
                    current_t = t_next
                else:
                    raise NotImplementedError(f"ODE solver method '{method}' not implemented. Use 'Euler' or 'Heun'.")

                # Check for numerical issues during integration
                if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)) or \
                        torch.any(torch.isnan(log_p)) or torch.any(torch.isinf(log_p)):
                    error_msg = f"NaN or Inf detected during ODE integration at step {i + 1} (t={current_t[0].item():.4f}). Stopping."
                    warnings.warn(error_msg, RuntimeWarning)
                    if self.__logger: self.__logger.error(error_msg)
                    # Return NaN for the affected batch elements or all? Return all NaN.
                    return torch.full_like(log_p, float('nan'))

                # Update progress bar postfix (rank 0)
                if pbar and tqdm_module:
                    step_time = time.time() - step_start_time
                    postfix_dict = {
                        "t": f"{current_t[0].item():.3f}",
                        "step_dt(s)": f"{step_time:.3f}",
                        "logp": f"{log_p.mean().item():.2e}"
                    }
                    ode_iterator.set_postfix(postfix_dict, refresh=False)

        # Add log probability from the prior distribution at the final time T
        try:
            # Ensure x used for prior log_prob is the final integrated x at time t_end
            prior_log_prob = self._sde.prior(D).log_prob(x)
            # Ensure prior log prob is same shape as log_p (Batch,)
            if prior_log_prob.shape != log_p.shape:
                # Handle cases where prior might return different shape (e.g., sum over features)
                # This depends heavily on the prior implementation. Assume it returns (B,) for now.
                if prior_log_prob.ndim == x.ndim and prior_log_prob.shape[0] == B:  # Sum spatial/channel dims
                    prior_log_prob = prior_log_prob.flatten(start_dim=1).sum(dim=1)
                elif prior_log_prob.numel() == B:  # Assume flattenable to B
                    prior_log_prob = prior_log_prob.view(B)
                else:
                    raise ValueError(
                        f"Shape mismatch between accumulated log_p {log_p.shape} and prior log_prob {prior_log_prob.shape}")

            log_p += prior_log_prob
            if self.__logger: self.__logger.debug(
                f"Added prior log probability at t={t_end:.4f}. Mean prior logp: {prior_log_prob.mean().item():.2e}")

        except Exception as e:
            warnings.warn(f"Rank 0 failed to compute or add prior log probability at t={t_end:.4f}: {e}",
                          RuntimeWarning)
            # Return NaN if prior fails, as likelihood is incomplete
            return torch.full_like(log_p, float('nan'))

        if self.__logger: self.__logger.info(
            f"Log-likelihood estimation finished. Mean logp: {log_p.mean().item():.4f}")
        return log_p

# --- Example Usage Helper Function ---
# Note: Callbacks like ModelCheckpoint, EarlyStopping, TensorBoardLogger
# would need to be imported separately and potentially adapted for DDP awareness
# (e.g., saving/logging only on rank 0). For simplicity, they are omitted here.

# Assume dummy components are defined as in nn_handler.py or elsewhere
# from your_module import get_dummy_components

# def run_ddp_example():
#     """Example of running NNHandler with DDP."""
#     # DDP requires setup via torchrun or similar launcher
#     # This function assumes it's being run as part of a DDP process.
#
#     # 1. Initialize DDP (Handled inside NNHandler __init__ based on env vars)
#     # dist.init_process_group(backend='nccl') # Usually done by launcher
#     rank = int(os.environ.get('RANK', 0))
#     local_rank = int(os.environ.get('LOCAL_RANK', 0))
#     world_size = int(os.environ.get('WORLD_SIZE', 1))
#
#     # Determine device for this rank
#     if torch.cuda.is_available():
#         device = f'cuda:{local_rank}'
#         torch.cuda.set_device(device)
#     else:
#         device = 'cpu'
#
#     print(f"[Rank {rank}/{world_size}] Starting DDP Example on device '{device}'")
#
#     # 2. Get Dummy Components
#     components = get_dummy_components(device=device) # Pass rank-specific device
#
#     # 3. Initialize NNHandler (use_distributed=True or None for auto-detect)
#     handler = NNHandler(
#         model_class=components['model_class'],
#         device=device, # Pass initial device guess, DDP init will confirm/override
#         logger_mode=NNHandler.LoggingMode.CONSOLE if rank == 0 else None, # Log only on rank 0
#         logger_level=logging.INFO,
#         model_type=components['model_type'],
#         use_distributed=True, # Explicitly enable DDP check
#         **components['model_kwargs']
#     )
#
#     # 4. Configure Handler (Done on all ranks, DDP ensures consistency where needed)
#     handler.set_optimizer(components['optimizer_class'], **components['optimizer_kwargs'])
#     handler.set_loss_fn(components['loss_fn'])
#     # Loaders will use DistributedSampler automatically due to DDP mode
#     handler.set_train_loader(components['train_dataset'], **components['train_loader_kwargs'])
#     handler.set_val_loader(components['val_dataset'], **components['val_loader_kwargs'])
#     for name, fn in components['metrics'].items():
#         handler.add_metric(name, fn)
#
#     # Add Callbacks (Ensure they are DDP aware, e.g., save/log only on rank 0)
#     # Example: Add a simple custom callback that prints epoch end on rank 0
#     class PrintEpochEndRank0(Callback):
#         def on_epoch_end(self, epoch, logs=None):
#             if self.handler._rank == 0:
#                 print(f"\n[Rank 0 Callback] Epoch {epoch+1} finished. Aggregated Logs: {logs}\n")
#     handler.add_callback(PrintEpochEndRank0())
#
#     # Configure Auto Save (Rank 0 performs save)
#     checkpoint_dir = "./nn_handler_ddp_checkpoints"
#     handler.auto_save(interval=2, save_path=checkpoint_dir, name="ddp_autosave_epoch{epoch:02d}", overwrite=True)
#
#     # Print handler status before training (Rank 0)
#     if rank == 0:
#         print("\n--- Handler Status Before Training (Rank 0) ---")
#         print(handler)
#         print("-" * 40)
#
#     # 5. Train
#     print(f"[Rank {rank}] --- Starting Training ---")
#     handler.train(
#         epochs=5, # Shorter training for example
#         validate_every=1,
#         use_amp=(handler.device.type == 'cuda'), # Use AMP if on CUDA
#         progress_bar=(rank == 0) # Show progress bar only on rank 0
#     )
#     print(f"[Rank {rank}] --- Training Finished ---")
#     dist.barrier() # Wait for all ranks to finish training
#
#     # 6. Post-Training Information (Rank 0)
#     if rank == 0:
#         print("\n--- Handler Status After Training (Rank 0) ---")
#         print(handler)
#         print("-" * 40)
#         print("\n--- Plotting Results (Rank 0) ---")
#         try: handler.plot_losses(save_path="./nn_handler_ddp_losses.png")
#         except Exception as e: print(f"Could not plot losses: {e}")
#         try: handler.plot_metrics(save_path_prefix="./nn_handler_ddp")
#         except Exception as e: print(f"Could not plot metrics: {e}")
#         print("-" * 40)
#
#     # 7. Saving / Loading Test (Rank 0 saves, all load)
#     final_save_path = os.path.join(checkpoint_dir, "ddp_final_model_state.pth")
#     if rank == 0:
#         print(f"\n--- Saving Final Handler State (Rank 0) ---")
#         print(f"Saving to: {final_save_path}")
#     handler.save(final_save_path) # save() handles rank 0 check and barriers
#
#     dist.barrier() # Ensure save completes before loading
#     print(f"[Rank {rank}] --- Loading Handler State ---")
#     if os.path.exists(final_save_path):
#         try:
#             # All ranks load the state, mapping to their device
#             loaded_handler = NNHandler.load(final_save_path, device=device)
#             print(f"[Rank {rank}] Handler loaded successfully.")
#
#             # Example: Run prediction using loaded handler (Rank 0 gets results)
#             if rank == 0: print("[Rank 0] --- Running Prediction with Loaded Handler ---")
#             # Create a DDP loader for prediction (shuffle=False)
#             pred_dataset = components['val_dataset']
#             pred_sampler = DistributedSampler(pred_dataset, num_replicas=world_size, rank=rank, shuffle=False)
#             pred_loader = DataLoader(pred_dataset, batch_size=components['val_loader_kwargs']['batch_size'], sampler=pred_sampler)
#
#             predictions = loaded_handler.predict(pred_loader) # predict() gathers on rank 0
#
#             if rank == 0 and predictions:
#                 print(f"[Rank 0] Prediction successful. Number of predicted batches: {len(predictions)}")
#                 # print(f" Shape of first prediction batch: {predictions[0].shape if isinstance(predictions[0], Tensor) else type(predictions[0])}")
#             elif rank != 0:
#                 assert predictions is None, f"Rank {rank} should receive None from predict()"
#
#         except Exception as e:
#             print(f"[Rank {rank}] ERROR loading or predicting with handler state: {e}")
#             import traceback
#             traceback.print_exc()
#     else:
#         if rank == 0: print(f"ERROR: Final save file not found at {final_save_path}")
#
#     print(f"[Rank {rank}] DDP Example finished.")
#     # dist.destroy_process_group() # Cleanup usually handled by launcher

# if __name__ == "__main__":
#     # To run this example:
#     # 1. Ensure you have the dummy components (model, dataset, loss) available.
#     # 2. Save this entire script as nn_handler_distributed.py (or similar).
#     # 3. Run using torchrun:
#     #    torchrun --nproc_per_node=<num_gpus> nn_handler_distributed.py
#     # Example for 2 GPUs:
#     #    torchrun --nproc_per_node=2 nn_handler_distributed.py
#
#     # Need to define get_dummy_components or import it
#     # For standalone test, let's define it here minimally
#     class DummyModel(nn.Module):
#         def __init__(self, input_features, output_classes):
#           super().__init__()
#           self.layer = nn.Linear(input_features, output_classes)
#         def forward(self, x):
#           return self.layer(x)
#     class DummyDataset(Dataset):
#         def __init__(self, size=64, features=10, classes=2):
#           self.size=size
#           self.data=torch.randn(size, features)
#           self.labels=torch.randint(0,classes,(size,))
#         def __len__(self):
#           return self.size
#         def __getitem__(self, idx):
#           return self.data[idx], self.labels[idx]
#     def dummy_loss(output, target): return nn.CrossEntropyLoss()(output, target)
#     def dummy_accuracy(output, target):
#       preds = torch.argmax(output, dim=1)
#       return (preds == target).float().mean()
#     def get_dummy_components(device='cpu', n_features=10, n_classes=2, batch_size=8, dataset_size=128):
#         return { 'model_class': DummyModel, 'model_kwargs': {'input_features': n_features, 'output_classes': n_classes},
#                  'model_type': NNHandler.ModelType.CLASSIFICATION, 'train_dataset': DummyDataset(dataset_size, n_features, n_classes),
#                  'val_dataset': DummyDataset(dataset_size // 2, n_features, n_classes), 'train_loader_kwargs': {'batch_size': batch_size},
#                  'val_loader_kwargs': {'batch_size': batch_size}, 'optimizer_class': torch.optim.Adam, 'optimizer_kwargs': {'lr': 1e-3},
#                  'loss_fn': dummy_loss, 'metrics': {'accuracy': dummy_accuracy}, 'device': device }
#
#     run_ddp_example()
