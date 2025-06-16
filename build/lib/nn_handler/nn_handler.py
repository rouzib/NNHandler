"""
Author: Nicolas Payot
Date: 08/22/2024

This script provides an extensive python module for handling all the necessary operations related to training,
validating, and managing a PyTorch neural network. This includes the declaration, optimization, and iteration of a
PyTorch model, along with advanced features like metric tracking, a flexible callback system, gradient accumulation,
mixed precision training, and enhanced model checkpointing. Interfaced functionalities for detailed logging and
integration with tools like TensorBoard are implemented as well. The NNHandler class facilitates model definition,
data loading, optimizer/scheduler setup, loss calculation, and manages the entire training lifecycle, aiming to
streamline and accelerate the development and experimentation process with PyTorch models.
"""
import abc
import contextlib
import os
import logging
import sys
import types
import warnings
from collections import OrderedDict, defaultdict
from enum import Enum
import inspect
import ast
from typing import Callable, Union, Optional, Dict, List, Any, Tuple, TypedDict
import math
import time
import copy

import torch
import torch.nn as nn
from torch import Tensor
from torch.func import vjp
from torch.utils.data import DataLoader, Dataset

from .autosaver import AutoSaver
from .callbacks.base import Callback
from .sampler import Sampler

try:
    from torch.amp import GradScaler, autocast

    _amp_available = True
except ImportError:
    _amp_available = False


    # Define dummy classes if amp is not available
    class GradScaler:
        def __init__(self, enabled=False): pass

        def scale(self, loss): return loss

        def step(self, optimizer): optimizer.step()

        def update(self): pass

        def __call__(self, *args, **kwargs): pass  # make it callable for load/save state_dict logic

        def state_dict(self): return {}

        def load_state_dict(self, state_dict): pass


    def autocast(device_type="cpu", enabled=False):
        return torch.autocast(device_type, enabled=False)  # No-op autocast

try:
    from torch_ema import ExponentialMovingAverage

    _ema_available = True
except ImportError:
    _ema_available = False


    # Define a dummy EMA class if not available
    class ExponentialMovingAverage:
        def __init__(self, parameters, decay): pass

        def update(self): pass

        def average_parameters(self): yield  # Dummy context manager

        def copy_to(self, parameters=None): pass

        def state_dict(self): return {}

        def load_state_dict(self, state_dict): pass

try:
    import matplotlib.pyplot as plt

    _matplotlib_available = True
except ImportError:
    _matplotlib_available = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NNHandler:
    r"""A comprehensive wrapper for PyTorch neural networks enabling streamlined training, validation, and management.

       NNHandler provides an interface to manage the entire lifecycle of a PyTorch model,
       including setup, training, validation, metric tracking, checkpointing, and inference.
       It supports various model types, custom components (loss, optimizer, scheduler, metrics),
       advanced training techniques (EMA, AMP, gradient accumulation), and a flexible callback system.

    Attributes:
        _metrics (Dict[str, Callable]): Dictionary of metric functions {name: function}.
        _train_metrics_history (defaultdict[str, List]): History of training metrics per epoch.
        _val_metrics_history (defaultdict[str, List]): History of validation metrics per epoch.
        _callbacks (List[Callback]): List of callbacks attached to the handler.
        _stop_training (bool): Flag used by callbacks (like EarlyStopping) to signal training termination.
        _grad_scaler (GradScaler): Gradient scaler for Automatic Mixed Precision (AMP).
        _ema (Optional[ExponentialMovingAverage]): Exponential Moving Average handler.

    Args:
        model_class (type[nn.Module]): The PyTorch model class to train.
        device (Union[torch.device, str]): The device ('cpu', 'cuda', torch.device).
        logger_mode (Optional[NNHandler.LoggingMode]): Logging configuration.
        logger_filename (str): Filename for file logging.
        logger_level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        save_model_code (bool): Attempt to save model source code (use with caution).
        model_type (NNHandler.ModelType): Type of model (influences training loop logic).
        **model_kwargs: Keyword arguments passed to the model constructor.
    """

    class Model_Type(Enum):
        CLASSIFICATION = "classification"
        GENERATIVE = "generative"  # e.g., Autoencoders, VAEs, GANs (where input is reconstructed/generated)
        REGRESSION = "regression"  # Added for clarity
        SCORE_BASED = "score_based"

    class LoggingMode(Enum):
        CONSOLE = "console"
        FILE = "file"
        BOTH = "both"

    # --- Core Attributes ---
    _optimizer: Optional[torch.optim.Optimizer] = None
    _optimizer_kwargs: Dict[str, Any] = {}

    _scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    _scheduler_kwargs: Dict[str, Any] = {}

    _loss_fn: Optional[Callable] = None
    _loss_fn_kwargs: Dict[str, Any] = {}
    _pass_epoch_to_loss: bool = False

    _train_loader: Optional[DataLoader] = None
    _train_loader_kwargs: Dict[str, Any] = {}

    _val_loader: Optional[DataLoader] = None
    _val_loader_kwargs: Dict[str, Any] = {}

    _model: Optional[nn.Module] = None
    _model_class: Optional[type[nn.Module]] = None
    _model_kwargs: Optional[Dict[str, Any]] = None
    _model_type: Model_Type = Model_Type.CLASSIFICATION
    _compiled_model: bool = False

    _sampler: Optional[Sampler] = None
    _sampler_kwargs: Dict[str, Any] = {}

    _sde: Optional[Any] = None  # Assuming SDE class structure
    _sde_kwargs: Dict[str, Any] = {}

    _device: Union[torch.device, str] = "cpu"
    _seed: Optional[int] = None

    _auto_saver: AutoSaver = AutoSaver()
    _ema: Optional[ExponentialMovingAverage] = None  # Added for EMA
    _ema_decay: float = 0.0  # Added for EMA configuration persistence

    _train_losses: List[float] = []
    _val_losses: List[float] = []
    _other_train_losses: List[Any] = []  # Keep for compatibility if loss returns tuple
    _other_val_losses: List[Any] = []  # Keep for compatibility

    # --- New Attributes ---
    _metrics: Dict[str, Callable] = {}  # {metric_name: metric_function(output, target)}
    _train_metrics_history: Dict[str, List[float]] = defaultdict(list)
    _val_metrics_history: Dict[str, List[float]] = defaultdict(list)
    _callbacks: List[Callback] = []
    _stop_training: bool = False  # Flag for early stopping
    _grad_scaler: GradScaler = GradScaler(enabled=False)  # For AMP

    __logger: Optional[logging.Logger] = None

    def __init__(self,
                 model_class: type[nn.Module],
                 device: Union[torch.device, str] = "cpu",
                 logger_mode: Optional[LoggingMode] = None,
                 logger_filename: str = "NNHandler.log",
                 logger_level: int = logging.INFO,  # Default to INFO
                 save_model_code: bool = False,
                 model_type: Model_Type = Model_Type.CLASSIFICATION,
                 **model_kwargs):

        self.device = device  # Use property setter for validation/setup
        self._train_losses = []
        self._val_losses = []
        self._other_train_losses = []
        self._other_val_losses = []
        self._train_metrics_history = defaultdict(list)
        self._val_metrics_history = defaultdict(list)
        self._callbacks = []
        self._stop_training = False
        self._metrics = {}
        self._ema = None
        self._ema_decay = 0.0

        # Initialize logger first
        if logger_mode is not None:
            self.initialize_logger(logger_mode, filename=logger_filename, level=logger_level)

        if self.__logger is not None:
            self.__logger.info(f"Initializing NNHandler with {model_class.__name__} on device '{self._device}'.")
            if self._device == "cuda" and torch.cuda.device_count() > 1:
                self.__logger.info(f"Detected {torch.cuda.device_count()} GPUs. DataParallel will be used.")

        # Initialize model
        self.set_model(model_class=model_class, save_model_code=save_model_code, model_type=model_type, **model_kwargs)

        # Initialize AutoSaver (already done by default)
        self._auto_saver = AutoSaver(save_model_code=save_model_code)

        # Initialize GradScaler (default disabled)
        self._grad_scaler = GradScaler(enabled=False)

    def initialize_logger(self, mode: LoggingMode = LoggingMode.CONSOLE, filename: str = "NNHandler.log",
                          level: int = logging.INFO):
        """Initializes the logger for the NNHandler class."""
        # Prevent adding multiple handlers if called again
        if self.__logger and self.__logger.hasHandlers():
            # Remove existing handlers before adding new ones
            for handler in self.__logger.handlers[:]:
                self.__logger.removeHandler(handler)

        self.__logger = logging.getLogger(f"NNHandler_{self._model_class.__name__ if self._model_class else 'NoModel'}")
        self.__logger.setLevel(level)
        # Avoid propagating logs to root logger if handlers are added here
        self.__logger.propagate = False

        formatter = logging.Formatter(
            "[%(levelname)s | %(asctime)s | %(name)s.%(funcName)s()]  %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'  # Add date format
        )

        if mode in [self.LoggingMode.CONSOLE, self.LoggingMode.BOTH]:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            self.__logger.addHandler(console_handler)

        if mode in [self.LoggingMode.FILE, self.LoggingMode.BOTH]:
            log_dir = os.path.dirname(filename)
            if log_dir:  # Create log directory if it doesn't exist
                os.makedirs(log_dir, exist_ok=True)
            # Use 'a' mode to append if file exists, or remove manually if needed
            file_handler = logging.FileHandler(filename, mode='a')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.__logger.addHandler(file_handler)

        if self.__logger.hasHandlers():
            self.__logger.info(f"Logger initialized (mode: {mode.name}, level: {logging.getLevelName(level)}).")
        else:
            print("WARNING: Logger initialization resulted in no handlers.", file=sys.stderr)
            self.__logger = None  # Disable logger if no handlers added

    # --- Property Setters/Getters (Enhanced/Refined) ---

    @property
    def device(self) -> Union[torch.device, str]:
        return self._device

    @device.setter
    def device(self, value: Union[torch.device, str]):
        """Sets the computation device and moves the model accordingly."""
        resolved_device: torch.device
        if isinstance(value, torch.device):
            resolved_device = value
        elif isinstance(value, str):
            if value == "cuda" and not torch.cuda.is_available():
                warnings.warn("CUDA specified but not available. Falling back to CPU.")
                resolved_device = torch.device("cpu")
            else:
                try:
                    resolved_device = torch.device(value)
                except RuntimeError as e:
                    raise ValueError(f"Invalid device string '{value}': {e}") from e
        else:
            raise TypeError(f"Device must be a string or torch.device, got {type(value)}.")

        self._device = resolved_device  # Store the resolved torch.device object
        if self._model:
            self._model.to(self._device)
            # Re-wrap with DataParallel if device changed to multi-GPU CUDA
            if self._device.type == 'cuda' and torch.cuda.device_count() > 1 and not isinstance(self._model,
                                                                                                nn.DataParallel):
                self._model = nn.DataParallel(self._model)
                if self.__logger: self.__logger.info("Model wrapped in nn.DataParallel for multi-GPU usage.")
            # Unwrap if device changed away from multi-GPU CUDA
            elif (self._device.type != 'cuda' or torch.cuda.device_count() <= 1) and isinstance(self._model,
                                                                                                nn.DataParallel):
                self._model = self._model.module  # Get the original model back
                if self.__logger: self.__logger.info("Model unwrapped from nn.DataParallel.")

        if self.__logger:
            self.__logger.info(f"Device set to '{self._device}'.")

    @property
    def seed(self) -> Optional[int]:
        return self._seed

    @seed.setter
    def seed(self, seed_value: Optional[int]):
        if seed_value is not None:
            if not isinstance(seed_value, int):
                message = f"Seed must be an integer or None, got {type(seed_value)}."
                if self.__logger: self.__logger.error(message)
                raise TypeError(message)
            torch.manual_seed(seed_value)
            if self._device.type == 'cuda':
                torch.cuda.manual_seed_all(seed_value)  # Seed all GPUs
            if self.__logger: self.__logger.info(f"Global random seed set to: {seed_value}")
        self._seed = seed_value

    @property
    def model(self) -> Optional[nn.Module]:
        return self._model

    @property
    def model_kwargs(self) -> Optional[Dict[str, Any]]:
        return self._model_kwargs

    @property
    def model_code(self) -> Optional[str]:
        return self._auto_saver.model_code

    # module_code property removed

    def set_model(self, model_class: type[nn.Module], save_model_code: bool = False,
                  model_type: Optional[Model_Type] = None, **model_kwargs):
        """Sets or replaces the model for the handler."""
        if not issubclass(model_class, nn.Module):
            message = f"model_class must be a subclass of torch.nn.Module, got {model_class}."
            if self.__logger: self.__logger.error(message)
            raise TypeError(message)

        if model_type is not None:
            if issubclass(model_type.__class__, Enum) and not isinstance(model_type, self.Model_Type):
                model_type = model_type.value
            if isinstance(model_type, str):
                for internal_model_type in self.Model_Type:
                    if internal_model_type.value == model_type:
                        model_type = internal_model_type
            if not isinstance(model_type, self.Model_Type):
                raise TypeError(f"model_type must be an instance of NNHandler.ModelType, got {type(model_type)}")
            self._model_type = model_type

        self._model_class = model_class
        self._model_kwargs = model_kwargs
        self._model = model_class(**model_kwargs).to(self._device)  # Move to device immediately
        self._compiled_model = False  # Reset compiled flag

        # Handle DataParallel if necessary
        if self._device.type == 'cuda' and torch.cuda.device_count() > 1:
            self._model = nn.DataParallel(self._model)
            if self.__logger: self.__logger.info("Model wrapped in nn.DataParallel.")

        # Handle auto-saving code
        self._auto_saver.save_model_code = save_model_code
        if save_model_code:
            self._auto_saver.try_save_model_code(self._model_class, self.__logger)

        # Re-initialize optimizer if it was already set
        if self._optimizer is not None:
            if self.__logger: self.__logger.warning("Model replaced. Re-initializing optimizer with previous settings.")
            self.set_optimizer(self._optimizer.__class__, **self._optimizer_kwargs)
            # Scheduler is handled within set_optimizer

        if self.__logger:
            self.__logger.info(f"Model set to {model_class.__name__} with type {self._model_type.name}.")
            param_count = self.count_parameters()
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
        # Ensure parameters are correctly passed (handle DataParallel)
        params_to_optimize = self.module.parameters() if isinstance(self._model,
                                                                    nn.DataParallel) else self._model.parameters()
        self._optimizer = optimizer_class(params_to_optimize, **optimizer_kwargs)

        if self.__logger:
            self.__logger.info(f"Optimizer set to {optimizer_class.__name__} with kwargs: {optimizer_kwargs}")
            # self.__logger.debug(f"Optimizer state: {self._optimizer}") # Maybe too verbose

        # Re-initialize scheduler with the new optimizer
        if self._scheduler is not None:
            if self.__logger: self.__logger.info("Re-initializing scheduler with new optimizer.")
            # Ensure scheduler class is stored before overwriting _scheduler
            scheduler_class = self._scheduler.__class__
            self.set_scheduler(scheduler_class, **self._scheduler_kwargs)

    @property
    def scheduler(self) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        return self._scheduler

    def set_scheduler(self, scheduler_class: Optional[type[torch.optim.lr_scheduler.LRScheduler]], **scheduler_kwargs):
        """Sets the learning rate scheduler."""
        if scheduler_class is None:
            self._scheduler = None
            self._scheduler_kwargs = {}
            if self.__logger: self.__logger.info("Scheduler removed.")
            return

        # Check if it's a valid scheduler type (_LRScheduler or ReduceLROnPlateau)
        if not (issubclass(scheduler_class, torch.optim.lr_scheduler._LRScheduler) or
                issubclass(scheduler_class, torch.optim.lr_scheduler.ReduceLROnPlateau) or
                issubclass(scheduler_class, torch.optim.lr_scheduler.LRScheduler)):
            message = (f"scheduler_class {scheduler_class} must be a subclass of "
                       f"torch.optim.lr_scheduler._LRScheduler or ReduceLROnPlateau or LRScheduler.")
            if self.__logger: self.__logger.error(message)
            raise TypeError(message)

        if self._optimizer is None:
            message = "Optimizer must be set before setting the scheduler."
            if self.__logger: self.__logger.error(message)
            raise ValueError(message)

        self._scheduler_kwargs = scheduler_kwargs
        self._scheduler = scheduler_class(self._optimizer, **scheduler_kwargs)
        if self.__logger:
            self.__logger.info(f"Scheduler set to {scheduler_class.__name__} with kwargs: {scheduler_kwargs}")

    # --- SDE and Sampler (similar logic, improved logging/typing) ---
    @property
    def sde(self) -> Optional[Any]:
        return self._sde

    @sde.setter
    def sde(self, sde_instance: Any):
        """Sets the SDE instance directly."""
        # Basic check if it has expected methods (can be improved with ABC)
        if not all(hasattr(sde_instance, attr) for attr in ['prior', 'drift', 'diffusion', 'sigma', 'T', 'epsilon']):
            warnings.warn("Provided SDE instance might be missing expected methods/attributes.", RuntimeWarning)

        if self._model_type != self.Model_Type.SCORE_BASED:
            if self.__logger:
                self.__logger.warning(f"Model Type was {self._model_type.name}. Changed to SCORE_BASED as SDE was set.")
            self._model_type = self.Model_Type.SCORE_BASED

        self._sde = sde_instance
        self._sde_kwargs = {}  # Clear kwargs if instance is set directly
        if self.__logger: self.__logger.info(f"SDE instance set to: {sde_instance}")

    def set_sde(self, sde_class: type, **sde_kwargs):
        """Sets the SDE by providing the class and keyword arguments."""
        # Could add more checks on sde_class if an SDE ABC exists
        if self._model_type != self.Model_Type.SCORE_BASED:
            if self.__logger: self.__logger.warning(
                f"Model Type was {self._model_type.name}. Changed to SCORE_BASED as SDE was set.")
            self._model_type = self.Model_Type.SCORE_BASED

        self._sde = sde_class(**sde_kwargs)
        self._sde_kwargs = sde_kwargs
        if self.__logger: self.__logger.info(f"SDE set to {sde_class.__name__} with kwargs: {sde_kwargs}")

    @property
    def sampler(self) -> Optional[Sampler]:
        return self._sampler

    @sampler.setter
    def sampler(self, sampler_instance: Sampler):
        """Sets the Sampler instance directly."""
        if not isinstance(sampler_instance, Sampler):
            message = "sampler must be an instance of a class inheriting from NNHandler.Sampler."
            if self.__logger: self.__logger.error(message)
            raise TypeError(message)
        self._sampler = sampler_instance
        self._sampler_kwargs = {}  # Clear kwargs
        if self.__logger: self.__logger.info(f"Sampler instance set to: {sampler_instance}")

    def set_sampler(self, sampler_class: type[Sampler], **sampler_kwargs):
        """Sets the Sampler by providing the class and keyword arguments."""
        if not issubclass(sampler_class, Sampler):
            message = f"sampler_class {sampler_class} must be a subclass of NNHandler.Sampler."
            if self.__logger: self.__logger.error(message)
            raise TypeError(message)
        self._sampler_kwargs = sampler_kwargs
        self._sampler = sampler_class(**sampler_kwargs)
        if self.__logger: self.__logger.info(f"Sampler set to {sampler_class.__name__} with kwargs: {sampler_kwargs}")

    def get_samples(self, N, device=None):
        if self._sampler is None:
            raise RuntimeError(f"Sampler has not been set.")
        return self._sampler.sample(N, device=self.device if device is None else device)

    # --- Loss Function ---
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
        if self.__logger: self.__logger.info(f"Loss function set to {loss_function.__name__}.")

    def set_loss_fn(self, loss_fn: Callable, pass_epoch_to_loss: bool = False, **kwargs):
        """Sets the loss function with optional kwargs and epoch passing flag."""
        if not callable(loss_fn):
            message = "loss_fn must be a callable function."
            if self.__logger: self.__logger.error(message)
            raise TypeError(message)
        self._loss_fn = loss_fn
        self._pass_epoch_to_loss = pass_epoch_to_loss
        self._loss_fn_kwargs = kwargs or {}
        if self.__logger:
            self.__logger.info(f"Loss function set to {loss_fn.__name__} with kwargs: {self._loss_fn_kwargs}.")
            if pass_epoch_to_loss:
                self.__logger.info("Current epoch will be passed to the loss function.")

    @property
    def pass_epoch_to_loss(self) -> bool:
        return self._pass_epoch_to_loss

    @pass_epoch_to_loss.setter
    def pass_epoch_to_loss(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("pass_epoch_to_loss must be a boolean.")
        self._pass_epoch_to_loss = value
        if self.__logger: self.__logger.info(f"pass_epoch_to_loss set to {value}.")

    @property
    def loss_fn_kwargs(self) -> Dict[str, Any]:
        return self._loss_fn_kwargs

    @loss_fn_kwargs.setter
    def loss_fn_kwargs(self, value: Dict[str, Any]):
        if not isinstance(value, dict):
            raise TypeError("loss_fn_kwargs must be a dictionary.")
        self._loss_fn_kwargs = value or {}
        if self.__logger: self.__logger.info(f"Loss function kwargs updated to: {self._loss_fn_kwargs}")

    # --- Data Loaders ---
    @property
    def train_loader(self) -> Optional[DataLoader]:
        return self._train_loader

    @property
    def train_loader_kwargs(self) -> Dict[str, Any]:
        return self._train_loader_kwargs

    def set_train_loader(self, dataset: Dataset, **loader_kwargs):
        """Sets the training data loader."""
        if not isinstance(dataset, Dataset):
            raise TypeError(f"dataset must be an instance of torch.utils.data.Dataset, got {type(dataset)}")
        self._train_loader_kwargs = loader_kwargs
        self._train_loader = DataLoader(dataset, **loader_kwargs)
        if self.__logger: self.__logger.info(
            f"Training DataLoader set with dataset {type(dataset).__name__} and kwargs: {loader_kwargs}")

    @property
    def val_loader(self) -> Optional[DataLoader]:
        return self._val_loader

    @property
    def val_loader_kwargs(self) -> Dict[str, Any]:
        return self._val_loader_kwargs

    def set_val_loader(self, dataset: Dataset, **loader_kwargs):
        """Sets the validation data loader."""
        if not isinstance(dataset, Dataset):
            raise TypeError(f"dataset must be an instance of torch.utils.data.Dataset, got {type(dataset)}")
        self._val_loader_kwargs = loader_kwargs
        self._val_loader = DataLoader(dataset, **loader_kwargs)
        if self.__logger: self.__logger.info(
            f"Validation DataLoader set with dataset {type(dataset).__name__} and kwargs: {loader_kwargs}")

    # --- Metrics ---
    @property
    def metrics(self) -> Dict[str, Callable]:
        return self._metrics

    def add_metric(self, name: str, metric_fn: Callable):
        """Adds a metric function to be tracked during training and validation.

        Args:
            name (str): The name of the metric (e.g., 'accuracy').
            metric_fn (Callable): A function that takes (output, target) and returns a scalar value.
                                   It should handle tensors on the correct device.
        """
        if not callable(metric_fn):
            raise TypeError("metric_fn must be callable.")
        if not isinstance(name, str) or not name:
            raise ValueError("Metric name must be a non-empty string.")
        self._metrics[name] = metric_fn
        # Initialize history lists
        self._train_metrics_history[name] = []
        self._val_metrics_history[name] = []
        if self.__logger: self.__logger.info(f"Added metric '{name}'.")

    def clear_metrics(self):
        """Removes all tracked metrics."""
        self._metrics.clear()
        self._train_metrics_history.clear()
        self._val_metrics_history.clear()
        if self.__logger: self.__logger.info("All metrics cleared.")

    @property
    def train_losses(self) -> List[float]:
        return self._train_losses

    @property
    def val_losses(self) -> List[float]:
        return self._val_losses

    @property
    def train_metrics_history(self) -> Dict[str, List[float]]:
        return dict(self._train_metrics_history)  # Return copy

    @property
    def val_metrics_history(self) -> Dict[str, List[float]]:
        return dict(self._val_metrics_history)  # Return copy

    # --- Auto Saving (delegated to AutoSaver) ---
    @property
    def save_interval(self) -> Optional[int]:
        return self._auto_saver.save_interval

    @save_interval.setter
    def save_interval(self, interval: Optional[int]):
        try:
            self._auto_saver.save_interval = interval
            if self.__logger: self.__logger.info(f"Auto-save interval set to {interval} epochs.")
        except (TypeError, ValueError) as e:
            if self.__logger: self.__logger.error(f"Failed to set save_interval: {e}")
            raise e

    @property
    def save_path(self) -> Optional[str]:
        return self._auto_saver.save_path

    @save_path.setter
    def save_path(self, path: Optional[str]):
        try:
            if path is not None and not os.path.isdir(path):
                # Attempt to create the directory
                try:
                    os.makedirs(path, exist_ok=True)
                    if self.__logger: self.__logger.info(f"Created save directory: {path}")
                except OSError as e:
                    message = f"Save path '{path}' is not a valid directory and could not be created: {e}"
                    if self.__logger: self.__logger.error(message)
                    raise ValueError(message) from e

            self._auto_saver.save_path = path
            if self.__logger: self.__logger.info(f"Auto-save path set to '{path}'.")
        except TypeError as e:
            if self.__logger: self.__logger.error(f"Failed to set save_path: {e}")
            raise e

    @property
    def save_model_name(self) -> str:
        return self._auto_saver.save_model_name

    @save_model_name.setter
    def save_model_name(self, name: str):
        try:
            self._auto_saver.save_model_name = name
            if self.__logger: self.__logger.info(f"Auto-save model name format set to '{name}'.")
        except TypeError as e:
            if self.__logger: self.__logger.error(f"Failed to set save_model_name: {e}")
            raise e

    @property
    def overwrite_last_saved(self) -> bool:
        return self._auto_saver.overwrite_last_saved

    @overwrite_last_saved.setter
    def overwrite_last_saved(self, overwrite: bool):
        try:
            self._auto_saver.overwrite_last_saved = overwrite
            if self.__logger: self.__logger.info(f"Auto-save overwrite set to {overwrite}.")
        except TypeError as e:
            if self.__logger: self.__logger.error(f"Failed to set overwrite_last_saved: {e}")
            raise e

    def auto_save(self, interval: Optional[int], save_path: str = '.', name: str = "model_state_epoch",
                  overwrite: bool = False):
        """Configures periodic model saving. Use interval=None or 0 to disable."""
        try:
            self.save_interval = interval
            self.save_path = save_path
            self.save_model_name = name
            self.overwrite_last_saved = overwrite
            if self.__logger:
                if interval is None or interval == 0:
                    self.__logger.info("Auto-save disabled.")
                else:
                    self.__logger.info(
                        f"Auto-save configured: interval={interval}, path='{save_path}', name='{name}', overwrite={overwrite}")
        except (TypeError, ValueError) as e:
            if self.__logger: self.__logger.error(f"Failed to configure auto_save: {e}")
            # Don't raise here, just log the error

    # --- Callbacks ---
    @property
    def callbacks(self) -> List[Callback]:
        return self._callbacks

    def add_callback(self, callback: Callback):
        """Adds a callback to the handler."""
        if not isinstance(callback, Callback):
            raise TypeError("callback must be an instance of Callback.")
        callback.set_handler(self)  # Link handler to callback
        self._callbacks.append(callback)
        if self.__logger: self.__logger.info(f"Added callback: {type(callback).__name__}")

    def _run_callbacks(self, method_name: str, *args, **kwargs):
        """Helper to run a specific method on all callbacks."""
        for callback in self._callbacks:
            method = getattr(callback, method_name, None)
            if callable(method):
                try:
                    method(*args, **kwargs)
                except Exception as e:
                    if self.__logger:
                        self.__logger.error(f"Error in callback {type(callback).__name__}.{method_name}: {e}",
                                            exc_info=True)
                    else:
                        print(f"ERROR in callback {type(callback).__name__}.{method_name}: {e}", file=sys.stderr)
                        import traceback
                        traceback.print_exc()

    # --- Training Loop ---

    def _prepare_batch(self, batch: Any) -> dict[str, Any | None]:
        """
        Processes a given batch of data and prepares it for the model's inference or training. The preparation
        is tailored to the specific model type (e.g., classification, regression, generative, or score-based)
        and its input signature. This includes extracting inputs, targets, and any additional parameters
        required by the model's forward method.

        Args:
            batch: The batch of data to be processed. It could be a single tensor or a sequence (e.g., list
                or tuple) containing inputs, targets, and optionally additional input elements for the
                model's forward call.

        Returns:
            dict: A dictionary containing the processed batch elements. This dictionary includes the following
                keys:
                - "inputs": The input tensor prepared for the model.
                - "targets": The target tensor if applicable (e.g., for classification or regression models),
                  or `None` otherwise.
                - Additional keys for any extra parameters required by the model's forward call, inferred
                  from the model's input signature.

        Raises:
            RuntimeError: If the `_model` attribute is not set when this method is called.
            ValueError: If the batch is not formatted correctly or does not meet the requirements for the
                specified `_model_type`.
        """
        inputs: Any
        targets: Optional[Any] = None
        additional_params = {}

        if self._model is not None:
            # Infer valid parameters from `_model`
            model_sig = inspect.signature(self._model.forward).parameters.keys()
            if len(model_sig) > 1:
                valid_params = list(model_sig)[1:]
            else:
                valid_params = None
        else:
            raise RuntimeError("Model is not set.")

        if self._model_type in [self.Model_Type.CLASSIFICATION, self.Model_Type.REGRESSION]:
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                raise ValueError(
                    f"{self._model_type.name} expects batch to be a sequence (inputs, targets). Got: {type(batch)}")
            inputs = batch[0].to(self._device, non_blocking=True)
            targets = batch[1].to(self._device, non_blocking=True)

            if len(batch) > 2:
                if valid_params is None:
                    raise ValueError(
                        "Invalid model signature. Expected more than 1 element in the forward call of the model since "
                        "the dataloader returns batches of more than 2 elements (i.e. inputs, targets, ...).")
                if len(valid_params) != len(batch) - 2:
                    raise ValueError(
                        f"Got {len(valid_params)} additional parameters in the forward call of the model, but only "
                        f"{len(batch) - 2} batched data are given.")
                extra_from_batch = {f"{valid_params[i]}": item for i, item in enumerate(batch[2:], start=0)}
                additional_params.update(extra_from_batch)

        elif self._model_type == self.Model_Type.GENERATIVE:
            # Assumes input is also the target (e.g., autoencoder)
            if isinstance(batch, (list, tuple)):
                inputs = batch[0].to(self._device, non_blocking=True)
                targets = inputs  # Target is the input itself

                if len(batch) > 1:
                    if valid_params is None:
                        raise ValueError(
                            "Invalid model signature. Expected more than 1 element in the forward call of the model since "
                            "the dataloader returns batches of more than 1 elements (i.e. inputs, ...).")
                    if len(valid_params) != len(batch) - 1:
                        raise ValueError(
                            f"Got {len(valid_params)} additional parameters in the forward call of the model, but only "
                            f"{len(batch) - 1} batched data are given.")
                    extra_from_batch = {f"{valid_params[i]}": item for i, item in enumerate(batch[1:], start=0)}
                    additional_params.update(extra_from_batch)

            else:
                inputs = batch.to(self._device, non_blocking=True)
                targets = inputs

        elif self._model_type == self.Model_Type.SCORE_BASED:
            # Score-based models typically only need the data points
            if isinstance(batch, (list, tuple)):
                inputs = batch[0].to(self._device, non_blocking=True)
            else:
                inputs = batch.to(self._device, non_blocking=True)
            targets = None  # Loss function handles target generation internally
        else:
            raise ValueError(f"Unsupported ModelType: {self._model_type}")

        return {"inputs": inputs, "targets": targets, **additional_params}

    def _calculate_loss(self, model_output: Any, targets: Optional[Any], inputs: Any, current_epoch: int) -> Union[
        Tensor, Tuple[Tensor, List[Tensor]]]:
        """Calculates the loss based on ModelType."""
        if self._loss_fn is None:
            raise RuntimeError("Loss function not set.")

        loss_args = []
        loss_kwargs = self._loss_fn_kwargs

        if self._model_type in [self.Model_Type.CLASSIFICATION, self.Model_Type.REGRESSION]:
            if targets is None:
                raise RuntimeError(f"{self._model_type.name} requires targets for loss calculation.")
            loss_args = [model_output, targets]
        elif self._model_type == self.Model_Type.GENERATIVE:
            if targets is None:  # Should have been set in _prepare_batch
                raise RuntimeError("Generative model requires targets (usually inputs) for loss calculation.")
            loss_args = [model_output, targets]  # Often compares output to original input
        elif self._model_type == self.Model_Type.SCORE_BASED:
            if self._sde is None:
                raise RuntimeError("Score-based model requires an SDE to be set for loss calculation.")
            # Loss function signature assumed: loss(data, sde, model, device, epoch?, **kwargs)
            loss_args = [inputs, self._sde, self.module, self._device]  # Pass unwrapped model if DataParallel
        else:
            raise ValueError(f"Unsupported ModelType for loss: {self._model_type}")

        # Add epoch if required
        if self._pass_epoch_to_loss:
            # Check if epoch is already expected as a kwarg or pos arg (basic check)
            sig = inspect.signature(self._loss_fn)
            takes_epoch = 'epoch' in sig.parameters or any(p.kind == p.VAR_POSITIONAL for p in sig.parameters.values())
            if takes_epoch:
                # Decide whether to pass as pos arg or kwarg (simplistic: prefer kwarg if named 'epoch')
                if 'epoch' in sig.parameters:
                    loss_kwargs = {**loss_kwargs, 'epoch': current_epoch}
                else:
                    # Add epoch as the last positional argument if not explicitly named
                    loss_args.append(current_epoch)
            else:
                if self.__logger: self.__logger.warning(
                    f"Loss function {self._loss_fn.__name__} set with pass_epoch_to_loss=True, but signature doesn't seem to accept 'epoch'.")
                # Still pass it, maybe handled by **kwargs in loss_fn
                loss_kwargs = {**loss_kwargs, 'epoch': current_epoch}

        # Call the loss function
        loss = self._loss_fn(*loss_args, **loss_kwargs)
        return loss

    def _calculate_metrics(self, model_output: Any, targets: Optional[Any]) -> Dict[str, float]:
        """Calculates all defined metrics."""
        batch_metrics = {}
        if not self._metrics:
            return batch_metrics
        if targets is None and self._model_type in [self.Model_Type.CLASSIFICATION, self.Model_Type.REGRESSION]:
            # Cannot calculate metrics requiring targets if targets are None
            return batch_metrics

        with torch.no_grad():  # Ensure metrics don't track gradients
            for name, metric_fn in self._metrics.items():
                try:
                    # Assume metric_fn takes (output, target)
                    value = metric_fn(model_output, targets)
                    if isinstance(value, torch.Tensor):
                        batch_metrics[name] = value.item()
                    else:
                        batch_metrics[name] = float(value)  # Ensure float
                except Exception as e:
                    if self.__logger: self.__logger.error(f"Error calculating metric '{name}': {e}")
                    batch_metrics[name] = math.nan  # Indicate error
        return batch_metrics

    def _train_step(self, batch: Any, current_epoch: int, accumulation_steps: int = 1, use_amp: bool = False) \
            -> tuple[int | float, list[Any] | None, dict[str, float] | dict[str, Any]]:
        """Performs a single training step (forward, loss, backward, metrics)."""
        if self._model is None or self._optimizer is None or self._loss_fn is None:
            raise RuntimeError("Model, optimizer, and loss function must be set for training.")

        self._model.train()  # Ensure model is in training mode

        batch_data = self._prepare_batch(batch)
        # Extract inputs, targets, and additional parameters
        inputs = batch_data["inputs"]
        targets = batch_data["targets"]
        additional_params = {k: v for k, v in batch_data.items() if k not in ["inputs", "targets"]}

        batch_metrics = {}
        batch_loss = torch.tensor(0.0, device=self._device)

        # Mixed Precision Context
        with autocast(device_type=self._device.type, enabled=use_amp):
            # Forward pass
            if self._model_type == self.Model_Type.SCORE_BASED:
                # Score-based loss often calculates score internally, no explicit forward needed here
                # Loss calculation handles the model call
                model_output = None  # Placeholder, not used directly for metrics usually
            else:
                model_output = self._model(inputs, **additional_params)

            # Loss calculation
            loss_val = self._calculate_loss(model_output, targets, inputs, current_epoch)

            # Handle tuple loss (optional secondary losses) - Keep for backward compatibility
            other_losses = None
            if isinstance(loss_val, tuple):
                other_losses = [l.item() for l in loss_val[1:]]  # Store items
                loss = loss_val[0]
            else:
                loss = loss_val

            # Check for NaNs
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                warnings.warn("NaN or Inf detected in training loss. Skipping backward pass for this batch.",
                              RuntimeWarning)
                # Return NaN loss and empty metrics to indicate skip
                return math.nan, {}, {}

            # Normalize loss for gradient accumulation
            loss = loss / accumulation_steps

        # Backward pass & Gradient Scaling (if AMP)
        self._grad_scaler.scale(loss).backward()

        # Metrics calculation (after forward pass, before optimizer step)
        # Use model_output and targets, handle score-based case where output might be None
        if self._model_type != self.Model_Type.SCORE_BASED and model_output is not None:
            batch_metrics = self._calculate_metrics(model_output, targets)

        # Return loss item and metrics dict
        return loss.item() * accumulation_steps, other_losses, batch_metrics  # Return un-normalized loss

    def _val_step(self, batch: Any, current_epoch: int) \
            -> tuple[int | float | bool, list[Any] | None, dict[str, float] | dict[Any, Any]]:
        """Performs a single validation step (forward, loss, metrics)."""
        if self._model is None or self._loss_fn is None:
            raise RuntimeError("Model and loss function must be set for validation.")

        self._model.eval()  # Ensure model is in evaluation mode

        batch_data = self._prepare_batch(batch)
        # Extract inputs, targets, and additional parameters
        inputs = batch_data["inputs"]
        targets = batch_data["targets"]
        additional_params = {k: v for k, v in batch_data.items() if k not in ["inputs", "targets"]}

        batch_metrics = {}
        batch_loss = torch.tensor(0.0, device=self._device)

        with torch.no_grad():  # No gradients needed for validation
            # Forward pass (similar to train step)
            if self._model_type == self.Model_Type.SCORE_BASED:
                model_output = None  # Loss handles model call
            else:
                model_output = self._model(inputs, **additional_params)

            # Loss calculation
            loss_val = self._calculate_loss(model_output, targets, inputs, current_epoch)
            # Handle tuple loss
            other_losses = None
            if isinstance(loss_val, tuple):
                other_losses = [l.item() for l in loss_val[1:]]  # Store items
                loss = loss_val[0]
            else:
                loss = loss_val

            # Metrics calculation
            if self._model_type != self.Model_Type.SCORE_BASED and model_output is not None:
                batch_metrics = self._calculate_metrics(model_output, targets)

        # Return loss item and metrics dict
        batch_loss_item = loss.item() if not (torch.isnan(loss).any() or torch.isinf(loss).any()) else math.nan
        return batch_loss_item, other_losses, batch_metrics

    def train(self,
              epochs: int,
              validate_every: int = 1,
              gradient_accumulation_steps: int = 1,
              use_amp: bool = False,
              gradient_clipping_norm: Optional[float] = None,
              ema_decay: float = 0.0,
              seed: Optional[int] = None,
              progress_bar: bool = True,
              debug_print_interval: Optional[int] = None,  # Print detailed logs every N batches
              save_on_last_epoch: bool = True,
              epoch_train_and_val_pbar: bool = False):
        """Starts the training process of the PyTorch model.

        Args:
            epochs (int): Total number of epochs to train for.
            validate_every (int): Run validation every N epochs. Set to 0 or None to disable validation.
            gradient_accumulation_steps (int): Number of steps to accumulate gradients over before optimizing.
            use_amp (bool): Enable Automatic Mixed Precision (AMP) training (requires CUDA).
            gradient_clipping_norm (Optional[float]): Max norm for gradient clipping. None to disable.
            ema_decay (float): Decay factor for Exponential Moving Average of model weights. 0 to disable.
            seed (Optional[int]): A specific seed for this training run (overrides class seed if set).
            progress_bar (bool): Display a tqdm progress bar.
            debug_print_interval (Optional[int]): Print loss/metrics every N batches (for debugging).
            save_on_last_epoch (bool): Ensure model is saved via AutoSaver on the final epoch, if AutoSaver is enabled.
            epoch_train_and_val_pbar (bool): Display a tqdm progress bar for training and validation.
        """
        # --- Pre-Training Checks ---
        if self._model is None or self._optimizer is None or self._train_loader is None or self._loss_fn is None:
            message = "Model, optimizer, training loader, and loss function must be set before training."
            if self.__logger: self.__logger.error(message)
            raise RuntimeError(message)
        if (validate_every is not None and validate_every > 0) and self._val_loader is None:
            message = "Validation requested (validate_every > 0), but validation loader is not set."
            if self.__logger: self.__logger.error(message)
            raise ValueError(message)
        if use_amp and not _amp_available:
            warnings.warn("AMP requested but torch.amp not available. Disabling AMP.", RuntimeWarning)
            use_amp = False
        if use_amp and self._device.type != 'cuda':
            warnings.warn("AMP requested but device is not CUDA. Disabling AMP.", RuntimeWarning)
            use_amp = False
        if ema_decay > 0 and not _ema_available:
            warnings.warn("EMA requested but torch_ema not available. Disabling EMA.", RuntimeWarning)
            ema_decay = 0.0

        # --- Setup ---
        if seed is not None:  # Apply run-specific seed
            self.seed = seed
        elif self._seed is not None:  # Apply class seed if run-specific not given
            self.seed = self._seed  # Use setter to ensure seeding happens

        start_epoch = len(self._train_losses)  # Resume from last epoch if already trained
        total_epochs = start_epoch + epochs
        if self.__logger:
            self.__logger.info(f"Starting training from epoch {start_epoch + 1} to {total_epochs}.")

        # Initialize EMA if needed
        self._ema = None
        self._ema_decay = ema_decay  # Store decay factor
        if self._ema_decay > 0:
            try:
                # Pass unwrapped model parameters if using DataParallel
                params = self.module.parameters() if isinstance(self._model,
                                                                nn.DataParallel) else self._model.parameters()
                self._ema = ExponentialMovingAverage(params, decay=self._ema_decay)
                # Load EMA state if resuming (handled in load method)
                if self.__logger: self.__logger.info(
                    f"Initialized Exponential Moving Average with decay {self._ema_decay}.")
            except Exception as e:
                if self.__logger: self.__logger.error(f"Failed to initialize EMA: {e}. Disabling EMA.")
                self._ema = None
                self._ema_decay = 0.0

        # Initialize GradScaler for AMP
        self._grad_scaler = GradScaler(enabled=use_amp)
        if use_amp and self.__logger: self.__logger.info("Automatic Mixed Precision (AMP) enabled.")

        # Progress bar setup
        pbar = None
        if progress_bar:
            try:
                from tqdm.auto import tqdm  # Use auto version for better notebook compatibility
                pbar = tqdm(range(start_epoch, total_epochs), desc="Epochs", unit="epoch")
            except ImportError:
                if self.__logger: self.__logger.warning(
                    "tqdm not found. Progress bar disabled. Install with 'pip install tqdm'")
                progress_bar = False  # Disable if import fails

        # --- Callback: on_train_begin ---
        self._stop_training = False  # Reset stop flag
        train_logs = {'start_epoch': start_epoch, 'total_epochs': total_epochs}
        self._run_callbacks('on_train_begin', logs=train_logs)

        # --- Training Loop ---
        for epoch in range(start_epoch, total_epochs):
            epoch_start_time = time.time()
            current_epoch_1_based = epoch + 1
            epoch_logs = {}

            # --- Callback: on_epoch_begin ---
            self._run_callbacks('on_epoch_begin', epoch=epoch, logs=epoch_logs)

            # --- Training Phase ---
            self._model.train()
            train_loss_accum = 0.0
            train_metrics_accum = defaultdict(float)
            processed_batches = 0
            batches_in_epoch = len(self._train_loader)

            train_pbar = None
            if epoch_train_and_val_pbar and pbar:
                train_pbar = tqdm(enumerate(self._train_loader), total=batches_in_epoch,
                                  desc=f"Epoch {current_epoch_1_based} Train", leave=False, unit="batch")
            else:
                train_pbar = enumerate(self._train_loader)

            for batch_idx, batch_data in train_pbar:
                batch_logs = {'batch': batch_idx,
                              'size': len(batch_data[0]) if isinstance(batch_data, (list, tuple)) else len(batch_data)}
                # --- Callback: on_train_batch_begin ---
                self._run_callbacks('on_train_batch_begin', batch=batch_idx, logs=batch_logs)

                # Perform train step (handles forward, loss, backward)
                loss_item, other_losses, metrics_items = self._train_step(batch_data, epoch,
                                                                          gradient_accumulation_steps, use_amp)

                # Accumulate results (handle potential NaN from train_step)
                if not math.isnan(loss_item):
                    train_loss_accum += loss_item
                    for name, value in metrics_items.items():
                        train_metrics_accum[name] += value
                    processed_batches += 1  # Only count successful batches for averaging

                # Optimizer step (potentially includes clipping and EMA update)
                is_accumulation_step = (batch_idx + 1) % gradient_accumulation_steps == 0
                is_last_batch = (batch_idx + 1) == batches_in_epoch

                if is_accumulation_step or is_last_batch:
                    # Unscale gradients before clipping
                    if use_amp:
                        self._grad_scaler.unscale_(self._optimizer)

                    # Gradient Clipping
                    if gradient_clipping_norm is not None:
                        params_to_clip = self.module.parameters() if isinstance(self._model,
                                                                                nn.DataParallel) else self._model.parameters()
                        torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=gradient_clipping_norm)

                    # Optimizer Step (scaled by GradScaler if AMP)
                    self._grad_scaler.step(self._optimizer)

                    # Update GradScaler
                    self._grad_scaler.update()

                    # Zero gradients
                    self._optimizer.zero_grad()

                    # EMA Update (after optimizer step, uses original parameters)
                    if self._ema:
                        self._ema.update()

                batch_logs['loss'] = loss_item
                batch_logs['other_losses'] = other_losses
                batch_logs.update(metrics_items)
                # --- Callback: on_train_batch_end ---
                self._run_callbacks('on_train_batch_end', batch=batch_idx, logs=batch_logs)

                # Debug printing
                if debug_print_interval and (batch_idx + 1) % debug_print_interval == 0:
                    debug_str = f"[Epoch {current_epoch_1_based}/{total_epochs}, Batch {batch_idx + 1}/{batches_in_epoch}] Loss: {loss_item:.4e}"
                    for name, val in metrics_items.items(): debug_str += f", {name}: {val:.4f}"
                    if self.__logger:
                        self.__logger.debug(debug_str)
                    elif progress_bar:
                        train_pbar.set_postfix_str(debug_str, refresh=False)  # Show in inner pbar

            # Calculate average training loss and metrics for the epoch
            avg_train_loss = train_loss_accum / processed_batches if processed_batches > 0 else math.nan
            self._train_losses.append(avg_train_loss)
            epoch_logs['loss'] = avg_train_loss
            for name, total_val in train_metrics_accum.items():
                avg_val = total_val / processed_batches if processed_batches > 0 else math.nan
                self._train_metrics_history[name].append(avg_val)
                epoch_logs[name] = avg_val  # Add train metrics to epoch logs

            # --- Validation Phase ---
            run_validation = (
                    validate_every is not None and validate_every > 0 and current_epoch_1_based % validate_every == 0)
            if run_validation:
                # --- Callback: on_val_begin ---
                self._run_callbacks('on_val_begin', logs=epoch_logs)

                val_loss_accum = 0.0
                val_metrics_accum = defaultdict(float)
                val_processed_batches = 0
                val_batches_in_epoch = len(self._val_loader)

                val_pbar = None
                if epoch_train_and_val_pbar and pbar:
                    val_pbar = tqdm(enumerate(self._val_loader), total=val_batches_in_epoch,
                                    desc=f"Epoch {current_epoch_1_based} Val", leave=False, unit="batch")
                else:
                    val_pbar = enumerate(self._val_loader)

                # Apply EMA weights for validation if enabled
                ema_context = self._ema.average_parameters() if self._ema else contextlib.nullcontext()
                with ema_context:
                    for val_batch_idx, val_batch_data in val_pbar:
                        batch_logs = {'batch': val_batch_idx,
                                      'size': len(val_batch_data[0]) if isinstance(val_batch_data,
                                                                                   (list, tuple)) else len(
                                          val_batch_data)}
                        # --- Callback: on_val_batch_begin ---
                        self._run_callbacks('on_val_batch_begin', batch=val_batch_idx, logs=batch_logs)

                        # Perform validation step
                        loss_item, other_losses, metrics_items = self._val_step(val_batch_data, epoch)

                        # Accumulate results
                        if not math.isnan(loss_item):
                            val_loss_accum += loss_item
                            for name, value in metrics_items.items():
                                val_metrics_accum[name] += value
                            val_processed_batches += 1

                        batch_logs['val_loss'] = loss_item
                        batch_logs['other_val_losses'] = other_losses
                        for name, val in metrics_items.items(): batch_logs[f'val_{name}'] = val
                        # --- Callback: on_val_batch_end ---
                        self._run_callbacks('on_val_batch_end', batch=val_batch_idx, logs=batch_logs)

                # Calculate average validation loss and metrics
                avg_val_loss = val_loss_accum / val_processed_batches if val_processed_batches > 0 else math.nan
                self._val_losses.append(avg_val_loss)
                epoch_logs['val_loss'] = avg_val_loss
                for name, total_val in val_metrics_accum.items():
                    avg_val = total_val / val_processed_batches if val_processed_batches > 0 else math.nan
                    self._val_metrics_history[name].append(avg_val)
                    epoch_logs[f'val_{name}'] = avg_val  # Prefix val metrics

                # --- Callback: on_val_end ---
                self._run_callbacks('on_val_end', logs=epoch_logs)
            else:
                # If no validation this epoch, append NaN to maintain length consistency
                self._val_losses.append(math.nan)
                for name in self._metrics.keys():
                    self._val_metrics_history[name].append(math.nan)

            # --- Scheduler Step ---
            if self._scheduler:
                # Handle ReduceLROnPlateau specifically
                if isinstance(self._scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if run_validation:  # Only step if validation was performed
                        metric_to_monitor = epoch_logs.get('val_loss')  # Default to val_loss
                        # Could add logic to monitor other metrics if needed
                        if metric_to_monitor is not None and not math.isnan(metric_to_monitor):
                            self._scheduler.step(metric_to_monitor)
                            if self.__logger: self.__logger.debug(
                                f"Scheduler ReduceLROnPlateau stepped with metric: {metric_to_monitor:.4e}")
                        else:
                            if self.__logger: self.__logger.warning(
                                "ReduceLROnPlateau scheduler requires valid 'val_loss' (or other metric) to step.")
                    # else: No stepping if no validation
                else:
                    # Standard schedulers step every epoch
                    self._scheduler.step()
                    if self.__logger: self.__logger.debug(f"Scheduler {type(self._scheduler).__name__} stepped.")

            # Log learning rate (handled by LearningRateMonitor callback via epoch_logs)
            if self._optimizer:
                for i, param_group in enumerate(self._optimizer.param_groups):
                    epoch_logs[f'lr_group_{i}'] = param_group['lr']
                    if i == 0: epoch_logs['lr'] = param_group['lr']

            # --- End of Epoch Logging & Progress Bar Update ---
            epoch_time = time.time() - epoch_start_time
            epoch_logs['epoch_time'] = epoch_time
            log_msg = f"Epoch {current_epoch_1_based}/{total_epochs} | Time: {epoch_time:.2f}s | Train Loss: {avg_train_loss:.4e}"
            for name, val in self._train_metrics_history.items():
                log_msg += f" | Train {name}: {val[-1]:.4f}" if len(val) > 0 else ""
            if run_validation:
                log_msg += f" | Val Loss: {avg_val_loss:.4e}"
                for name, val_list in self._val_metrics_history.items():
                    log_msg += f" | Val {name}: {val_list[-1]:.4f}" if len(val_list) > 0 else ""
            if self._optimizer: log_msg += f" | LR: {epoch_logs.get('lr', 'N/A'):.2e}"

            if self.__logger: self.__logger.info(log_msg)
            if progress_bar and pbar: pbar.set_postfix_str(
                log_msg[log_msg.find('|') + 1:].strip())  # Show summary in main pbar

            # --- Callback: on_epoch_end ---
            # This is where ModelCheckpoint, EarlyStopping etc. are triggered
            self._run_callbacks('on_epoch_end', epoch=epoch, logs=epoch_logs)

            # --- Auto Saving ---
            self._auto_save_epoch(epoch, total_epochs, save_on_last_epoch)

            # --- Check for Early Stopping ---
            if self._stop_training:
                if self.__logger: self.__logger.info(f"Training stopped early at epoch {current_epoch_1_based}.")
                break  # Exit the main training loop

            # Update outer progress bar
            if progress_bar and pbar: pbar.update(1)

        # --- End of Training ---
        if progress_bar and pbar: pbar.close()

        # Apply final EMA weights if used
        if self._ema:
            if self.__logger: self.__logger.info("Applying final EMA weights to the model.")
            # Create a temporary copy of original params if needed for saving later
            # original_params = {name: p.data.clone() for name, p in self.module.named_parameters()}
            self._ema.copy_to(
                self.module.parameters() if isinstance(self._model, nn.DataParallel) else self._model.parameters())

        # --- Callback: on_train_end ---
        final_logs = {'final_epoch': epoch + 1}
        final_logs.update(epoch_logs)  # Add logs from the last epoch
        self._run_callbacks('on_train_end', logs=final_logs)

        if self.__logger: self.__logger.info("Training finished.")

    def _auto_save_epoch(self, epoch: int, total_epochs: int, save_on_last_epoch: bool):
        """Handles the logic for auto-saving the model state at the end of an epoch."""
        if self._auto_saver.save_interval is None or self._auto_saver.save_path is None:
            return  # Auto-save disabled

        current_epoch_1_based = epoch + 1
        should_save = False

        # Interval saving
        if self._auto_saver.save_interval > 0 and (current_epoch_1_based % self._auto_saver.save_interval == 0):
            should_save = True

        # Save on last epoch requested
        if save_on_last_epoch and (current_epoch_1_based == total_epochs):
            should_save = True

        # Save if interval is -1 (only last epoch)
        if self._auto_saver.save_interval == -1 and (current_epoch_1_based == total_epochs):
            should_save = True

        if should_save:
            # Use epoch_logs if available, otherwise just use epoch number
            format_dict = {'epoch': current_epoch_1_based}
            # Ideally get logs from callbacks or end of epoch, but might not be available here easily
            # Keep it simple for now: just use epoch number
            try:
                filename = f"{self._auto_saver.save_model_name}.pth".format(**format_dict)
            except KeyError:  # Handle if format string expects metrics not available here
                filename = f"{self._auto_saver.save_model_name}_epoch{current_epoch_1_based}.pth"

            save_path = os.path.join(self._auto_saver.save_path, filename)

            if self.__logger: self.__logger.info(f"Auto-saving model state to {save_path}...")

            # Perform save (inside EMA context if EMA is active, though might be redundant if final EMA applied later)
            ema_context = self._ema.average_parameters() if self._ema else contextlib.nullcontext()
            with ema_context:
                self.save(save_path)  # Save full state

            # Handle overwriting previous auto-save
            if self._auto_saver.overwrite_last_saved and self._auto_saver.last_saved_model:
                if self._auto_saver.last_saved_model != save_path:  # Avoid removing the file just saved
                    try:
                        os.remove(self._auto_saver.last_saved_model)
                        if self.__logger: self.__logger.debug(
                            f"Removed previous auto-saved model: {self._auto_saver.last_saved_model}")
                    except OSError as e:
                        warn_msg = f"Could not remove previous auto-saved model '{self._auto_saver.last_saved_model}': {e}"
                        warnings.warn(warn_msg, RuntimeWarning)
                        if self.__logger: self.__logger.warning(warn_msg)

            self._auto_saver.last_saved_model = save_path

    def monitor_keys(self):
        """
        Returns a non-exhaustive list of key metrics during training.

        This method is designed to provide a set of key metric names
        that are monitored or logged during a training process. These
        keys are typically related to training performance, validation
        checks, and optimization.

        :return: A list of key metric identifiers that include 'loss',
            'val_loss', 'epoch_time', and 'lr'.
        :rtype: list[str]
        """
        return ['loss', 'val_loss', 'epoch_time', 'lr']

    def save(self, path: str):
        """Saves the complete state of the NNHandler to a file.

        Includes model weights, optimizer/scheduler states, loss/metric history,
        EMA state (if used), AutoSaver state, Sampler state (if used), SDE state,
        and callback states.

        Args:
            path (str): The file path to save the state.
        """
        if self._model is None or self._optimizer is None or self._loss_fn is None:
            warnings.warn(
                "Attempting to save handler state, but essential components (model, optimizer, loss_fn) might be missing.",
                RuntimeWarning)

        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        state = {
            # Core Components
            "model_state_dict": self.module.state_dict(),  # Save unwrapped model state
            "optimizer_state_dict": self._optimizer.state_dict() if self._optimizer else None,
            "scheduler_state_dict": self._scheduler.state_dict() if self._scheduler else None,
            "grad_scaler_state_dict": self._grad_scaler.state_dict() if _amp_available and self._grad_scaler else None,
            # Save scaler state if AMP used

            # Configuration
            "model_class": self._model_class,
            "model_kwargs": self._model_kwargs,
            "model_type": self._model_type,
            "optimizer_class": self._optimizer.__class__ if self._optimizer else None,
            "optimizer_kwargs": self._optimizer_kwargs,
            "scheduler_class": self._scheduler.__class__ if self._scheduler else None,
            "scheduler_kwargs": self._scheduler_kwargs,
            "loss_fn": self._loss_fn,  # Note: saving function objects can be tricky
            "loss_fn_kwargs": self._loss_fn_kwargs,
            "pass_epoch_to_loss": self._pass_epoch_to_loss,
            "metrics": self._metrics,  # Save metric functions (names might be better?)
            "seed": self._seed,
            "device": str(self._device),  # Save device as string

            # Data Loader Kwargs (Dataset itself is not saved)
            "train_loader_kwargs": self._train_loader_kwargs,
            "val_loader_kwargs": self._val_loader_kwargs,

            # History
            "train_losses": self._train_losses,
            "val_losses": self._val_losses,
            "train_metrics_history": dict(self._train_metrics_history),  # Convert defaultdict
            "val_metrics_history": dict(self._val_metrics_history),

            # AutoSaver State
            "auto_saver_state": self._auto_saver.state_dict(),

            # EMA State
            "ema_state_dict": self._ema.state_dict() if self._ema else None,
            "ema_decay": self._ema_decay,

            # SDE / Sampler State
            "sde_class": self._sde.__class__ if self._sde else None,
            "sde_kwargs": self._sde_kwargs,
            # Consider adding sde.state_dict() if SDEs have internal state
            "sampler_class": self._sampler.__class__ if self._sampler else None,
            "sampler_kwargs": self._sampler_kwargs,
            "sampler_state_dict": self._sampler.save() if self._sampler else None,  # Use sampler's save method

            # Callback States
            "callback_states": {cb.__class__.__name__: cb.state_dict() for cb in self._callbacks},

            # Versioning (Optional but recommended)
            "nn_handler_version": "2.0",  # Add a version marker
        }

        try:
            torch.save(state, path)
            if self.__logger: self.__logger.info(f"NNHandler state saved successfully to: {path}")
        except Exception as e:
            if self.__logger: self.__logger.error(f"Failed to save NNHandler state to {path}: {e}", exc_info=True)
            raise e  # Re-raise the exception

    @staticmethod
    def load(path: str,
             device: Union[str, torch.device] = "cpu",
             load_from_code: bool = False,
             weights_only: bool = False,  # Note: weights_only=True prevents loading optimizer etc.
             strict_load: bool = False,  # Passed to model.load_state_dict
             # Options to skip loading certain parts
             skip_optimizer: bool = False,
             skip_scheduler: bool = False,
             skip_history: bool = False,
             skip_callbacks: bool = False,
             skip_sampler_sde: bool = False,
             skip_ema: bool = False
             ) -> 'NNHandler':
        """Loads a saved NNHandler state from a file.

        Args:
            path (str): Path to the saved state file.
            device (Union[str, torch.device]): Device to load the model onto.
            load_from_code (bool): If True, attempts to load the model class definition
                from saved source code (requires caution and trust in the source).
            weights_only (bool): Passed to `torch.load`. If True, only loads model weights
                and disables loading of optimizer, scheduler, history, etc.
            strict_load (bool): Whether to strictly enforce that the keys in `state_dict`
                               match the keys returned by this module's `state_dict()` function.
            skip_optimizer (bool): Don't load optimizer state.
            skip_scheduler (bool): Don't load scheduler state.
            skip_history (bool): Don't load loss/metric history.
            skip_callbacks (bool): Don't load callback states (callbacks might still be added later).
            skip_sampler_sde (bool): Don't load sampler/SDE state.
            skip_ema (bool): Don't load EMA state.

        Returns:
            NNHandler: An instance loaded with the saved state.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        # --- Handle Code Loading (if requested) ---
        model_class_from_code = None
        if load_from_code:
            try:
                model_code, module_name = NNHandler.load_model_code(path)  # Use static method
                if model_code and module_name:
                    # Create a temporary module to execute the code in
                    temp_module = types.ModuleType(module_name)
                    # Important: Add necessary imports to the module's dict if not handled by exec
                    # This part is tricky and depends on the saved code structure
                    # For simplicity, assume imports are handled within model_code string for now
                    sys.modules[module_name] = temp_module  # Add to sys.modules temporarily

                    exec(model_code, temp_module.__dict__)

                    # Attempt to retrieve the model class from the temporary module
                    # Need the actual class name, which isn't directly stored in AutoSaver easily
                    # We retrieve it from the saved state *after* loading it
                    # This creates a chicken-and-egg problem.
                    # Workaround: Load state first to get class name, then exec code? Risky.
                    # Better: Assume the class name is needed *before* exec?
                    # Let's load the state first, *then* potentially use the loaded code if needed.
                    print("INFO: Code loading requested. Will load state first, then use code context.")
                    # This approach might fail if the class isn't available globally during torch.load
                else:
                    warnings.warn("load_from_code=True but no code found in saved state.", RuntimeWarning)
                    load_from_code = False  # Disable if code isn't actually there
            except Exception as e:
                warnings.warn(
                    f"Failed to load or execute model code from {path}: {e}. Proceeding without code execution.",
                    RuntimeWarning)
                load_from_code = False

        # --- Load State Dictionary ---
        # Map location ensures tensors are loaded onto the desired device directly
        map_location = torch.device(device)
        try:
            # Set weights_only based on arg, potentially overriding to False if needed
            effective_weights_only = weights_only and skip_optimizer and skip_scheduler and skip_history and skip_callbacks and skip_sampler_sde and skip_ema
            state = torch.load(path, map_location=map_location, weights_only=effective_weights_only)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {path}: {e}") from e

        if effective_weights_only and not weights_only:
            warnings.warn(
                "weights_only=False specified, but all other load components were skipped. Loading effectively weights only.",
                RuntimeWarning)

        # --- Extract Configuration ---
        model_class = state.get("model_class")
        model_kwargs = state.get("model_kwargs", {})
        model_type = state.get("model_type", NNHandler.Model_Type.CLASSIFICATION)  # Default if missing

        if model_class is None:
            raise ValueError("Saved state is missing 'model_class'. Cannot reconstruct handler.")

        # --- Instantiate Handler ---
        # Use dummy values initially, will be overwritten
        handler = NNHandler(model_class, device=device, model_type=model_type, **model_kwargs)
        handler._model = None  # Prevent double initialization in set_model

        # --- Load Model Weights ---
        # Re-create model instance (already done in __init__, but ensure it's fresh)
        handler._model = model_class(**model_kwargs)
        # Handle DataParallel potential mismatch
        model_state_dict = state.get("model_state_dict", {}) if "model_state_dict" in state else state.get("model", {})
        # Check if state dict keys start with 'module.' (saved with DataParallel)
        is_saved_parallel = any(key.startswith('module.') for key in model_state_dict.keys())
        # Check if current model setup uses DataParallel
        is_current_parallel = isinstance(handler._model, nn.DataParallel)

        if is_saved_parallel and not is_current_parallel:
            # Need to remove 'module.' prefix
            new_state_dict = OrderedDict()
            for k, v in model_state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v  # Remove 'module.'
                else:
                    new_state_dict[k] = v  # Keep as is (shouldn't happen in pure DP save)
            model_state_dict = new_state_dict
        elif not is_saved_parallel and is_current_parallel:
            # Need to add 'module.' prefix (might happen if loading single GPU weights to multi-GPU)
            new_state_dict = OrderedDict()
            for k, v in model_state_dict.items():
                new_state_dict['module.' + k] = v
            model_state_dict = new_state_dict
        # Else (both parallel or both not parallel): Use keys as is

        # Load the state dict
        try:
            # If using DP, load into the .module, otherwise directly
            target_model = handler.module if is_current_parallel else handler._model
            missing_keys, unexpected_keys = target_model.load_state_dict(model_state_dict, strict=strict_load)
            if handler.__logger:
                if unexpected_keys: handler.__logger.warning(
                    f"Unexpected keys found in model state_dict: {unexpected_keys}")
                if missing_keys: handler.__logger.warning(f"Missing keys found in model state_dict: {missing_keys}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model state_dict: {e}") from e

        handler._model.to(handler.device)  # Ensure model is on the correct device after loading

        # --- Load Other Components (if not weights_only) ---
        if not effective_weights_only:
            # Optimizer
            opt_class = state.get("optimizer_class")
            opt_kwargs = state.get("optimizer_kwargs", {})
            opt_state = state.get("optimizer_state_dict") if "optimizer_state_dict" in state else state.get("optimizer")
            if not skip_optimizer and opt_class and opt_state:
                try:
                    handler.set_optimizer(opt_class, **opt_kwargs)
                    handler._optimizer.load_state_dict(opt_state)
                    if handler.__logger: handler.__logger.info("Optimizer state loaded.")
                except Exception as e:
                    warnings.warn(f"Failed to load optimizer state: {e}. Optimizer may need re-initialization.",
                                  RuntimeWarning)
                    handler._optimizer = None  # Reset if loading failed

            # Scheduler
            sched_class = state.get("scheduler_class")
            sched_kwargs = state.get("scheduler_kwargs", {})
            sched_state = state.get("scheduler_state_dict") if "scheduler_state_dict" in state else state.get(
                "scheduler")
            if not skip_scheduler and sched_class and sched_state and handler._optimizer:  # Requires optimizer
                try:
                    handler.set_scheduler(sched_class, **sched_kwargs)
                    handler._scheduler.load_state_dict(sched_state)
                    if handler.__logger: handler.__logger.info("Scheduler state loaded.")
                except Exception as e:
                    warnings.warn(f"Failed to load scheduler state: {e}. Scheduler may need re-initialization.",
                                  RuntimeWarning)
                    handler._scheduler = None

            # GradScaler (AMP)
            scaler_state = state.get("grad_scaler_state_dict")
            if _amp_available and scaler_state:
                try:
                    handler._grad_scaler.load_state_dict(scaler_state)
                    if handler.__logger: handler.__logger.info("GradScaler state loaded.")
                except Exception as e:
                    warnings.warn(f"Failed to load GradScaler state: {e}. AMP might behave unexpectedly.",
                                  RuntimeWarning)

            # Loss Function (re-assigning function object)
            handler._loss_fn = state.get("loss_fn")
            handler._loss_fn_kwargs = state.get("loss_fn_kwargs", {})
            handler._pass_epoch_to_loss = state.get("pass_epoch_to_loss", False)
            # Note: Relying on the function object being picklable and available in the loading environment.

            # Metrics (re-assigning function objects)
            handler._metrics = state.get("metrics", {})

            # History
            if not skip_history:
                handler._train_losses = state.get("train_losses", [])
                handler._val_losses = state.get("val_losses", [])
                handler._train_metrics_history = defaultdict(list, state.get("train_metrics_history", {}))
                handler._val_metrics_history = defaultdict(list, state.get("val_metrics_history", {}))
                if handler.__logger: handler.__logger.info("Training history loaded.")

            # AutoSaver State
            auto_saver_state = state.get("auto_saver_state") if "auto_saver_state" in state else state.get(
                "auto_saver_kwargs")
            if auto_saver_state:
                handler._auto_saver.load_state_dict(auto_saver_state)
                # Re-trigger code saving check if needed
                if handler._auto_saver.save_model_code and not handler._auto_saver.model_code:
                    handler._auto_saver.try_save_model_code(handler._model_class, handler.__logger)
                if handler.__logger: handler.__logger.info("AutoSaver state loaded.")

            # EMA State
            handler._ema_decay = state.get("ema_decay", 0.0)
            ema_state = state.get("ema_state_dict")
            if not skip_ema and handler._ema_decay > 0 and ema_state and _ema_available:
                try:
                    params = handler.module.parameters() if isinstance(handler._model,
                                                                       nn.DataParallel) else handler._model.parameters()
                    handler._ema = ExponentialMovingAverage(params, decay=handler._ema_decay)
                    handler._ema.load_state_dict(ema_state)
                    if handler.__logger: handler.__logger.info("EMA state loaded.")
                except Exception as e:
                    warnings.warn(f"Failed to load EMA state: {e}. EMA might need re-initialization.", RuntimeWarning)
                    handler._ema = None
                    handler._ema_decay = 0.0

            # SDE / Sampler
            sde_class = state.get("sde_class")
            sde_kwargs = state.get("sde_kwargs", {})
            sampler_class = state.get("sampler_class")
            sampler_kwargs = state.get("sampler_kwargs", {})
            sampler_state = state.get("sampler_state_dict")
            sampler_state = sampler_state if sampler_state else state.get("sampler_saved")
            if not skip_sampler_sde:
                if sde_class:
                    try:
                        handler.set_sde(sde_class, **sde_kwargs)
                        if handler.__logger: handler.__logger.info("SDE loaded.")
                    except Exception as e:
                        warnings.warn(f"Failed to load SDE: {e}", RuntimeWarning)
                if sampler_class:
                    try:
                        handler.set_sampler(sampler_class, **sampler_kwargs)
                        if sampler_state and handler._sampler:
                            handler._sampler.load(**sampler_state)  # Use sampler's load method
                        if handler.__logger: handler.__logger.info("Sampler loaded.")
                    except Exception as e:
                        warnings.warn(f"Failed to load Sampler: {e}", RuntimeWarning)

            # Callbacks - Recreate instances and load state
            callback_states = state.get("callback_states", {})
            if not skip_callbacks and handler._callbacks:  # Only load state if callbacks are already added
                loaded_cb_names = set()
                for cb in handler._callbacks:
                    cb_name = cb.__class__.__name__
                    if cb_name in callback_states:
                        try:
                            cb.load_state_dict(callback_states[cb_name])
                            loaded_cb_names.add(cb_name)
                            if handler.__logger: handler.__logger.debug(f"Loaded state for callback '{cb_name}'.")
                        except Exception as e:
                            warnings.warn(f"Failed to load state for callback '{cb_name}': {e}", RuntimeWarning)
                # Warn about states present in checkpoint but not matched to a current callback
                unmatched_states = set(callback_states.keys()) - loaded_cb_names
                if unmatched_states:
                    warnings.warn(
                        f"Callback states found in checkpoint but not loaded (no matching callback added): {unmatched_states}",
                        RuntimeWarning)
            elif not skip_callbacks and callback_states:
                warnings.warn(
                    "Callback states found in checkpoint, but no callbacks are currently added to the handler. States were not loaded.",
                    RuntimeWarning)

            # Other config
            handler._seed = state.get("seed")
            handler._train_loader_kwargs = state.get("train_loader_kwargs", {})
            handler._val_loader_kwargs = state.get("val_loader_kwargs", {})

        if handler.__logger:
            handler.__logger.info(f"NNHandler loaded successfully from: {path}")

        return handler

    @staticmethod
    def initialize(**kwargs):
        """Initializes NNHandler using keyword arguments. Deprecated in favor of direct __init__ and setters."""
        warnings.warn("NNHandler.initialize is deprecated. Use direct instantiation and setter methods instead.",
                      DeprecationWarning)
        # Basic translation, might miss newer features
        required_args = ['model_class', 'optimizer_class', 'loss_fn', 'train_data']
        if not all(k in kwargs for k in required_args):
            raise ValueError(f"Missing one of the required arguments for initialize: {required_args}")

        # Pop arguments for constructor
        model_class = kwargs.pop('model_class')
        device = kwargs.pop('device', 'cpu')
        logger_mode = kwargs.pop('logger_mode', None)
        logger_filename = kwargs.pop('logger_filename', 'NNHandler.log')
        logger_level = kwargs.pop('logger_level', logging.INFO)
        save_model_code = kwargs.pop('save_model_code', False)
        model_type = kwargs.pop('model_type', NNHandler.Model_Type.CLASSIFICATION)
        model_kwargs = kwargs.pop('model_kwargs', {})

        handler = NNHandler(model_class, device, logger_mode, logger_filename, logger_level,
                            save_model_code, model_type, **model_kwargs)

        # Set other components
        handler.set_optimizer(kwargs.pop('optimizer_class'), **kwargs.pop('optimizer_kwargs', {}))
        handler.set_scheduler(kwargs.pop('scheduler_class', None), **kwargs.pop('scheduler_kwargs', {}))
        handler.set_train_loader(kwargs.pop('train_data'), **kwargs.pop('train_loader_kwargs', {}))
        handler.set_loss_fn(kwargs.pop('loss_fn'), kwargs.pop('pass_epoch_to_loss', False),
                            **kwargs.pop('loss_fn_kwargs', {}))

        val_data = kwargs.pop('val_data', None)
        if val_data:
            handler.set_val_loader(val_data, **kwargs.pop('val_loader_kwargs', {}))

        sde_class = kwargs.pop('sde_class', None)
        if sde_class:
            handler.set_sde(sde_class, **kwargs.pop('sde_kwargs', {}))

        handler.seed = kwargs.pop('seed', None)

        # Auto save config
        auto_save_interval = kwargs.pop('auto_save_interval', None)
        if auto_save_interval is not None:  # Check if user provided it
            handler.auto_save(auto_save_interval,
                              kwargs.pop('auto_save_path', '..'),
                              kwargs.pop('auto_save_name', 'model_state_epoch'),
                              kwargs.pop('auto_save_overwrite', False))

        if kwargs:  # Check for any unused kwargs
            warnings.warn(f"Unused arguments passed to NNHandler.initialize: {list(kwargs.keys())}", RuntimeWarning)

        return handler

    @staticmethod
    def initialize_from_checkpoint(checkpoint_path: str,
                                   model_class: Optional[type[nn.Module]] = None,  # Make optional
                                   model_type: Optional[Model_Type] = None,  # Make optional
                                   device: Union[str, torch.device] = "cpu",
                                   **model_kwargs) -> 'NNHandler':
        """Loads model weights from a checkpoint file into a new NNHandler instance.

        This is primarily for inference or fine-tuning when you only have the weights
        and need to reconstruct the handler structure.

        Args:
            checkpoint_path (str): Path to the saved checkpoint file. This file should
                                   contain *only* the model's state_dict.
            model_class (Optional[type[nn.Module]]): The model class to instantiate.
                                        Required if not using load_from_code=True during a full load.
            model_type (Optional[NNHandler.Model_Type]): The type of the model.
            device (Union[str, torch.device]): Device to load onto.
            **model_kwargs: Keyword arguments for the model constructor.

        Returns:
            NNHandler: An instance with the loaded weights. Other components (optimizer, etc.)
                       will need to be set manually.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        if model_class is None:
            raise ValueError("model_class must be provided when initializing from a weights-only checkpoint.")

        # Use default model type if not provided
        effective_model_type = model_type if model_type is not None else NNHandler.Model_Type.CLASSIFICATION

        # Instantiate a bare handler
        handler = NNHandler(model_class, device=device, model_type=effective_model_type, **model_kwargs)
        handler._model = None  # Prevent double init

        # Load the state dict (assuming it's just model weights)
        state_dict = torch.load(checkpoint_path, map_location=handler.device)

        # Instantiate the model
        model = model_class(**model_kwargs)

        # Handle DataParallel key mismatch (same logic as in load)
        is_saved_parallel = any(key.startswith('module.') for key in state_dict.keys())
        is_current_parallel = handler.device.type == 'cuda' and torch.cuda.device_count() > 1

        if is_saved_parallel and not is_current_parallel:
            state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items() if k.startswith('module.'))
        elif not is_saved_parallel and is_current_parallel:
            state_dict = OrderedDict(('module.' + k, v) for k, v in state_dict.items())
            model = nn.DataParallel(model)  # Wrap the model if loading onto multi-GPU

        # Load weights
        try:
            target_model = model.module if isinstance(model, nn.DataParallel) else model
            missing_keys, unexpected_keys = target_model.load_state_dict(state_dict, strict=True)  # Be strict here
            if handler.__logger:
                if unexpected_keys: handler.__logger.warning(
                    f"Unexpected keys found in checkpoint state_dict: {unexpected_keys}")
                if missing_keys: handler.__logger.warning(
                    f"Missing keys found in checkpoint state_dict: {missing_keys}")
        except Exception as e:
            raise RuntimeError(f"Failed to load weights from checkpoint {checkpoint_path}: {e}") from e

        handler._model = model.to(handler.device)  # Assign loaded model to handler

        if handler.__logger:
            handler.__logger.info(f"Model weights loaded from checkpoint: {checkpoint_path}")
            handler.__logger.warning(
                "Initialized from checkpoint: Optimizer, scheduler, history, etc. are NOT loaded and need manual setup.")

        return handler

    def __str__(self) -> str:
        """Provides a string representation of the handler's current state."""
        return self.print(False)

    def print(self, show_model_structure=False):
        """
        Prints a detailed status summary of the neural network handler, including information
        about the model, optimizer, scheduler, loss function, metrics, callbacks, and additional
        session-related configurations. Optionally, includes the full model structure.

        Args:
            show_model_structure (bool, optional): Whether to include the detailed structure
                of the model in the output. Defaults to False.

        Returns:
            str: A formatted string containing the detailed neural network handler status.
        """
        try:
            model_str = str(self.module) if self._model else "None"
        except Exception:
            model_str = "Error retrieving model string"
        optimizer_str = str(self._optimizer.__class__.__name__) if self._optimizer else "None"
        scheduler_str = str(self._scheduler.__class__.__name__) if self._scheduler else "None"
        loss_fn_str = self._loss_fn.__name__ if self._loss_fn else "None"
        num_params = f"{self.count_parameters():,}" if self._model else "N/A"
        num_epochs = len(self._train_losses)
        metrics_str = ", ".join(self._metrics.keys()) if self._metrics else "None"
        callbacks_str = ", ".join(cb.__class__.__name__ for cb in self._callbacks) if self._callbacks else "None"

        repr_str = f"NNHandler Status:\n"
        repr_str += f"  Model Class:      {self._model_class.__name__ if self._model_class else 'None'}\n"
        repr_str += f"  Model Type:       {self._model_type.name}\n"
        repr_str += f"  Trainable Params: {num_params}\n"
        repr_str += f"  Device:           {str(self._device)}\n"
        repr_str += f"  Trained Epochs:   {num_epochs}\n"
        repr_str += f"  Optimizer:        {optimizer_str}\n"
        repr_str += f"  Scheduler:        {scheduler_str}\n"
        repr_str += f"  Loss Function:    {loss_fn_str}\n"
        repr_str += f"  Metrics:          {metrics_str}\n"
        repr_str += f"  Callbacks:        {callbacks_str}\n"
        repr_str += f"  Train Loader:     {'Set' if self._train_loader else 'Not Set'}\n"
        repr_str += f"  Val Loader:       {'Set' if self._val_loader else 'Not Set'}\n"
        repr_str += f"  Auto Saving:      {'Enabled' if self._auto_saver.save_interval is not None else 'Disabled'}"
        if self._auto_saver.save_interval is not None:
            repr_str += f" (Interval: {self._auto_saver.save_interval}, Path: '{self._auto_saver.save_path}', Model Name:'{self._auto_saver.save_model_name}')"
        repr_str += f"\n  EMA Enabled:      {self._ema is not None} (Decay: {self._ema_decay if self._ema else 'N/A'})\n"
        repr_str += f"  AMP Enabled:      {self._grad_scaler.is_enabled() if _amp_available else False}\n"

        if show_model_structure:
            repr_str += f"  Model Structure:\n    {model_str.replace('n', 'n    ')}\n"  # Often too long

        return repr_str

    # --- Model Interaction ---
    @property
    def module(self) -> nn.Module:
        """Returns the underlying model, unwrapping DataParallel if necessary."""
        if self._model is None:
            raise RuntimeError("Model has not been set.")
        if isinstance(self._model, nn.DataParallel):
            return self._model.module
        return self._model

    def __call__(self, *args, **kwargs, ) -> Any:
        """Performs a forward pass using the underlying model."""
        if self._model is None:
            raise RuntimeError("Model has not been set.")
        # Special handling for score-based models might be needed if __call__ isn't just forward
        if self._model_type == self.Model_Type.SCORE_BASED:
            # Default to score method
            return self.score(*args, **kwargs)
        else:
            return self._model(*args, **kwargs)

    @torch.no_grad()
    def predict(self, data_loader: DataLoader, apply_ema: bool = True) -> List[Any]:
        """Performs inference on a given DataLoader and returns predictions.

        Args:
            data_loader (DataLoader): DataLoader containing the data to predict on.
            apply_ema (bool): If True and EMA is enabled, use EMA weights for prediction.

        Returns:
            List[Any]: A list containing the outputs for each batch.
                       The structure depends on the model's output.
        """
        if self._model is None:
            raise RuntimeError("Model has not been set.")

        self._model.eval()
        predictions = []
        pbar = None
        try:
            from tqdm.auto import tqdm
            pbar = tqdm(data_loader, desc="Predicting", leave=False)
        except ImportError:
            pbar = data_loader  # Iterate without progress bar

        # Apply EMA weights for prediction if enabled
        ema_context = self._ema.average_parameters() if (self._ema and apply_ema) else contextlib.nullcontext()

        with ema_context:
            for batch_data in pbar:
                # Prepare batch - assuming prediction only needs inputs
                if isinstance(batch_data, (list, tuple)):
                    inputs = batch_data[0].to(self._device, non_blocking=True)
                else:
                    inputs = batch_data.to(self._device, non_blocking=True)

                # Forward pass
                # Handle score-based __call__ if necessary, or just model forward
                # Assuming __call__ does the right thing (e.g., model forward pass)
                outputs = self(inputs)  # Use handler's __call__

                # Move outputs to CPU and detach before appending (optional, saves GPU memory)
                if isinstance(outputs, torch.Tensor):
                    predictions.append(outputs.cpu().detach())
                elif isinstance(outputs, (list, tuple)):  # Handle multiple outputs
                    predictions.append([o.cpu().detach() if isinstance(o, torch.Tensor) else o for o in outputs])
                else:
                    predictions.append(outputs)  # Append as is if not tensor/tuple

        self._model.train()  # Set back to train mode potentially?
        return predictions

    def eval(self, activate: bool = True):
        """Sets the model to evaluation or training mode."""
        if self._model is None: return
        if activate:
            self._model.eval()
            if self.__logger: self.__logger.info("Model set to evaluation mode.")
        else:
            self._model.train()
            if self.__logger: self.__logger.info("Model set to training mode.")

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Counts model parameters."""
        if self._model is None: return 0
        model_to_count = self.module  # Count parameters of the underlying module
        if trainable_only:
            return sum(p.numel() for p in model_to_count.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in model_to_count.parameters())

    def plot_losses(self, log_y_scale: bool = False, save_path: Optional[str] = None):
        """Plots training and validation losses."""
        if not _matplotlib_available:
            warnings.warn("Matplotlib not found. Cannot plot losses. Install with 'pip install matplotlib'",
                          RuntimeWarning)
            return

        epochs = range(1, len(self.train_losses) + 1)
        plt.figure()
        plt.plot(epochs, self.train_losses, label='Training Loss', marker='o', linestyle='-', ms=2, linewidth=0.5)

        # Plot validation loss only if data exists and is not all NaN
        valid_val_losses = [l for l in self.val_losses if not math.isnan(l)]
        if valid_val_losses:
            val_epochs = [e for e, l in zip(epochs, self.val_losses) if not math.isnan(l)]
            plt.plot(val_epochs, valid_val_losses, label='Validation Loss', marker='x', linestyle='--', ms=2,
                     linewidth=0.5)

        if log_y_scale:
            plt.yscale("log")

        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(frameon=False)
        plt.grid(True, which='both', linestyle='--', linewidth=0.25)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            if self.__logger: self.__logger.info(f"Loss plot saved to {save_path}")
            plt.close()  # Close plot if saving to file
        else:
            plt.show()

    def plot_metrics(self, log_y_scale: bool = False, save_path_prefix: Optional[str] = None):
        """Plots training and validation metrics."""
        if not _matplotlib_available:
            warnings.warn("Matplotlib not found. Cannot plot metrics. Install with 'pip install matplotlib'",
                          RuntimeWarning)
            return
        if not self._metrics:
            print("No metrics configured to plot.")
            return

        epochs = range(1, len(self.train_losses) + 1)  # Use loss length as reference for epochs

        for name in self._metrics.keys():
            plt.figure()
            train_metric = self._train_metrics_history.get(name, [])
            val_metric = self._val_metrics_history.get(name, [])

            if train_metric:
                # Ensure length matches epoch count (might be shorter if resumed)
                train_epochs = epochs[:len(train_metric)]
                plt.plot(train_epochs, train_metric, label=f'Train {name}', marker='o', linestyle='-')

            valid_val_metrics = [m for m in val_metric if not math.isnan(m)]
            if valid_val_metrics:
                val_epochs = [e for e, m in zip(epochs, val_metric) if not math.isnan(m)]
                # Ensure length matches epoch count
                val_epochs_plot = val_epochs[:len(valid_val_metrics)]
                plt.plot(val_epochs_plot, valid_val_metrics, label=f'Validation {name}', marker='x', linestyle='--')

            if log_y_scale:
                plt.yscale("log")

            plt.title(f"Training and Validation Metric: {name}")
            plt.xlabel("Epoch")
            plt.ylabel(name)
            plt.legend(frameon=False)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)

            if save_path_prefix:
                filepath = f"{save_path_prefix}_metric_{name}.png"
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                plt.savefig(filepath)
                if self.__logger: self.__logger.info(f"Metric plot for '{name}' saved to {filepath}")
                plt.close()  # Close plot if saving
            else:
                plt.show()

    def compile_model(self, **kwargs):
        """Compiles the model using torch.compile."""
        if not hasattr(torch, 'compile'):
            warnings.warn("torch.compile not available in this PyTorch version. Skipping compilation.", RuntimeWarning)
            return
        if self._model is None:
            raise RuntimeError("Model must be set before compiling.")
        if self._compiled_model:
            if self.__logger: self.__logger.warning("Model is already compiled. Skipping.")
            return

        try:
            if self.__logger: self.__logger.info(f"Compiling model with torch.compile (options: {kwargs})...")
            start_time = time.time()
            # Compile the underlying module if using DataParallel
            target_model = self.module
            compiled_module = torch.compile(target_model, **kwargs)

            # If using DataParallel, replace the original module inside it
            if isinstance(self._model, nn.DataParallel):
                self._model.module = compiled_module
            else:
                self._model = compiled_module

            self._compiled_model = True
            end_time = time.time()
            if self.__logger: self.__logger.info(f"Model compiled successfully in {end_time - start_time:.2f} seconds.")

        except Exception as e:
            if self.__logger: self.__logger.error(f"Failed to compile model: {e}", exc_info=True)
            # Optionally revert model or just log error? For now, just log.
            # self._model = self.module # Revert? Needs care with DP.

    # Set the signature of compile_model to be that of torch.compile if possible.
    if hasattr(torch, 'compile'):
        compile_model.__signature__ = inspect.signature(torch.compile)
    else:  # Provide a default signature if compile isn't available
        compile_model.__signature__ = inspect.Signature(parameters=[
            inspect.Parameter('kwargs', inspect.Parameter.VAR_KEYWORD)
        ])

    # --- Score-Based Model Methods (Mostly Unchanged, ensure device usage) ---
    @staticmethod
    def load_model_code(path: str) -> Tuple[Optional[str], Optional[str]]:
        """Loads model code and module name from a saved state file."""
        from io import BytesIO
        import zipfile
        import pickle

        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        # Define a robust unpickler that tries to ignore missing classes/storages
        class RobustUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # Try to find the class, but return a dummy if not found
                try:
                    return super().find_class(module, name)
                except (AttributeError, ModuleNotFoundError):
                    warnings.warn(f"Could not find class {module}.{name} during unpickling. Replacing with dummy.",
                                  RuntimeWarning)
                    return type(name, (object,), {'__module__': module})  # Return a dummy class

            def persistent_load(self, pid):
                # Handle PyTorch storage types (simplified)
                # This part is highly version-dependent and complex.
                # The goal here is just to extract the non-tensor data.
                if isinstance(pid, tuple) and len(pid) > 0 and pid[0] == 'storage':
                    try:
                        storage_type, key, location, size = pid[1:]
                        # Return a dummy storage-like object or None
                        warnings.warn(f"Ignoring PyTorch storage {storage_type} during code extraction.",
                                      RuntimeWarning)
                        return None  # Ignore storage data
                    except Exception:
                        warnings.warn(f"Failed to parse persistent_load pid: {pid}", RuntimeWarning)
                        return None
                warnings.warn(f"Ignoring unknown persistent_load pid: {pid}", RuntimeWarning)
                return None  # Ignore other persistent objects

        model_code: Optional[str] = None
        module_name: Optional[str] = None

        try:
            if zipfile.is_zipfile(path):
                # New zipfile format
                with zipfile.ZipFile(path, 'r') as zip_file:
                    if 'data.pkl' in zip_file.namelist():
                        data_bytes = zip_file.read('data.pkl')
                        data_file = BytesIO(data_bytes)
                        unpickler = RobustUnpickler(data_file)
                        obj = unpickler.load()
                        # Navigate potential nested structure
                        if isinstance(obj, dict) and "auto_saver_state" in obj:
                            auto_saver_state = obj.get("auto_saver_state", {})
                            model_code = auto_saver_state.get("_model_code")
                            module_name = auto_saver_state.get("_module_name")
                    else:
                        raise RuntimeError("data.pkl not found in zip archive.")
            else:
                # Old pickle format (potentially less safe)
                with open(path, 'rb') as f:
                    unpickler = RobustUnpickler(f)
                    obj = unpickler.load()
                    if isinstance(obj, dict) and "auto_saver_kwargs" in obj:  # Legacy key name?
                        auto_saver_state = obj.get("auto_saver_kwargs", {})
                        model_code = auto_saver_state.get("_model_code")
                        module_name = auto_saver_state.get("_module_name")
                    elif isinstance(obj, dict) and "auto_saver_state" in obj:
                        auto_saver_state = obj.get("auto_saver_state", {})
                        model_code = auto_saver_state.get("_model_code")
                        module_name = auto_saver_state.get("_module_name")
                    else:
                        warnings.warn("Could not find 'auto_saver_state' or 'auto_saver_kwargs' in pickle object.",
                                      RuntimeWarning)

        except (pickle.UnpicklingError, EOFError, zipfile.BadZipFile, KeyError, AttributeError) as e:
            warnings.warn(f"Error extracting code/module name from {path}: {e}. Returning None.", RuntimeWarning)
            return None, None
        except Exception as e:  # Catch other potential errors
            warnings.warn(f"Unexpected error extracting code/module name from {path}: {e}. Returning None.",
                          RuntimeWarning)
            return None, None

        return model_code, module_name

    def score(self, t: Tensor, x: Tensor, *args) -> Tensor:
        """Computes the score function s(t, x) = ∇ log p_t(x).

        Requires model_type to be SCORE_BASED and SDE to be set.
        """
        if self._model_type != self.Model_Type.SCORE_BASED:
            raise NotImplementedError("Score function is only supported for SCORE_BASED models.")
        if self._sde is None:
            raise RuntimeError("SDE must be set to compute the score.")
        if self._model is None:
            raise RuntimeError("Model must be set to compute the score.")

        # Ensure t and x are tensors on the correct device
        t_dev = t.to(self._device) if isinstance(t, Tensor) else torch.tensor(t, device=self._device)
        x_dev = x.to(self._device)

        # Get model output (assumed to be model(t, x, *args) * sigma(t))
        model_output = self.module(t_dev, x_dev, *args)  # Use unwrapped model

        # Get sigma(t)
        sigma_t = self._sde.sigma(t_dev)

        # Ensure sigma_t has correct shape for broadcasting
        _, *D_x = x_dev.shape
        sigma_t = sigma_t.view(-1, *[1] * len(D_x))  # Reshape sigma for division

        # Calculate score: model_output / sigma(t)
        # Add small epsilon to prevent division by zero if sigma can be zero
        score_val = model_output / (sigma_t + 1e-8)

        return score_val

    @torch.no_grad()
    def sample(self, shape: Tuple[int, ...], steps: int, condition: Optional[list] = None,
               likelihood_score_fn: Optional[Callable] = None, guidance_factor: float = 1.,
               apply_ema: bool = True, bar: bool = True) -> Tensor:
        """Performs sampling using the Euler-Maruyama integration of the SDE.

        Args:
            shape (tuple): Shape of the tensor to sample (Batch, Channels, H, W...).
            steps (int): Number of discretization steps.
            condition (Optional[list]): Conditional inputs for the score function.
            likelihood_score_fn (Optional[Callable]): Additional drift for posterior sampling f(t, x).
            guidance_factor (float): Scaling for likelihood drift.
            apply_ema (bool): If True and EMA is enabled, use EMA weights for sampling.
            bar (bool): Display tqdm progress bar.

        Returns:
            torch.Tensor: The sampled data tensor.
        """
        if self._model_type != self.Model_Type.SCORE_BASED:
            raise NotImplementedError("Sampling is only supported for SCORE_BASED models.")
        if self._sde is None:
            raise RuntimeError("SDE must be set for sampling.")
        if self._model is None:
            raise RuntimeError("Model must be set for sampling.")

        if condition is None: condition = []
        self.eval()  # Ensure model is in eval mode

        tqdm_module = None
        if bar:
            try:
                from tqdm.auto import tqdm as tqdm_auto
                tqdm_module = tqdm_auto
            except ImportError:
                if self.__logger: self.__logger.warning("tqdm not found. Progress bar disabled for sampling.")
                bar = False

        B, *D = shape
        sampling_from = "prior" if likelihood_score_fn is None else "posterior"
        if likelihood_score_fn is None:
            # Define a lambda that returns 0 with correct device and shape
            def zero_likelihood_score(t, x):
                return torch.zeros_like(x)

            likelihood_score_fn = zero_likelihood_score
        else:
            # Ensure user provided function handles device and shapes correctly
            pass

        # Initial sample from prior
        try:
            x = self._sde.prior(D).sample([B]).to(self._device)
        except Exception as e:
            raise RuntimeError(f"Failed to sample from SDE prior: {e}") from e

        # Time schedule
        # Check if SDE class name is VPSDE without importing score_models directly
        is_vpsde = hasattr(self._sde, '__class__') and self._sde.__class__.__name__ == "VPSDE"

        time_schedule = torch.linspace(self._sde.T, self._sde.epsilon, steps + 1, device=self._device)
        if is_vpsde:
            t_schedule = torch.tensor([
                self._sde.epsilon + 0.5 * (self._sde.T - self._sde.epsilon) * (1 + math.cos(math.pi * i / steps))
                for i in range(steps + 1)
            ], device=self._device)
            if self.__logger: self.__logger.debug("Using custom cosine time schedule for VPSDE.")
        else:
            t_schedule = time_schedule  # Linear schedule
            if self.__logger: self.__logger.debug("Using linear time schedule.")

        dt = -(self._sde.T - self._sde.epsilon) / steps  # Constant dt for linear schedule (negative)

        pbar_iterator = tqdm_module(range(steps), desc=f"Sampling ({sampling_from})") if bar else range(steps)

        # Apply EMA weights for the whole sampling process if requested
        ema_context = self._ema.average_parameters() if (self._ema and apply_ema) else contextlib.nullcontext()
        x_mean = torch.zeros_like(x)  # Initialize x_mean

        with ema_context:
            for i in pbar_iterator:
                t_current = t_schedule[i]
                t_next = t_schedule[i + 1]
                step_dt = t_next - t_current  # dt for this step (should be negative)

                # Ensure t is a tensor with batch dimension
                t_batch = torch.ones(B, device=self._device) * t_current

                if t_current <= self._sde.epsilon:  # Check against current t
                    if self.__logger: self.__logger.warning(
                        f"Reached time epsilon ({self._sde.epsilon:.4f}) early at step {i}. Stopping.")
                    break  # Stop if time goes below epsilon

                # Get score, drift, diffusion
                g = self._sde.diffusion(t_batch, x)
                # Calculate score s(t, x)
                score_val = self.score(t_batch, x, *condition)
                # Calculate likelihood score (ensure it returns tensor on correct device)
                likelihood_score_val = likelihood_score_fn(t_batch, x)
                # Combine scores
                combined_score = score_val + guidance_factor * likelihood_score_val

                # Calculate reverse SDE drift: f(t,x) - g(t)^2 * s_theta(t,x)
                f = self._sde.drift(t_batch, x)
                drift = f - g ** 2 * combined_score

                # Noise term for Euler-Maruyama step
                # dw ~ N(0, dt) -> sqrt(-dt) * Z where Z ~ N(0, I) because dt is negative
                dw = torch.randn_like(x) * torch.sqrt(-step_dt)

                # Euler-Maruyama step: x_{t-1} = x_t + f_reverse(t, x_t) * dt + g(t) * dw
                x_mean = x + drift * step_dt  # Mean update
                x = x_mean + g * dw  # Add noise

                if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
                    error_msg = f"NaN or Inf detected in sampling at step {i + 1} (t={t_current:.4f}). Stopping."
                    warnings.warn(error_msg, RuntimeWarning)
                    if self.__logger: self.__logger.error(error_msg)
                    # Return the last valid mean if possible, otherwise the corrupted x
                    return x_mean if not (torch.any(torch.isnan(x_mean)) or torch.any(torch.isinf(x_mean))) else x

                if bar and tqdm_module:
                    pbar_iterator.set_postfix_str(
                        f"t={t_current.item():.3f}, sigma={g[0].item():.2e}, |x| ~ {x.abs().mean().item():.2e}")

        # Return the mean of the last step (often considered cleaner)
        return x_mean

    @torch.no_grad()
    def log_likelihood(self, x: Tensor, *args, ode_steps: int,
                       n_cotangent_vectors: int = 1, noise_type: str = "rademacher",
                       method: str = "Euler", t0: float = None, t1: float = None,  # Use floats for t0, t1
                       apply_ema: bool = True, pbar: bool = False) -> Tensor:
        """Estimates the log-likelihood using the instantaneous change of variables formula (ODE method).

        Args:
            x (Tensor): Input data tensor (Batch, ...).
            *args: Additional arguments passed to the score function.
            ode_steps (int): Number of discretization steps for the ODE solver.
            n_cotangent_vectors (int): Number of Hutchinson trace estimator samples.
            noise_type (str): Type of noise for Hutchinson ('rademacher' or 'gaussian').
            method (str): ODE solver ('Euler' or 'Heun').
            t0 (float): Starting time for the ODE integration (defaults to SDE epsilon).
            t1 (float): Ending time for the ODE integration (defaults to SDE T).
            apply_ema (bool): If True and EMA is enabled, use EMA weights.
            pbar (bool): Display tqdm progress bar.

        Returns:
            Tensor: Estimated log-likelihood for each sample in the batch (Batch,).
        """
        if self._model_type != self.Model_Type.SCORE_BASED:
            raise NotImplementedError("Log-likelihood estimation is only supported for SCORE_BASED models.")
        if self._sde is None:
            raise RuntimeError("SDE must be set for log-likelihood estimation.")
        if self._model is None:
            raise RuntimeError("Model must be set for log-likelihood estimation.")

        self.eval()  # Ensure model is in eval mode

        # Set default integration times if not provided
        t_start = t0 if t0 is not None else self._sde.epsilon
        t_end = t1 if t1 is not None else self._sde.T

        if not (t_start >= self._sde.epsilon and t_end <= self._sde.T and t_start < t_end):
            raise ValueError(
                f"Invalid time range [{t_start}, {t_end}]. Must be within [{self._sde.epsilon}, {self._sde.T}].")

        # Ensure x is on the correct device
        x = x.to(self._device)
        B, *D = x.shape

        # --- Define ODE Drift and Divergence ---
        # Use self.module to access the unwrapped model for score calculation
        target_model = self.module

        def ode_drift_func(t: Tensor, x_in: Tensor, *drift_args) -> Tensor:
            # Ensure t and x are tensors with batch dim
            t_batch = t if t.ndim > 0 else torch.tensor([t.item()] * x_in.shape[0], device=self._device)
            f = self._sde.drift(t_batch, x_in)
            g = self._sde.diffusion(t_batch, x_in)
            # Score calculation needs the model
            score_val = target_model(t_batch, x_in, *drift_args) / (
                    self._sde.sigma(t_batch).view(-1, *[1] * len(D)) + 1e-8)
            f_tilde = f - 0.5 * g ** 2 * score_val
            return f_tilde

        def divergence_func(t: Tensor, x_in: Tensor, *div_args) -> Tensor:
            # Prepare inputs for vjp
            samples = x_in.repeat_interleave(n_cotangent_vectors, dim=0)
            t_repeated = t.repeat_interleave(n_cotangent_vectors, dim=0)
            # Repeat args if they are per-sample? Assuming args are global for now.

            # Sample cotangent vectors
            if noise_type == 'rademacher':
                vectors = torch.randint(low=0, high=2, size=samples.shape, device=self._device).float() * 2 - 1
            elif noise_type == 'gaussian':
                vectors = torch.randn_like(samples)
            else:
                raise ValueError(f"Unknown noise type: {noise_type}")

            # Define function for vjp: drift(t, x, *args)
            f_vjp = lambda x_vjp: ode_drift_func(t_repeated, x_vjp, *div_args)

            # Compute vjp and divergence
            _, vjp_fn = vjp(f_vjp, samples)
            vjp_product = vjp_fn(vectors)[0]  # Get the vector-Jacobian product

            # Reshape for summation: (B * n_cotangent, ...) -> (B * n_cotangent, Dim)
            vjp_product_flat = vjp_product.flatten(start_dim=1)
            vectors_flat = vectors.flatten(start_dim=1)

            # Calculate divergence for each sample in the expanded batch
            # Sum over feature dimensions: (B * n_cotangent,)
            div_expanded = torch.sum(vectors_flat * vjp_product_flat, dim=1)

            # Average over Hutchinson samples for each original batch item
            # Reshape to (B, n_cotangent) and average
            div_avg = div_expanded.view(B, n_cotangent_vectors).mean(dim=1)
            return div_avg

        # --- ODE Integration ---
        tqdm_module = None
        if pbar:
            try:
                from tqdm.auto import tqdm as tqdm_auto
                tqdm_module = tqdm_auto
            except ImportError:
                if self.__logger: self.__logger.warning("tqdm not found. Progress bar disabled for log-likelihood.")
                pbar = False

        log_p = torch.zeros(B, device=self._device)  # Log likelihood accumulator per sample
        current_t = torch.ones(B, device=self._device) * t_start
        dt = (t_end - t_start) / ode_steps

        pbar_iterator = tqdm_module(range(ode_steps), desc="Log-Likelihood ODE") if pbar else range(ode_steps)

        # Apply EMA weights for the whole process if requested
        ema_context = self._ema.average_parameters() if (self._ema and apply_ema) else contextlib.nullcontext()

        with ema_context:
            for i in pbar_iterator:
                step_start_time = time.time()
                # Calculate drift and divergence at current time t
                current_drift = ode_drift_func(current_t, x, *args)
                current_div = divergence_func(current_t, x, *args)

                if method == "Euler":
                    # Update x: x = x + f(t, x) * dt
                    x = x + current_drift * dt
                    # Update log_p: log_p += div(t, x) * dt (Note: divergence is calculated *before* x update)
                    log_p += current_div * dt
                    # Update time: t = t + dt
                    current_t = current_t + dt

                elif method == "Heun":
                    # Predictor step: x_pred = x + f(t, x) * dt
                    x_pred = x + current_drift * dt
                    t_next = current_t + dt
                    # Corrector step: calculate drift and div at t+dt with x_pred
                    next_drift = ode_drift_func(t_next, x_pred, *args)
                    next_div = divergence_func(t_next, x_pred, *args)  # Use x_pred for div at t+dt
                    # Update x: x = x + 0.5 * (f(t, x) + f(t+dt, x_pred)) * dt
                    x = x + 0.5 * (current_drift + next_drift) * dt
                    # Update log_p: log_p += 0.5 * (div(t, x_old) + div(t+dt, x_pred)) * dt
                    # Note: Heun for the integral term uses div at t and t+dt
                    log_p += 0.5 * (current_div + next_div) * dt
                    # Update time
                    current_t = t_next

                else:
                    raise NotImplementedError(f"ODE solver method '{method}' not implemented. Use 'Euler' or 'Heun'.")

                if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)) or \
                        torch.any(torch.isnan(log_p)) or torch.any(torch.isinf(log_p)):
                    error_msg = f"NaN or Inf detected during ODE integration at step {i + 1} (t={current_t[0].item():.4f}). Stopping."
                    warnings.warn(error_msg, RuntimeWarning)
                    if self.__logger: self.__logger.error(error_msg)
                    # Return NaN for the affected batch elements or all? Return all NaN for safety.
                    return torch.full_like(log_p, float('nan'))

                if pbar and tqdm_module:
                    step_time = time.time() - step_start_time
                    pbar_iterator.set_postfix_str(
                        f"t={current_t[0].item():.3f}, Δt={step_time:.3f}s, logp={log_p.mean().item():.2e}")

        # Add log probability from the prior distribution at the final time T
        try:
            prior_log_prob = self._sde.prior(D).log_prob(x)
            log_p += prior_log_prob
        except Exception as e:
            warnings.warn(f"Failed to compute prior log probability at t={t_end:.4f}: {e}", RuntimeWarning)
            # Return NaN if prior fails? Or just the integrated part? Return NaN for consistency.
            return torch.full_like(log_p, float('nan'))

        return log_p

# --- Example Usage Helper Function ---
# import contextlib  # Needed for nullcontext
#
#
# def get_dummy_components(device='cpu', n_features=10, n_classes=2, batch_size=16, dataset_size=64):
#     """Helper to create dummy components for testing NNHandler."""
#
#     # Dummy Model
#     class DummyModel(nn.Module):
#         def __init__(self, input_features, output_classes):
#             super().__init__()
#             self.layer = nn.Linear(input_features, output_classes)
#
#         def forward(self, x):
#             return self.layer(x)
#
#     # Dummy Dataset
#     class DummyDataset(Dataset):
#         def __init__(self, size, features, classes, model_type):
#             self.size = size
#             self.features = features
#             self.classes = classes
#             self.model_type = model_type
#             # Generate random data once
#             self.data = torch.randn(size, features)
#             if model_type == NNHandler.ModelType.CLASSIFICATION:
#                 self.labels = torch.randint(0, classes, (size,))
#             elif model_type == NNHandler.ModelType.REGRESSION:
#                 self.labels = torch.randn(size, classes)  # Regression target
#             else:  # Generative/Score-based only need data
#                 self.labels = None
#
#         def __len__(self):
#             return self.size
#
#         def __getitem__(self, idx):
#             if self.labels is not None:
#                 return self.data[idx], self.labels[idx]
#             else:
#                 return self.data[idx]  # Only return data for generative/score
#
#     # Dummy Loss
#     def dummy_loss(output, target, **kwargs):
#         if isinstance(output, torch.Tensor) and isinstance(target, torch.Tensor):
#             # Simple MSE loss for regression/generative, CrossEntropy for classification
#             if target.dtype == torch.long:  # Classification
#                 return nn.CrossEntropyLoss()(output, target)
#             else:  # Regression or Generative
#                 return nn.MSELoss()(output, target)
#         return torch.tensor(0.0)  # Default case
#
#     # Dummy Metric (Accuracy)
#     def dummy_accuracy(output, target):
#         if isinstance(output, torch.Tensor) and isinstance(target, torch.Tensor) and target.dtype == torch.long:
#             preds = torch.argmax(output, dim=1)
#             return (preds == target).float().mean()
#         return 0.0  # Not applicable otherwise
#
#     # Create instances
#     model_cls = DummyModel
#     model_kwargs = {'input_features': n_features, 'output_classes': n_classes}
#
#     # Adjust dataset/loss based on a chosen type (e.g., CLASSIFICATION)
#     model_type = NNHandler.ModelType.CLASSIFICATION
#     train_dataset = DummyDataset(dataset_size, n_features, n_classes, model_type)
#     val_dataset = DummyDataset(dataset_size // 2, n_features, n_classes, model_type)
#
#     train_loader_kwargs = {'batch_size': batch_size, 'shuffle': True}
#     val_loader_kwargs = {'batch_size': batch_size, 'shuffle': False}
#
#     optimizer_cls = torch.optim.Adam
#     optimizer_kwargs = {'lr': 1e-3}
#
#     loss_fn = dummy_loss
#     metrics = {'accuracy': dummy_accuracy}
#
#     return {
#         'model_class': model_cls, 'model_kwargs': model_kwargs, 'model_type': model_type,
#         'train_dataset': train_dataset, 'val_dataset': val_dataset,
#         'train_loader_kwargs': train_loader_kwargs, 'val_loader_kwargs': val_loader_kwargs,
#         'optimizer_class': optimizer_cls, 'optimizer_kwargs': optimizer_kwargs,
#         'loss_fn': loss_fn, 'metrics': metrics,
#         'device': device
#     }
#
#
# if __name__ == "__main__":
#     print("Running NNHandler Enhanced Demo...")
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")
#
#     # 1. Get Dummy Components
#     components = get_dummy_components(device=device)
#
#     # 2. Initialize NNHandler
#     handler = NNHandler(
#         model_class=components['model_class'],
#         device=components['device'],
#         logger_mode=NNHandler.LoggingMode.CONSOLE,  # Log to console
#         logger_level=logging.INFO,
#         model_type=components['model_type'],
#         **components['model_kwargs']
#     )
#
#     # 3. Configure Handler
#     handler.set_optimizer(components['optimizer_class'], **components['optimizer_kwargs'])
#     handler.set_loss_fn(components['loss_fn'])
#     handler.set_train_loader(components['train_dataset'], **components['train_loader_kwargs'])
#     handler.set_val_loader(components['val_dataset'], **components['val_loader_kwargs'])
#     for name, fn in components['metrics'].items():
#         handler.add_metric(name, fn)
#
#     # Add Callbacks (Example)
#     checkpoint_dir = "./nn_handler_checkpoints"
#     # Save best model based on validation accuracy
#     model_checkpoint = ModelCheckpoint(
#         filepath=os.path.join(checkpoint_dir, "best_model_epoch{epoch:02d}_val_acc{val_accuracy:.2f}.pth"),
#         monitor='val_accuracy',  # Monitor validation accuracy
#         mode='max',  # Maximize accuracy
#         save_best_only=True,
#         save_weights_only=False,  # Save full handler state
#         verbose=1
#     )
#     early_stopping = EarlyStopping(
#         monitor='val_accuracy',  # Monitor validation accuracy
#         mode='max',
#         patience=5,  # Stop after 5 epochs of no improvement
#         verbose=1,
#         restore_best_weights=True  # Restore weights from best epoch
#     )
#     lr_monitor = LearningRateMonitor()
#
#     # Add TensorBoard if available
#     try:
#         tb_logger = TensorBoardLogger(log_dir="./nn_handler_tb_logs")
#         handler.add_callback(tb_logger)
#         print("Added TensorBoardLogger.")
#     except ImportError:
#         print("tensorboardX not found, skipping TensorBoard callback.")
#
#     handler.add_callback(model_checkpoint)
#     handler.add_callback(early_stopping)
#     handler.add_callback(lr_monitor)
#
#     # Configure Auto Save (e.g., every 2 epochs, keep only last)
#     handler.auto_save(interval=2, save_path=checkpoint_dir, name="autosave_epoch{epoch:02d}", overwrite=True)
#
#     # Print handler status before training
#     print("\n--- Handler Status Before Training ---")
#     print(handler)
#     print("-" * 30)
#
#     # 4. Train
#     print("\n--- Starting Training ---")
#     handler.train(
#         epochs=10,
#         validate_every=1,
#         use_amp=(device == 'cuda'),  # Use AMP if on CUDA
#         gradient_accumulation_steps=1,
#         progress_bar=True
#     )
#     print("-" * 30)
#
#     # 5. Post-Training Information
#     print("\n--- Handler Status After Training ---")
#     print(handler)
#     print("-" * 30)
#
#     print("\n--- Plotting Results ---")
#     # Plot losses
#     try:
#         handler.plot_losses(save_path="./nn_handler_losses.png")
#         print("Loss plot saved to ./nn_handler_losses.png")
#     except Exception as e:
#         print(f"Could not plot losses: {e}")
#
#     # Plot metrics
#     try:
#         handler.plot_metrics(save_path_prefix="./nn_handler")
#         print("Metric plots saved with prefix ./nn_handler_metric_")
#     except Exception as e:
#         print(f"Could not plot metrics: {e}")
#     print("-" * 30)
#
#     # 6. Saving and Loading Example
#     print("\n--- Saving and Loading Test ---")
#     final_save_path = os.path.join(checkpoint_dir, "final_model_state.pth")
#     print(f"Saving final handler state to: {final_save_path}")
#     handler.save(final_save_path)
#
#     # Check if best model was saved by ModelCheckpoint
#     best_model_path = model_checkpoint.filepath.format(
#         epoch=early_stopping.stopped_epoch if early_stopping.stopped_epoch > 0 else handler.train_losses.index(
#             min(handler.train_losses)) + 1,  # Approximat
#         val_accuracy=model_checkpoint.best.item() if isinstance(model_checkpoint.best, torch.Tensor) else 0.0)
#
#     # Find the actual best saved file (since formatting might vary slightly)
#     best_saved_file = None
#     if os.path.exists(checkpoint_dir):
#         potential_files = [f for f in os.listdir(checkpoint_dir) if
#                            f.startswith("best_model_epoch") and f.endswith(".pth")]
#         if potential_files:
#             # Sort by modification time or parse epoch/metric? Parsing is better.
#             best_score = -torch.inf
#             for f in potential_files:
#                 try:
#                     score = float(f.split('val_acc')[-1].replace('.pth', ''))
#                     if score > best_score:
#                         best_score = score
#                         best_saved_file = os.path.join(checkpoint_dir, f)
#                 except:
#                     continue  # Ignore files that don't match format
#
#     if best_saved_file and os.path.exists(best_saved_file):
#         print(f"\nLoading BEST handler state from ModelCheckpoint: {best_saved_file}")
#         try:
#             loaded_handler_best = NNHandler.load(best_saved_file, device=device)
#             print("Best handler loaded successfully.")
#             print(f"Best validation accuracy achieved: {model_checkpoint.best.item():.4f}")
#
#             # Example: Use loaded handler for prediction
#             print("Running prediction with loaded best model...")
#             dummy_pred_loader = DataLoader(components['val_dataset'], batch_size=4)
#             predictions = loaded_handler_best.predict(dummy_pred_loader,
#                                                       apply_ema=False)  # EMA state might not be saved in best weights checkpoint
#             print(
#                 f"Prediction output shape for first batch: {predictions[0].shape if isinstance(predictions[0], torch.Tensor) else 'N/A'}")
#
#         except Exception as e:
#             print(f"ERROR loading best handler state: {e}")
#             import traceback
#
#             traceback.print_exc()
#
#     else:
#         print("\nCould not find or load best model checkpoint saved by ModelCheckpoint.")
#         # Load final state instead
#         print(f"\nLoading FINAL handler state from: {final_save_path}")
#         if os.path.exists(final_save_path):
#             try:
#                 loaded_handler_final = NNHandler.load(final_save_path, device=device)
#                 print("Final handler loaded successfully.")
#             except Exception as e:
#                 print(f"ERROR loading final handler state: {e}")
#         else:
#             print("Final save file not found.")
#
#     print("\nDemo finished.")
