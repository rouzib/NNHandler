import logging
import os
import sys
import warnings
from collections import defaultdict
from typing import Optional, Dict, Any, Callable, List, Union, Type, Tuple
import time
import math

import torch
from matplotlib import pyplot as plt
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .__version__ import __version__
from .logger import initialize_logger
from .callbacks.base import Callback
from .checkpointing.autosaver import AutoSaver
from .checkpointing.saving import save_single_file, save_multi_files
from .checkpointing.loading import load, initialize_from_checkpoint
from .model_utils.score_models.sde_solver import SdeSolver
from .model_utils.sampler import Sampler
from .utils import ExponentialMovingAverage, GradScaler, _amp_available, _ema_available
from .utils.enums import ModelType, LoggingMode, DataLoaderType
from .utils.ddp_init import _resolve_device, _should_use_distributed, _initialize_distributed
from .utils.ddp_decorators import on_rank
from .utils.ddp_helpers import _create_distributed_loader, broadcast_if_ddp, _create_rank_cached_dataloader
from .trainer.trainer import train


class NNHandler:
    @property
    def version(self):
        return __version__

    # --- Core Attributes ---
    _optimizer: Optional[torch.optim.Optimizer] = None
    _optimizer_kwargs: Dict[str, Any] = {}
    _scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None  # Using base class for broader type hint
    _scheduler_kwargs: Dict[str, Any] = {}
    _loss_fn: Optional[Callable] = None
    _loss_fn_kwargs: Dict[str, Any] = {}
    _pass_epoch_to_loss: bool = False
    _train_loader: Optional[DataLoader] = None
    _train_loader_kwargs: Dict[str, Any] = {}
    _train_dataset: Optional[Dataset] = None  # Keep track of original dataset
    _val_loader: Optional[DataLoader] = None
    _val_loader_kwargs: Dict[str, Any] = {}
    _val_dataset: Optional[Dataset] = None  # Keep track of original dataset
    _model: Optional[nn.Module] = None
    _model_class: Optional[type[nn.Module]] = None
    _model_kwargs: Optional[Dict[str, Any]] = None
    _model_type: ModelType = None
    _compiled_model: bool = False
    _sampler: Optional[Sampler] = None  # Custom sampler
    _sampler_kwargs: Dict[str, Any] = {}
    _sde: Optional[Any] = None  # Assuming SDE class structure
    _sde_kwargs: Dict[str, Any] = {}
    _device: torch.device = None  # Resolved device
    _seed: Optional[int] = None
    _auto_saver: AutoSaver = None
    _ema: Optional[ExponentialMovingAverage] = None
    _ema_decay: float = None
    _train_losses: List[float] = []  # Aggregated history on rank 0
    _val_losses: List[float] = []  # Aggregated history on rank 0
    _metrics: Dict[str, Callable] = {}
    _train_metrics_history: Dict[str, List[float]] = defaultdict(list)  # Aggregated history on rank 0
    _val_metrics_history: Dict[str, List[float]] = defaultdict(list)  # Aggregated history on rank 0
    _callbacks: List[Callback] = []
    _stop_training: bool = False  # Flag for early stopping, needs broadcast
    _grad_scaler: GradScaler = None
    _modules_always_eval: List[nn.Module] = []
    _complimentary_kwargs: Dict[str, Any] = {}
    _ddp_kwargs: Dict[str, Any] = {}

    # --- DDP Specific Attributes ---
    _distributed: bool = False
    _rank: int = None
    _local_rank: int = None
    _world_size: int = None
    _train_sampler: Optional[DistributedSampler] = None  # For setting epoch
    _val_sampler: Optional[DistributedSampler] = None  # For setting epoch

    __logger: Optional[logging.Logger] = None  # Logger instance

    def _setup_distributed(self, use_distributed: Optional[bool] = None, device: Union[torch.device, str] = "cuda"):
        """
        Sets up the distributed processing environment for the system. If distributed
        processing is enabled, it initializes the process group and sets relevant
        attributes such as rank, local rank, world size, and device. Otherwise, it
        resolves the computation device for non-distributed operations.

        :param use_distributed: Optionally specifies whether to use distributed
            processing. If not provided, the decision is determined internally.
        :type use_distributed: Optional[bool]
        :param device: Specifies the computation device to use (e.g., "cuda",
            "cpu", or a torch.device). This is used only in non-distributed mode.
        :type device: Union[torch.device, str]
        :return: None
        """
        self._distributed = _should_use_distributed(use_distributed)
        if self._distributed:
            # Sets self._rank, self._local_rank, self._world_size, self._device
            self._distributed, self._rank, self._local_rank, self._world_size, self._device = _initialize_distributed()
        else:
            self._rank = 0
            self._local_rank = -1  # Convention for non-distributed or CPU
            self._world_size = 1
            self._device = _resolve_device(device)  # Resolve device if not DDP

    def __init__(self,
                 model_class: type[nn.Module],
                 device: Union[torch.device, str] = "cpu",
                 logger_mode: Optional[LoggingMode] = None,
                 logger_filename: str = "NNHandler.log",
                 logger_filemode: str = "a",
                 logger_level: int = logging.INFO,
                 save_model_code: bool = False,
                 model_type: Union[ModelType, str] = ModelType.CLASSIFICATION,
                 use_distributed: Optional[bool] = None,  # DDP control flag
                 ddp_kwargs: Optional[dict] = None,
                 **model_kwargs):
        self._setup_distributed(use_distributed, device)

        # --- Model Type ---
        self._model_type = ModelType.parse(model_type)

        self._model_class = model_class
        self._model_kwargs = model_kwargs

        if ddp_kwargs is not None:
            self._ddp_kwargs = ddp_kwargs

        if logger_mode is not None:
            self.__logger = initialize_logger("NNHandler", logger_mode, logger_filename, logger_filemode,
                                              logger_level)

        # --- Initialize AutoSaver ---
        # Only save code on rank 0
        self._auto_saver = AutoSaver(save_model_code=(save_model_code and self._rank == 0))

        # --- Initialize Model (after DDP setup and logging) ---
        # This needs to happen after device is set, potentially wraps with DDP
        self.set_model(model_class=self._model_class,
                       save_model_code=save_model_code,
                       model_type=self._model_type,  # Pass resolved type
                       ddp_kwargs=self._ddp_kwargs,
                       **self._model_kwargs)

        msg_lines = [
            f"--- NNHandler Initialization (Rank {self._rank}) ---",
            f"  Model Class:         {self._model_class.__name__}",
            f"  Model Type:          {self._model_type.name}",
            f"  Distributed (DDP):   {self._distributed}",
        ]
        if self._distributed:
            msg_lines.extend([
                f"  World Size:          {self._world_size}",
                f"  Global Rank:         {self._rank}",
                f"  Local Rank:          {self._local_rank}"
            ])
            if ddp_kwargs is not None:
                msg_lines.append(f"  DDP kwargs: {self._ddp_kwargs}")
        msg_lines.extend([
            f"  Target Device:       {self._device}",
            f"  AMP Available:       {_amp_available}",
            f"  EMA Available:       {_ema_available}",
        ])
        self.log('\n'.join(msg_lines))

        if self._distributed:
            self.log(f"Rank {self._rank} waiting at init barrier.", level=logging.DEBUG)
            dist.barrier()
            self.log(f"Rank {self._rank} passed init barrier.", level=logging.DEBUG)

    @on_rank(0)
    def log(self, msg: str, level: int = logging.INFO):
        """
        Logs a message with the specified logging level. If a logger instance
        is available, it forwards the log message and level to the logger.

        :param msg: The message to be logged.
        :type msg: str
        :param level: The logging level for the message. Defaults to logging.INFO.
        :type level: int
        :return: None
        """
        if self.__logger is not None:
            self.__logger.log(level, msg)
        # print(level, ":  ", str(msg))

    @on_rank(0)
    def warn(self, msg: str, category: Type[Warning] = None):
        """
        Warns the user by either logging the message via the logger if available
        or issuing a Python warning. This method provides a mechanism to handle
        warnings either through a centralized logger or the Python warnings module.

        :param msg: The warning message to be logged or issued.
        :type msg: str
        :param category: The warning category to classify the warning. Should be
            a subclass of Warning.
        :type category: Type[Warning]
        :return: None
        """
        if self.__logger is not None:
            self.__logger.warning(msg)
        elif category is not None:
            warnings.warn(msg, category)

    def raise_error(self, error: Callable, msg: str, reraised_exception: Optional[Exception] = None):
        """
        Raises the specified error with a given message and optionally re-raises
        an existing exception. This function logs the message at an error level
        before raising the exception. If a reraised_exception is provided,
        it chains the new error with it.

        :param error: The callable that represents the exception type to be raised.
        :param msg: The message to be logged and passed to the exception.
        :param reraised_exception: An optional existing exception to be re-raised
            along with the new error.
        :return: None
        """
        self.log(msg, level=logging.ERROR)
        if self._distributed:
            dist.destroy_process_group()
        if reraised_exception:
            raise error(msg) from reraised_exception
        raise error(msg)

    # --- Properties & Setters ---

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
                self.warn(
                    f"Cannot change device after DDP initialization. Device remains '{self._device}'. "
                    f"Attempted to set to '{resolved_value}'.",
                    RuntimeWarning)
            return

        # Logic for non-distributed mode
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
            self.log(f"Detected {torch.cuda.device_count()} GPUs. Wrapping model with nn.DataParallel.")
            self._model = nn.DataParallel(raw_model)
        elif not is_multi_gpu_cuda and is_already_dp:
            # Unwrap from DataParallel
            self.log("Device changed or single GPU detected. Unwrapping model from nn.DataParallel.")
            self._model = raw_model  # Assign the unwrapped module back

    @property
    def seed(self) -> Optional[int]:
        return self._seed

    @seed.setter
    def seed(self, seed_value: Optional[int]):
        """Sets the random seed for torch and CUDA (applied on all ranks)."""
        if seed_value is not None:
            if not isinstance(seed_value, int):
                self.raise_error(TypeError, f"Seed must be an integer or None, got {type(seed_value)}.")

            # Set seed on all ranks to ensure consistent initialization where needed
            # (e.g., model weights before DDP syncs them, random operations in datasets)
            torch.manual_seed(seed_value)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_value)  # Seed all GPUs relevant to this process

            self.log(f"Global random seed set to: {seed_value}")

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
    def complimentary_kwargs(self) -> Dict[str, Any]:
        return self._complimentary_kwargs

    def add_complimentary_kwargs(self, **kwargs):
        self._complimentary_kwargs.update(**kwargs)

    @property
    def model_code(self) -> Optional[str]:
        # AutoSaver handles rank check internally for saving code
        return self._auto_saver.model_code

    def set_model(self, model_class: type[nn.Module], save_model_code: bool = False,
                  model_type: Optional[Union[ModelType, str]] = None, ddp_kwargs: Optional[Dict] = None,
                  **model_kwargs):
        """Sets or replaces the model, handling DDP/DataParallel wrapping."""
        if not issubclass(model_class, nn.Module):
            self.raise_error(TypeError, f"model_class must be a subclass of torch.nn.Module, got {model_class}.")

        if ddp_kwargs is None:
            ddp_kwargs = {}

        # Resolve model type
        if model_type is not None:
            self._model_type = ModelType.parse(model_type)

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
            if "find_unused_parameters" in ddp_kwargs.keys():
                ddp_find_unused = ddp_kwargs.pop("find_unused_parameters")
            if self._device.type == 'cuda':
                # Ensure device_ids is a list containing the local rank
                self._model = DDP(base_model, device_ids=[self._local_rank], output_device=self._local_rank,
                                  find_unused_parameters=ddp_find_unused, **ddp_kwargs)
            else:
                # DDP on CPU doesn't use device_ids or output_device
                self._model = DDP(base_model, find_unused_parameters=ddp_find_unused, **ddp_kwargs)

            # Log DDP wrapping details
            self.log(
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
            self.log("Model replaced. Re-initializing optimizer with previous settings.")
            # Use saved class and kwargs
            self.set_optimizer(self._optimizer.__class__, **self._optimizer_kwargs)
            # Scheduler is re-initialized within set_optimizer if it was set

        self.log(f"Model set to {model_class.__name__} (Type: {self._model_type.name}).")
        self.log(f"Model contains {self.count_parameters():,} trainable parameters.")

    @property
    def optimizer(self) -> Optional[torch.optim.Optimizer]:
        return self._optimizer

    def set_optimizer(self, optimizer_class: type[torch.optim.Optimizer], **optimizer_kwargs):
        """Sets the optimizer and re-initializes the scheduler if present."""
        if not issubclass(optimizer_class, torch.optim.Optimizer):
            self.raise_error(TypeError,
                             f"optimizer_class must be a subclass of torch.optim.Optimizer, got {optimizer_class}.")
        if self._model is None:
            self.raise_error(RuntimeError, "Model must be set before setting the optimizer.")

        self._optimizer_kwargs = optimizer_kwargs
        # Pass the parameters of the potentially wrapped model
        # DDP/DataParallel correctly handle requests for .parameters()
        self._optimizer = optimizer_class(self.model.parameters(), **optimizer_kwargs)

        self.log(f"Optimizer set to {optimizer_class.__name__} with kwargs: {optimizer_kwargs}")

        # Re-initialize scheduler with the new optimizer if it exists
        if self._scheduler is not None:
            self.log("Re-initializing scheduler with new optimizer.")
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
            self.log("Scheduler removed.")
            return

        # Check if it's a valid scheduler type (_LRScheduler or ReduceLROnPlateau or new LRScheduler base)
        # Using LRScheduler base class covers most modern schedulers
        if not issubclass(scheduler_class, torch.optim.lr_scheduler.LRScheduler):
            # Keep specific checks for older common types just in case
            if not (issubclass(scheduler_class, torch.optim.lr_scheduler._LRScheduler) or
                    issubclass(scheduler_class, torch.optim.lr_scheduler.ReduceLROnPlateau)):
                self.raise_error(TypeError, f"scheduler_class {scheduler_class} must be a subclass of "
                                            f"torch.optim.lr_scheduler.LRScheduler (or older base classes).")

        if self._optimizer is None:
            self.raise_error(RuntimeError, "Optimizer must be set before setting the scheduler.")

        self._scheduler_kwargs = scheduler_kwargs
        self._scheduler = scheduler_class(self._optimizer, **scheduler_kwargs)

        self.log(f"Scheduler set to {scheduler_class.__name__} with kwargs: {scheduler_kwargs}")

    # --- SDE/Sampler Properties ---
    @property
    def sde(self) -> Optional[Any]:
        return self._sde

    @sde.setter
    def sde(self, sde_instance: Any):
        """Sets the SDE instance directly."""
        # Basic check if it has expected methods/attributes
        expected_attrs = ['prior', 'drift', 'diffusion', 'sigma', 'T', 'epsilon']
        if not all(hasattr(sde_instance, attr) for attr in expected_attrs):
            self.warn("Provided SDE instance might be missing expected attributes/methods "
                      f"(e.g., {expected_attrs}).", RuntimeWarning)

        if self._model_type != ModelType.SCORE_BASED:
            self.warn(f"Model Type was {self._model_type.name}. Changed to SCORE_BASED as SDE was set.")
            self._model_type = ModelType.SCORE_BASED

        self._sde = sde_instance
        self._sde_kwargs = {}  # Clear kwargs if instance is set directly
        self.log(f"SDE instance set to: {sde_instance}.")

    def set_sde(self, sde_class: type, **sde_kwargs):
        """Sets the SDE by providing the class and keyword arguments."""
        # Could add more checks on sde_class if an SDE ABC/protocol exists
        if self._model_type != ModelType.SCORE_BASED:
            self.warn(f"Model Type was {self._model_type.name}. Changed to SCORE_BASED as SDE was set.")
        try:
            self._sde = sde_class(**sde_kwargs)
            self._sde_kwargs = sde_kwargs
            self.log(f"SDE set to {sde_class.__name__} with kwargs: {sde_kwargs}")
        except Exception as e:
            self.raise_error(RuntimeError,
                             f"Failed to instantiate SDE class {sde_class.__name__} with kwargs {sde_kwargs}: {e}", e)

    @property
    def sampler(self) -> Optional[Sampler]:  # Custom Sampler base class
        return self._sampler

    @sampler.setter
    def sampler(self, sampler_instance: Sampler):
        """Sets the Sampler instance directly."""
        # Check against the imported custom Sampler base class
        if not isinstance(sampler_instance, Sampler):
            self.raise_error(TypeError,
                             "sampler must be an instance of a class inheriting from the custom Sampler base class.")
        self._sampler = sampler_instance
        self._sampler_kwargs = {}  # Clear kwargs
        self.log(f"Custom sampler instance set to: {sampler_instance}")

    def set_sampler(self, sampler_class: type[Sampler], **sampler_kwargs):
        """Sets the Sampler by providing the class and keyword arguments."""
        # Check against the imported custom Sampler base class
        if not issubclass(sampler_class, Sampler):
            self.raise_error(TypeError,
                             f"sampler_class {sampler_class.__name__} must be a subclass of the custom Sampler base class.")

        try:
            self._sampler = sampler_class(**sampler_kwargs)
            self._sampler_kwargs = sampler_kwargs
            self.log(f"Custom sampler set to {sampler_class.__name__} with kwargs: {sampler_kwargs}")
        except Exception as e:
            self.raise_error(RuntimeError,
                             f"Failed to instantiate Sampler class {sampler_class.__name__} with kwargs {sampler_kwargs}: {e}",
                             e)

    # --- Loss Function Properties ---
    @property
    def loss_fn(self) -> Optional[Callable]:
        return self._loss_fn

    @loss_fn.setter
    def loss_fn(self, loss_function: Callable):
        """Sets the loss function. Use set_loss_fn for kwargs."""
        if not callable(loss_function):
            self.raise_error(TypeError, "loss_fn must be callable.")
        self._loss_fn = loss_function
        self._loss_fn_kwargs = {}  # Reset kwargs when set directly
        self._pass_epoch_to_loss = False  # Reset flag
        self.log(f"Loss function set to {getattr(self._loss_fn, '__name__', repr(self._loss_fn))}.")

    def set_loss_fn(self, loss_fn: Callable, pass_epoch_to_loss: bool = False, **kwargs):
        """Sets the loss function with optional kwargs and epoch passing flag."""
        if not callable(loss_fn):
            self.raise_error(TypeError, "loss_fn must be callable.")
        self._loss_fn = loss_fn
        self._pass_epoch_to_loss = pass_epoch_to_loss
        self._loss_fn_kwargs = kwargs or {}
        self.log(f"Loss function set to {getattr(self._loss_fn, '__name__', repr(self._loss_fn))}.")
        if pass_epoch_to_loss:
            self.log("Current epoch will be passed to the loss function if it accepts 'epoch' kwarg.")

    @property
    def pass_epoch_to_loss(self) -> bool:
        return self._pass_epoch_to_loss

    @pass_epoch_to_loss.setter
    def pass_epoch_to_loss(self, value: bool):
        if not isinstance(value, bool):
            self.raise_error(TypeError, "pass_epoch_to_loss must be a boolean.")
        self._pass_epoch_to_loss = value
        self.log(f"pass_epoch_to_loss set to {value}.")

    @property
    def loss_fn_kwargs(self) -> Dict[str, Any]:
        return self._loss_fn_kwargs

    @loss_fn_kwargs.setter
    def loss_fn_kwargs(self, value: Dict[str, Any]):
        if not isinstance(value, dict):
            self.raise_error(TypeError, "loss_fn_kwargs must be a dictionary.")
        self._loss_fn_kwargs = value or {}
        self.log(f"Loss function kwargs updated to: {self._loss_fn_kwargs}")

    @property
    def ema(self):
        return self._ema

    # --- Data Loaders ---
    @property
    def train_loader(self) -> Optional[DataLoader]:
        return self._train_loader

    @property
    def train_loader_kwargs(self) -> Dict[str, Any]:
        return self._train_loader_kwargs

    def _setup_loader(self, dataset, val: bool, dataloader_type, **loader_kwargs):
        """
        Set up a data loader with appropriate defaults for distributed or non-distributed
        environment, as well as train or validation mode. Supports multiple types of
        data loader creation.

        The function processes kwargs to configure a DataLoader instance such as the number
        of workers, whether to shuffle data, or whether to drop the last incomplete batch.
        For distributed settings, it also sets the correct loader function based on the
        loader type (e.g., STANDARD, RANK_CACHED).

        Args:
            dataset: The dataset object to be wrapped by the DataLoader.
            val: A boolean indicating whether the loader is for validation mode (True)
                or training mode (False).
            dataloader_type: Specifies the type of dataloader to be created. It should
                correspond to a defined enumeration like `DataLoaderType`.
            loader_kwargs: Dictionary containing keyword arguments passed to configure the
                DataLoader, such as `'num_workers'`, `'pin_memory'`, `'shuffle'`, etc.

        Raises:
            ValueError: If an unsupported `dataloader_type` is provided when in a
                distributed setting.
        """
        # Set sensible defaults for non-distributed mode
        if not self._distributed:
            loader_kwargs.setdefault('num_workers', 0)
            loader_kwargs.setdefault('pin_memory', getattr(self._device, 'type', None) == 'cuda')
            loader_kwargs.setdefault('persistent_workers', loader_kwargs['num_workers'] > 0)

        # Set additional defaults depending on train/val
        if val:
            loader_kwargs.setdefault('shuffle', False)
            loader_kwargs.setdefault('drop_last', False)
        else:
            loader_kwargs.setdefault('shuffle', True)

        # Choose loader function and set attributes
        if self._distributed:
            loader_fn = None
            if dataloader_type == DataLoaderType.STANDARD:
                loader_fn = _create_distributed_loader
            elif dataloader_type == DataLoaderType.RANK_CACHED:
                loader_fn = _create_rank_cached_dataloader
            else:
                self.raise_error(ValueError, f"Unsupported dataloader_type: {dataloader_type}")
            loader, sampler = loader_fn(dataset, loader_kwargs, self._device, self.log, is_eval=val)
            if val:
                self._val_loader, self._val_sampler = loader, sampler
            else:
                self._train_loader, self._train_sampler = loader, sampler
        else:
            data_loader = DataLoader(dataset, **loader_kwargs)
            if val:
                self._val_loader = data_loader
            else:
                self._train_loader = data_loader

    def set_train_loader(self, dataset: Dataset, dataloader_type: DataLoaderType = DataLoaderType.STANDARD,
                         **loader_kwargs):
        """Sets the training data loader, using DistributedSampler in DDP mode."""
        if not isinstance(dataset, Dataset):
            raise TypeError(f"dataset must be an instance of torch.utils.data.Dataset, got {type(dataset)}")

        self._train_dataset = dataset  # Store original dataset
        self._train_loader_kwargs = loader_kwargs
        self._train_sampler = None  # Reset sampler

        self._setup_loader(dataset, val=False, dataloader_type=dataloader_type, **self._train_loader_kwargs)

        self.log(f"Train DataLoader ({'DDP' if self._distributed else 'Standard'}) set for {type(dataset).__name__}.")

    @property
    def val_loader(self) -> Optional[DataLoader]:
        return self._val_loader

    @property
    def val_loader_kwargs(self) -> Dict[str, Any]:
        return self._val_loader_kwargs

    def set_val_loader(self, dataset: Dataset, dataloader_type: DataLoaderType = DataLoaderType.STANDARD,
                       **loader_kwargs):
        """Sets the validation data loader, using DistributedSampler in DDP mode.."""
        if not isinstance(dataset, Dataset):
            raise TypeError(f"dataset must be an instance of torch.utils.data.Dataset, got {type(dataset)}")

        self._val_dataset = dataset  # Store original dataset
        self._val_loader_kwargs = loader_kwargs
        self._val_sampler = None  # Reset sampler

        self._setup_loader(dataset, val=True, dataloader_type=dataloader_type, **self._val_loader_kwargs)

        self.log(
            f"Validation DataLoader ({'DDP' if self._distributed else 'Standard'}) set for {type(dataset).__name__}.")

    # --- Metrics ---

    @property
    def metrics(self) -> Dict[str, Callable]:
        return self._metrics

    def add_metric(self, name: str, metric_fn: Callable):
        """Adds a metric function to be tracked during training and validation."""
        if not callable(metric_fn):
            self.raise_error(TypeError, "metric_fn must be callable.")
        if not isinstance(name, str) or not name:
            self.raise_error(ValueError, "Metric name must be a non-empty string.")

        self._metrics[name] = metric_fn
        # Initialize history lists (will store aggregated results on rank 0)
        self._train_metrics_history[name] = []
        self._val_metrics_history[name] = []
        self.log(f"Added metric '{name}'.")

    def clear_metrics(self):
        """Removes all tracked metrics."""
        self._metrics.clear()
        self._train_metrics_history.clear()
        self._val_metrics_history.clear()
        self.log("All metrics cleared.")

    # --- History Properties  ---
    @property
    def train_losses(self) -> List[float]:
        """Returns the history of aggregated training losses. If DDP, broadcasts from rank 0 to all ranks."""
        return broadcast_if_ddp(self._train_losses, src=0)

    @property
    def val_losses(self) -> List[float]:
        """Returns the history of aggregated validation losses. If DDP, broadcasts from rank 0 to all ranks."""
        return broadcast_if_ddp(self._val_losses, src=0)

    @property
    def train_metrics_history(self) -> Dict[str, List[float]]:
        """Returns the history of aggregated training metrics. If DDP, broadcasts from rank 0 to all ranks."""
        return broadcast_if_ddp(dict(self._train_metrics_history), src=0)  # Return copy

    @property
    def val_metrics_history(self) -> Dict[str, List[float]]:
        """Returns the history of aggregated validation metrics. If DDP, broadcasts from rank 0 to all ranks."""
        return broadcast_if_ddp(dict(self._val_metrics_history), src=0)  # Return copy

    # --- Auto Saving Properties ---
    @property
    def save_interval(self) -> Optional[int]:
        return self._auto_saver.save_interval

    @save_interval.setter
    def save_interval(self, interval: Optional[int]):
        try:
            # Allow setting on all ranks, but saving only happens on rank 0
            self._auto_saver.save_interval = interval
            self.log(f"Auto-save interval set to {interval} epochs.")
        except (TypeError, ValueError) as e:
            self.raise_error(e, f"Failed to set save_interval: {e}")

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
                    self.log(f"Ensured auto-save directory exists: {path}")
                except OSError as e:
                    self.raise_error(ValueError,
                                     f"Save path '{path}' is not a valid directory and could not be created: {e}",
                                     e)

            # Set path on all ranks (state needs to be consistent if saved/loaded)
            self._auto_saver.save_path = path
            self.log(f"Auto-save path set to '{path}'.")
        except TypeError as e:
            self.raise_error(TypeError, f"Failed to set save_path: {e}", e)

    @property
    def save_model_name(self) -> str:
        return self._auto_saver.save_model_name

    @save_model_name.setter
    def save_model_name(self, name: str):
        try:
            # Set on all ranks for consistency
            self._auto_saver.save_model_name = name
            self.log(f"Auto-save model name format set to '{name}'.")
        except TypeError as e:
            self.raise_error(TypeError, f"Failed to set save_model_name: {e}", e)

    @property
    def overwrite_last_saved(self) -> bool:
        return self._auto_saver.overwrite_last_saved

    @overwrite_last_saved.setter
    def overwrite_last_saved(self, overwrite: bool):
        try:
            # Set on all ranks for consistency
            self._auto_saver.overwrite_last_saved = overwrite
            self.log(f"Auto-save overwrite set to {overwrite}.")
        except TypeError as e:
            self.raise_error(TypeError, f"Failed to set overwrite_last_saved: {e}", e)

    def auto_save(self, interval: Optional[int], save_path: str = '.', name: str = "model_epoch{epoch:02d}",
                  overwrite: bool = False):
        """Configures periodic model saving (saving performed by rank 0)."""
        try:
            # Configure settings on all ranks
            self.save_interval = interval
            self.save_path = save_path
            self.save_model_name = name
            self.overwrite_last_saved = overwrite

            if interval is None or interval == 0:
                self.log("Auto-save disabled.")
            else:
                self.log(
                    f"Auto-save configured: Interval={interval}, Path='{save_path}', Name='{name}', Overwrite={overwrite}")
        except (TypeError, ValueError) as e:
            self.warn(f"Failed to configure auto_save: {e}")

    # --- Callbacks ---
    @property
    def callbacks(self) -> List[Callback]:
        return self._callbacks

    def add_callback(self, callback: Callback):
        """Adds a callback instance to the handler (all ranks)."""
        if not isinstance(callback, Callback):
            self.raise_error(TypeError, "callback must be an instance of the Callback base class.")

        # Callbacks should ideally be DDP-aware internally if they perform rank-specific actions
        callback.set_handler(self)  # Link handler to callback
        self._callbacks.append(callback)
        self.log(f"Added callback: {type(callback).__name__}")

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

    # --- Training ---
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
        train(self, epochs, validate_every, gradient_accumulation_steps, use_amp, gradient_clipping_norm, ema_decay,
              seed, progress_bar, debug_print_interval, save_on_last_epoch, epoch_train_and_val_pbar)

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
        return list(set(keys))

    @on_rank(0)
    def save(self, path: str, single_file: bool = True):
        """Saves the complete handler state to a file (executed by rank 0 only).
           Other ranks wait at a barrier. State includes model, optimizer, scheduler,
           history (rank 0), EMA, scaler, config, etc.
        """
        if self._model is None:
            self.warn("Attempting to save handler state, but model is missing. Skipping save.", RuntimeWarning)
            # Ensure barrier is still hit even if save is skipped
            return

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
            "complimentary_kwargs": self._complimentary_kwargs,
            "ddp_kwargs": self._ddp_kwargs,
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
        if single_file:
            save_single_file(self, state, path)
        else:
            save_multi_files(self, state, path)

    @classmethod
    def load(cls,
             path: str,
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

        return load(cls, path, device, strict_load, skip_optimizer, skip_scheduler, skip_history, skip_callbacks,
                    skip_sampler_sde, skip_ema)

    @classmethod
    def initialize_from_checkpoint(cls,
                                   checkpoint_path: str,
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
        return initialize_from_checkpoint(cls, checkpoint_path, model_class, model_type, device, strict_load,
                                          **model_kwargs)

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
            self.raise_error(RuntimeError, "Model has not been set.")

        # Special handling for score-based models if __call__ should invoke score()
        # This assumes score() is the primary inference method for these models.
        if self._model_type == ModelType.SCORE_BASED:
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
        self.log(f"Freezing parameters for module: {module.__class__.__name__}")
        for param in module.parameters():
            param.requires_grad = False
        self.log(f"Module parameters frozen. Model now contains {self.count_parameters(True)} trainable parameters.")

    def eval(self, activate: bool = True, log: bool = True):
        """Sets model to evaluation or training mode (all ranks)."""
        if self._model is None:
            return
        if activate:
            self._model.eval()
            if log: self.log(f"Model set to eval() mode (Rank {self._rank}).")
        else:
            self._model.train()
            for module in self._modules_always_eval:
                module.eval()
            if log: self.log(f"Model set to train() mode (Rank {self._rank}).")

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
            self.warn(f"keep_eval_on_module: Module {module} is already in {self._modules_always_eval}.",
                      RuntimeWarning)

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
    @on_rank(0)
    def plot_losses(self, log_y_scale: bool = False, save_path: Optional[str] = None):
        """Plots training and validation losses (rank 0 only)."""

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

    @on_rank(0)
    def plot_metrics(self, log_y_scale: bool = False, save_path_prefix: Optional[str] = None):
        """Plots training and validation metrics (rank 0 only)."""
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
            self.warn(
                "torch.compile not available in this PyTorch version (requires 2.0+). Skipping compilation.",
                RuntimeWarning)
            return

        if self._model is None:
            self.raise_error(RuntimeError, "Model must be set before compiling.")

        if self._compiled_model:
            self.warn("Model is already compiled. Skipping recompilation.")
            return

        try:
            self.log(f"Compiling model with torch.compile (Rank {self._rank})... Options: {kwargs}")

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

            self.log(f"Model compiled successfully on Rank {self._rank} in {end_time - start_time:.2f} seconds.")

        except Exception as e:
            self.raise_error(Exception, f"Failed to compile model on Rank {self._rank}: {e}", e)

    # --- Score-Based Methods ---
    def score(self, t: Tensor, x: Tensor, *args) -> Tensor:
        """Computes the score function s(t, x) = ∇ log p_t(x)."""
        if self._model_type != ModelType.SCORE_BASED:
            self.raise_error(NotImplementedError, "Score function is only supported for SCORE_BASED models.")
        if self._sde is None:
            self.raise_error(RuntimeError, "SDE must be set to compute the score.")
        if self._model is None:
            self.raise_error(RuntimeError, "Model must be set to compute the score.")

        # Ensure t and x are tensors on the correct device for this rank
        # Convert time t to tensor if it's not already
        if not isinstance(t, Tensor):
            t = torch.tensor(t, device=self._device)
        # Ensure t is broadcastable to x's batch dimension
        t_dev = t.expand(x.shape[0]).to(self._device)
        x_dev = x.to(self._device)

        # Get model output using the underlying model
        # Pass through any additional args
        model_output = self._model(t_dev, x_dev, *args)

        # Get sigma(t) from SDE
        sigma_t = self._sde.sigma(t_dev)  # Assume sigma(t) returns tensor of shape (B,)

        # Ensure sigma_t has correct shape for broadcasting division (B, 1, 1, ...)
        _, *D_x = x_dev.shape  # Get spatial/feature dimensions
        sigma_t_reshaped = sigma_t.view(-1, *[1] * len(D_x))

        # Calculate score: model_output / sigma(t)
        # Add small epsilon for numerical stability if sigma can be zero
        score_val = model_output / (sigma_t_reshaped + 1e-8)

        return score_val

    def sample(self, shape: Tuple[int, ...], steps: int, corrector_steps: int = 0, condition: Optional[list] = None,
               likelihood_score_fn: Optional[Callable] = None, guidance_factor: float = 1.,
               apply_ema: bool = True, bar: bool = True, stop_on_NaN: bool = True, patch_size: int = None,
               stride: int = None, patch_chunk: int = None, corrector_snr: float = 0.1,
               on_step: Optional[Callable[[int, torch.Tensor, torch.Tensor], None]] = None) -> Optional[Tensor]:
        """
        Generates samples using a stochastic differential equation (SDE) solver. The method involves solving
        a sampling process over a defined number of steps, with optional parameters for additional conditions,
        guidance scaling, and optional configurations for patch-based operations. The returned sample may vary
        depending on the provided arguments and configurations.

        Args:
            shape (Tuple[int, ...]): The shape of the tensor to be sampled.
            steps (int): The number of sampling steps for the SDE solver to run.
            corrector_steps (int): The number of corrector steps to apply during sampling. Defaults to 0.
            condition (Optional[list]): Optional list of conditions to guide sampling. Defaults to None.
            likelihood_score_fn (Optional[Callable]): Optional callable likelihood score function to adjust
                the sampling process. Defaults to None.
            guidance_factor (float): A scalar factor for classifier-free guidance. Typically used to modulate
                the influence of conditions on the sampling process. Defaults to 1.0.
            apply_ema (bool): If True, applies exponential moving average (EMA) to model parameters during sampling.
                Defaults to True.
            bar (bool): If True, enables a progress bar for tracking the sampling process. Defaults to True.
            stop_on_NaN (bool): If True, halts the sampling process when NaN values are encountered. Defaults to True.
            patch_size (int): Size of each patch when using a patch-based sampling strategy. Defaults to None.
            stride (int): Stride between adjacent patches in the patch-based sampling approach. Defaults to None.
            patch_chunk (int): Number of patches to process simultaneously in a chunk during patch-based
                sampling. Defaults to None.
            corrector_snr (float): Signal-to-noise ratio (SNR) threshold used by the corrector. Defaults to 0.1.
            on_step (Optional[Callable[[int, torch.Tensor, torch.Tensor], None]]): Optional callable to be called after
                each step. Takes i, t, and x as an input.

        Returns:
            Optional[Tensor]: A tensor containing the generated samples, or None if sampling fails or conditions
            result in termination.
        """
        solver = SdeSolver(self)
        return solver.solve(shape, steps, corrector_steps, condition, likelihood_score_fn, guidance_factor, apply_ema,
                            bar, stop_on_NaN, patch_size, stride, patch_chunk, corrector_snr, on_step)
