import logging
import os
import warnings
from collections import defaultdict
from typing import Optional, Union, OrderedDict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
from typing_extensions import deprecated

from ..utils import _resolve_device, ModelType, _amp_available, _ema_available, ExponentialMovingAverage


def load(NNHandler,
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
        state = torch.load(path, map_location=map_location, weights_only=False)  # Load onto the target device map
    except Exception as e:
        # Add rank info to error
        raise RuntimeError(
            f"Rank {current_rank}: Failed to load checkpoint from {path} using map_location '{map_location}': {e}") from e

    # --- Extract Configuration Needed for Handler Init ---
    model_class = state.get("model_class")
    model_kwargs = state.get("model_kwargs", {})
    model_type_str = state.get("model_type", ModelType.CLASSIFICATION.value)  # Get string value
    try:
        # Convert saved string back to enum member
        model_type = ModelType(model_type_str)
    except ValueError:
        # Handle case where saved value is invalid
        warnings.warn(
            f"Rank {current_rank}: Invalid model_type '{model_type_str}' found in checkpoint. Defaulting to CLASSIFICATION.",
            RuntimeWarning)
        model_type = ModelType.CLASSIFICATION

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
        handler.log(" Stripped 'module.' prefix from saved model state_dict for loading into non-parallel model.",
                    logging.DEBUG)
    elif not saved_parallel and current_parallel:
        # Saved raw, loading into wrapped model -> add 'module.'
        load_state_dict = OrderedDict(('module.' + k, v) for k, v in model_state_dict.items())
        handler.log(" Added 'module.' prefix to saved model state_dict for loading into parallel model.", logging.DEBUG)
    # Else: keys match (both parallel or both raw), use load_state_dict as is

    # Load into the underlying module
    try:
        missing_keys, unexpected_keys = handler.module.load_state_dict(load_state_dict, strict=strict_load)
        if missing_keys: handler.warn(
            f" Missing keys when loading model state_dict: {missing_keys}")
        if unexpected_keys: handler.warn(
            f" Unexpected keys when loading model state_dict: {unexpected_keys}")
        handler.log(" Model state_dict loaded successfully.")
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
            handler.log(" Optimizer state loaded.")
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
            handler.log(" Scheduler state loaded.")
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
            handler.log(f" GradScaler state loaded {enabled_info}.")
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
            handler.log(" Training history loaded (rank 0).")
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
            handler.log(" AutoSaver state loaded.")
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
            handler.log(" EMA state loaded.")
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
                handler.log(" SDE re-initialized from checkpoint info.")
            except Exception as e:
                warnings.warn(f"Rank {current_rank}: Failed to load/re-init SDE: {e}", RuntimeWarning)
        if sampler_class:
            try:
                handler.set_sampler(sampler_class, **sampler_kwargs)  # Re-initializes Sampler
                # Load sampler state if available
                if sampler_state and handler._sampler and hasattr(handler._sampler, 'load'):
                    handler._sampler.load(sampler_state)  # Use sampler's load method
                handler.log(" Sampler re-initialized and state loaded from checkpoint.")
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
                        handler.log(f" Callback '{cb_name}' state loaded.")
                        # Log loaded state on rank 0 only
                        handler.log(f" Loaded state for callback '{cb_name}'.", logging.DEBUG)
                    except Exception as e:
                        warnings.warn(f"Rank {current_rank}: Failed to load state for callback '{cb_name}': {e}",
                                      RuntimeWarning)

            # Warn about unmatched states (rank 0 only)
            if handler._rank == 0:
                unmatched_states = set(callback_states.keys()) - loaded_cb_names
                if unmatched_states:
                    handler.warn(
                        f"Callback states found in checkpoint but not loaded (no matching callback class existed in the saved file): {unmatched_states}",
                        RuntimeWarning)
        elif callback_states and handler._rank == 0:  # Check if states exist but no callbacks added
            handler.warn(
                "Callback states found in checkpoint, but no callbacks class currently saved in the handler instance. States not loaded.",
                RuntimeWarning)

    # Other config loaded on all ranks
    handler._seed = state.get("seed")  # Restore seed used for the original training run
    handler._train_loader_kwargs = state.get("train_loader_kwargs", {})  # Store for reference/re-creation
    handler._val_loader_kwargs = state.get("val_loader_kwargs", {})

    handler.log(f"NNHandler loaded successfully by Rank {current_rank} from: {os.path.basename(path)}")

    # Barrier to ensure all ranks finish loading before returning control
    if is_distributed_load:
        handler.log(f"Rank {current_rank} waiting at load barrier.", logging.DEBUG)
        dist.barrier()
        handler.log(f"Rank {current_rank} passed load barrier.", logging.DEBUG)

    return handler


@deprecated("Use load() instead")
def initialize(cls, **kwargs):
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
    mt = kwargs.pop('model_type', ModelType.CLASSIFICATION)
    # DDP flag not handled by this deprecated method
    mkw = kwargs.pop('model_kwargs', {})

    # Instantiate handler (will auto-detect DDP if env vars set)
    h = cls(mc, device=dev, logger_mode=logm, logger_filename=logf, logger_level=logl,
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


def initialize_from_checkpoint(NNHandler,
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
    eff_model_type = model_type if model_type is not None else ModelType.CLASSIFICATION
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
        if missing_keys: handler.warn(f" Missing keys when loading weights: {missing_keys}")
        if unexpected_keys: handler.warn(
            f" Unexpected keys when loading weights: {unexpected_keys}")
        handler.log(f"Model weights loaded successfully from {os.path.basename(checkpoint_path)}.")
        handler.warn(
            "Initialized from weights checkpoint: Optimizer, scheduler, history, etc., are NOT loaded.")
    except Exception as e:
        raise RuntimeError(
            f"Rank {current_rank}: Failed to load model weights from checkpoint state_dict: {e}") from e

    # Barrier to ensure all ranks finish loading weights
    if is_distributed_load:
        dist.barrier()

    return handler
