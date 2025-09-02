from typing import Any, Dict
import math

import torch

from .batch_utils import _prepare_batch
from .loss_utils import _calculate_loss, _calculate_metrics
from ..utils import ModelType, autocast


def _train_step(nn_handler: 'NNHandler', batch: Any, current_epoch: int, accumulation_steps: int, use_amp: bool) -> \
        tuple[
            float, Dict[str, float]]:
    """Performs a single training step on the local batch (forward, loss, backward).
       Returns the local loss item (un-normalized) and local metrics dict.
    """
    if nn_handler._model is None or nn_handler._optimizer is None or nn_handler._loss_fn is None:
        raise RuntimeError("Model, optimizer, and loss function must be set for training.")

    nn_handler._model.train()  # Ensure model is in training mode

    # Prepare batch data for the current device
    _model_type = nn_handler._model_type
    batch_data = _prepare_batch(nn_handler, batch)
    inputs = batch_data["inputs"]
    targets = batch_data["targets"]
    additional_params = {k: v for k, v in batch_data.items() if k not in ["inputs", "targets"]}

    # Mixed Precision Context
    # autocast needs device type ('cuda' or 'cpu')
    with autocast(device_type=nn_handler._device.type, enabled=use_amp):
        # Forward pass - use self.model to handle DDP/DP wrapping automatically
        if _model_type == ModelType.SCORE_BASED:
            # Score-based loss often calls the model internally
            model_output = None  # Placeholder
        else:
            model_output = nn_handler.model(inputs, **additional_params)

        # Loss calculation
        loss_val = _calculate_loss(nn_handler, model_output, targets, inputs, current_epoch)

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
            nn_handler.warn(
                f"NaN or Inf detected in training loss (Batch). Skipping backward pass for this batch.",
                RuntimeWarning)
            # Return NaN loss and empty metrics to indicate skip
            return torch.nan, {}

        # Normalize loss for gradient accumulation *before* scaling/backward
        loss = loss / accumulation_steps

    # Backward pass & Gradient Scaling (if AMP)
    # DDP automatically synchronizes gradients during backward
    nn_handler._grad_scaler.scale(loss).backward()

    # Metrics calculation (using local model_output and targets)
    # Skip metrics if score-based (or if model_output is None)
    batch_metrics = {}
    if _model_type != ModelType.SCORE_BASED and model_output is not None:
        batch_metrics = _calculate_metrics(nn_handler, model_output, targets)

    # Return the un-normalized loss item for accumulation and local metrics
    # Multiply by accumulation_steps to get the effective loss for this batch before normalization
    return loss.item() * accumulation_steps, batch_metrics


def _val_step(nn_handler: 'NNHandler', batch: Any, current_epoch: int) -> tuple[float, Dict[str, float]]:
    """Performs a single validation step on the local batch.
       Returns the local loss item and local metrics dict.
    """
    if nn_handler._model is None or nn_handler._loss_fn is None:
        raise RuntimeError("Model and loss function must be set for validation.")

    nn_handler.eval(activate=True, log=False)  # Ensure model is in evaluation mode

    _model_type = nn_handler._model_type
    batch_data = _prepare_batch(nn_handler, batch)
    inputs = batch_data["inputs"]
    targets = batch_data["targets"]
    additional_params = {k: v for k, v in batch_data.items() if k not in ["inputs", "targets"]}

    with torch.no_grad():  # No gradients needed for validation
        # Forward pass (similar to train step)
        if _model_type == ModelType.SCORE_BASED:
            model_output = None  # Loss handles model call
        else:
            # Use self.model for forward pass (handles DDP/DP wrapper)
            model_output = nn_handler.model(inputs, **additional_params)

        # Loss calculation
        loss_val = _calculate_loss(nn_handler, model_output, targets, inputs, current_epoch)
        # Handle tuple loss
        if isinstance(loss_val, tuple):
            loss = loss_val[0]
        else:
            loss = loss_val

        # Metrics calculation
        batch_metrics = {}
        if _model_type != ModelType.SCORE_BASED and model_output is not None:
            batch_metrics = _calculate_metrics(nn_handler, model_output, targets)

    # Return loss item and metrics dict
    # Handle potential NaN/Inf in validation loss
    loss_item = loss.item() if not (torch.isnan(loss).any() or torch.isinf(loss).any()) else math.nan
    return loss_item, batch_metrics
