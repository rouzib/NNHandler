import contextlib
import logging
from collections import defaultdict
import time
from typing import Optional, Dict, Union
import math

import torch
import torch.distributed as dist
from torch import Tensor
from tqdm.auto import tqdm

from .steps import _train_step, _val_step
from ..checkpointing.trainer_saving import auto_save_epoch
from ..utils import _amp_available, _ema_available, ExponentialMovingAverage, GradScaler, aggregate_loss, \
    aggregate_metrics


def train(nn_handler: 'NNHandler',
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
    # --- Pre-Training Checks ---
    _optimizer = nn_handler._optimizer
    _train_loader = nn_handler._train_loader
    _val_loader = nn_handler._val_loader
    _device = nn_handler._device
    _is_distributed = nn_handler._distributed

    if nn_handler._model is None or _optimizer is None or _train_loader is None or nn_handler._loss_fn is None:
        nn_handler.raise_error(RuntimeError,
                               f"Model, optimizer, training loader, and loss function must be set before training.")
    if (validate_every is not None and validate_every > 0) and _val_loader is None:
        nn_handler.raise_error(ValueError,
                               "Validation requested (validate_every > 0), but validation loader is not set.")
    if use_amp and not _amp_available:
        nn_handler.warn("AMP requested but torch.amp not available. Disabling AMP.", RuntimeWarning)
    if use_amp and _device.type != 'cuda':
        nn_handler.warn("AMP requested but device is not CUDA. Disabling AMP.", RuntimeWarning)
    if ema_decay > 0 and not _ema_available:
        nn_handler.warn("EMA requested but torch_ema not available. Disabling EMA.", RuntimeWarning)

    # Synchronize after checks to ensure all ranks proceed or fail together
    if _is_distributed:
        dist.barrier()

    # Determine effective settings after warnings
    effective_use_amp = use_amp and _amp_available and _device.type == 'cuda'
    effective_ema_decay = ema_decay if _ema_available and ema_decay > 0 else 0.0

    # Apply seed if provided (applies on all ranks via setter)
    if seed is not None:
        nn_handler.seed = seed  # Use setter

    start_epoch = len(nn_handler._train_losses)  # Resume from last epoch (based on rank 0 history)
    total_epochs = start_epoch + epochs

    # --- Setup (Logging, EMA, AMP Scaler) ---
    nn_handler.log(f"--- Starting Training Run ---")
    nn_handler.log(f"  Epochs:              {start_epoch + 1} -> {total_epochs}")
    nn_handler.log(f"  Distributed:         {_is_distributed} (World Size: {nn_handler._world_size})")
    nn_handler.log(f"  AMP Enabled:         {effective_use_amp}")
    nn_handler.log(f"  EMA Enabled:         {effective_ema_decay > 0} (Decay: {effective_ema_decay:.4f})")
    nn_handler.log(f"  Grad Accumulation:   {gradient_accumulation_steps}")
    nn_handler.log(f"  Grad Clipping Norm:  {gradient_clipping_norm}")
    nn_handler.log(f"  Validate Every:      {validate_every}")
    nn_handler.log(f"  Seed:                {nn_handler._seed}")

    # Initialize EMA if needed (on all ranks, state loaded later if applicable)
    nn_handler._ema = None
    nn_handler._ema_decay = effective_ema_decay
    if nn_handler._ema_decay > 0:
        try:
            # Pass the parameters of the potentially wrapped model
            nn_handler._ema = ExponentialMovingAverage(nn_handler.model.parameters(), decay=nn_handler._ema_decay)
            # Load EMA state if resuming (handled in load method)
            nn_handler.log(f"Initialized Exponential Moving Average with decay {nn_handler._ema_decay}.")
        except Exception as e:
            nn_handler.log(f"Failed to initialize EMA: {e}. Disabling EMA.", logging.ERROR)
            nn_handler._ema = None
            nn_handler._ema_decay = 0.0

    # Initialize GradScaler for AMP (on all ranks, state loaded later)
    nn_handler._grad_scaler = GradScaler(enabled=effective_use_amp)
    if effective_use_amp:  # Log on rank 0
        nn_handler.log("Automatic Mixed Precision (AMP) GradScaler enabled.")

    # Progress bar setup (only on rank 0)
    # Outer loop pbar
    pbar_outer = None
    if progress_bar and nn_handler._rank == 0:
        pbar_outer = tqdm(range(start_epoch, total_epochs), desc="Epochs", unit="epoch", dynamic_ncols=True)

    # --- Callback: on_train_begin ---
    nn_handler._stop_training = False  # Reset stop flag
    train_begin_logs = {'start_epoch': start_epoch, 'total_epochs': total_epochs, 'world_size': nn_handler._world_size}
    # Run on all ranks, callbacks should handle rank internally if needed
    nn_handler._run_callbacks('on_train_begin', logs=train_begin_logs)

    # =========================== Main Training Loop ===========================
    # Initialize variable for final epoch logs outside the loop
    final_epoch_logs_agg = {}

    for epoch in range(start_epoch, total_epochs):
        epoch_start_time = time.time()
        current_epoch_1_based = epoch + 1
        # Logs for this epoch, start empty, populated by train/val/callbacks
        # Aggregated logs will be stored on rank 0
        epoch_logs: Dict[str, Union[int, float]] = {'epoch': current_epoch_1_based}

        # --- Set Sampler Epoch (Important for DDP reproducibility) ---
        if _is_distributed:
            if nn_handler._train_sampler:
                nn_handler._train_sampler.set_epoch(epoch)
            if nn_handler._val_sampler:
                # Set epoch for val sampler too, important if shuffle=True for val
                nn_handler._val_sampler.set_epoch(epoch)

        # --- Callback: on_epoch_begin ---
        # Run on all ranks
        nn_handler._run_callbacks('on_epoch_begin', epoch=epoch, logs=epoch_logs)

        # ================== Training Phase ==================
        nn_handler.eval(activate=False, log=False)  # Set model to train mode
        # Accumulators for local results on this rank
        train_loss_accum_local = 0.0
        train_metrics_accum_local = defaultdict(float)
        train_batches_processed_local = 0  # Count successful batches
        batches_in_epoch = 0  # Determine batches per rank
        if _train_loader:
            try:
                batches_in_epoch = len(_train_loader)
            except TypeError:
                # Handle iterable datasets with no __len__
                batches_in_epoch = -1  # Indicate unknown length
                if epoch == start_epoch:  # Log once
                    nn_handler.warn("Train DataLoader has no length. Inner progress bar may be inaccurate.")

        # Setup inner progress bar for training (rank 0 only)
        train_iterator = enumerate(_train_loader) if _train_loader else []
        pbar_inner_train = None
        if epoch_train_and_val_pbar and nn_handler._rank == 0 and batches_in_epoch > 0:
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
            nn_handler._run_callbacks('on_train_batch_begin', batch=batch_idx, logs=batch_logs)

            # --- Train Step (Forward, Loss, Backward) ---
            sync_ctx = contextlib.nullcontext() # skip mGPU sync for gradient accumulation steps > 1
            if gradient_accumulation_steps > 1:
                sync_ctx = nn_handler._model.no_sync() if nn_handler._distributed else contextlib.nullcontext()
            with sync_ctx:
                local_loss_item, local_metrics_items = _train_step(nn_handler, batch_data, epoch,
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
                        nn_handler._grad_scaler.unscale_(_optimizer)  # Unscale inplace before clipping
                    # Clip gradients of the model parameters
                    torch.nn.utils.clip_grad_norm_(nn_handler.model.parameters(), max_norm=gradient_clipping_norm)

                # Optimizer Step (using scaler if AMP enabled)
                nn_handler._grad_scaler.step(_optimizer)
                # Update GradScaler's scale for next iteration
                nn_handler._grad_scaler.update()
                # Zero gradients *after* stepping scaler and optimizer
                _optimizer.zero_grad()

                # EMA Update (after optimizer step)
                if nn_handler._ema:
                    nn_handler._ema.update()

            # --- Callbacks & Debug Logging ---
            # Add local results to batch logs for callbacks
            batch_logs['loss'] = local_loss_item  # Local loss for this batch
            batch_logs.update(local_metrics_items)  # Local metrics for this batch
            # --- Callback: on_train_batch_end --- (All ranks)
            nn_handler._run_callbacks('on_train_batch_end', batch=batch_idx, logs=batch_logs)

            # Debug print local info (rank 0 only)
            if nn_handler._rank == 0 and debug_print_interval and (batch_idx + 1) % debug_print_interval == 0:
                debug_str = f"[R{nn_handler._rank} E{current_epoch_1_based} B{batch_idx + 1}] Local L:{local_loss_item:.3e}"
                debug_str += " Metrics: " + " ".join([f"{k}:{v:.3f}" for k, v in local_metrics_items.items()])
                nn_handler.log(debug_str, logging.DEBUG)
                # Update inner pbar postfix if exists
                if pbar_inner_train:
                    pbar_inner_train.set_postfix_str(f"Local: {debug_str}", refresh=False)

        # Close inner training progress bar if used
        if pbar_inner_train:
            pbar_inner_train.close()

        # --- Aggregate Training Results Across Ranks ---
        # Ensure all ranks finish local training loop
        if _is_distributed:
            dist.barrier()

        # Calculate average local loss and metrics
        avg_train_loss_local = train_loss_accum_local / train_batches_processed_local if train_batches_processed_local > 0 else math.nan
        avg_train_metrics_local = {
            name: total / train_batches_processed_local
            for name, total in train_metrics_accum_local.items()
        } if train_batches_processed_local > 0 else {name: math.nan for name in nn_handler._metrics}

        # Aggregate averages across ranks
        avg_train_loss_agg = aggregate_loss(avg_train_loss_local, nn_handler._world_size, _device)
        avg_train_metrics_agg = aggregate_metrics(avg_train_metrics_local, nn_handler._world_size, _device)

        # Store aggregated results on Rank 0
        if nn_handler._rank == 0:
            nn_handler._train_losses.append(avg_train_loss_agg)
            epoch_logs['loss'] = avg_train_loss_agg  # Add aggregated loss to epoch logs
            for name, value in avg_train_metrics_agg.items():
                nn_handler._train_metrics_history[name].append(value)
                epoch_logs[name] = value  # Add aggregated train metrics

            train_log_msg = f"Epoch {current_epoch_1_based} Train Aggregated: Loss={avg_train_loss_agg:.4e}"
            train_log_msg += " Metrics: " + ", ".join(
                [f"{k}={v:.4f}" for k, v in avg_train_metrics_agg.items()])
            nn_handler.log(train_log_msg, logging.DEBUG)

        # ================= Validation Phase ==================
        run_validation = (
                validate_every is not None and validate_every > 0 and current_epoch_1_based % validate_every == 0)
        avg_val_loss_agg = math.nan  # Initialize aggregated results
        avg_val_metrics_agg = {name: math.nan for name in nn_handler._metrics}

        if run_validation:
            # --- Callback: on_val_begin --- (All ranks)
            nn_handler._run_callbacks('on_val_begin', logs=epoch_logs)  # Pass current epoch logs

            # Accumulators for local validation results
            val_loss_accum_local = 0.0
            val_metrics_accum_local = defaultdict(float)
            val_batches_processed_local = 0
            val_batches_in_epoch = 0
            if _val_loader:
                try:
                    val_batches_in_epoch = len(_val_loader)
                except TypeError:
                    val_batches_in_epoch = -1  # Unknown length
                    if epoch == start_epoch:
                        nn_handler.warn(
                            "Validation DataLoader has no length. Inner progress bar may be inaccurate.")

            # Setup inner progress bar for validation (rank 0 only)
            val_iterator = enumerate(_val_loader) if _val_loader else []
            pbar_inner_val = None
            if epoch_train_and_val_pbar and nn_handler._rank == 0 and val_batches_in_epoch > 0:
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
            ema_context = nn_handler._ema.average_parameters() if nn_handler._ema else contextlib.nullcontext()
            with ema_context:
                nn_handler.eval(activate=True, log=False)  # Ensure eval mode inside EMA context if needed
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
                    nn_handler._run_callbacks('on_val_batch_begin', batch=val_batch_idx, logs=batch_logs)

                    # --- Val Step ---
                    local_loss_item, local_metrics_items = _val_step(nn_handler, val_batch_data, epoch)

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
                    nn_handler._run_callbacks('on_val_batch_end', batch=val_batch_idx, logs=batch_logs)

                    # Update inner pbar postfix if exists (rank 0)
                    if pbar_inner_val:
                        debug_str = f"Local L:{local_loss_item:.3e} "
                        debug_str += " ".join([f"{k}:{v:.3f}" for k, v in local_metrics_items.items()])
                        pbar_inner_val.set_postfix_str(debug_str, refresh=False)

            # Close inner validation progress bar if used
            if pbar_inner_val:
                pbar_inner_val.close()

            # --- Aggregate Validation Results Across Ranks ---
            if _is_distributed:
                dist.barrier()

            # Calculate average local validation results
            avg_val_loss_local = val_loss_accum_local / val_batches_processed_local if val_batches_processed_local > 0 else math.nan
            avg_val_metrics_local = {
                name: total / val_batches_processed_local
                for name, total in val_metrics_accum_local.items()
            } if val_batches_processed_local > 0 else {name: math.nan for name in nn_handler._metrics}

            # Aggregate averages across ranks
            avg_val_loss_agg = aggregate_loss(avg_val_loss_local, nn_handler._world_size, _device)
            avg_val_metrics_agg = aggregate_metrics(avg_val_metrics_local, nn_handler._world_size, _device)

            # Store aggregated results on Rank 0
            if nn_handler._rank == 0:
                nn_handler._val_losses.append(avg_val_loss_agg)
                epoch_logs['val_loss'] = avg_val_loss_agg  # Add aggregated val loss
                for name, value in avg_val_metrics_agg.items():
                    nn_handler._val_metrics_history[name].append(value)
                    epoch_logs[f'val_{name}'] = value  # Add aggregated val metrics with prefix

                val_log_msg = f"Epoch {current_epoch_1_based} Val Aggregated: Loss={avg_val_loss_agg:.4e}"
                val_log_msg += " Metrics: " + ", ".join(
                    [f"{k}={v:.4f}" for k, v in avg_val_metrics_agg.items()])
                nn_handler.log(val_log_msg, logging.DEBUG)

            # --- Callback: on_val_end --- (All ranks, logs contain aggregated results if rank 0)
            # Need to broadcast logs from rank 0 if callbacks on other ranks need aggregated results
            logs_list_for_broadcast = [epoch_logs if nn_handler._rank == 0 else None]
            if _is_distributed:
                dist.broadcast_object_list(logs_list_for_broadcast, src=0)
            received_val_end_logs = logs_list_for_broadcast[0]
            nn_handler._run_callbacks('on_val_end', logs=received_val_end_logs)

        elif nn_handler._rank == 0:  # No validation run, append NaN on rank 0 for consistent history length
            nn_handler._val_losses.append(math.nan)
            epoch_logs['val_loss'] = math.nan
            for name in nn_handler._metrics.keys():
                nn_handler._val_metrics_history[name].append(math.nan)
                epoch_logs[f'val_{name}'] = math.nan

        # --- Scheduler Step ---
        # Step scheduler on all ranks based on aggregated metric if needed
        if nn_handler._scheduler:
            if isinstance(nn_handler._scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # Requires aggregated validation metric (use avg_val_loss_agg)
                if run_validation:  # Only step if validation ran
                    # Ensure the aggregated loss is valid before stepping
                    if not math.isnan(avg_val_loss_agg):
                        nn_handler._scheduler.step(avg_val_loss_agg)
                        # Log LR change possibility on rank 0
                        if epoch == start_epoch:  # Log once
                            nn_handler.log(
                                f"Stepped ReduceLROnPlateau scheduler with aggregated val_loss: {avg_val_loss_agg:.4e}",
                                logging.DEBUG)
                    elif nn_handler._rank == 0:  # Warn on rank 0 if metric is NaN
                        nn_handler.warn(
                            f"Epoch {current_epoch_1_based}: ReduceLROnPlateau requires a valid aggregated validation metric (e.g., val_loss) "
                            "to step, but received NaN. Scheduler not stepped.", RuntimeWarning)
                # else: Do nothing if validation didn't run
            else:
                # Standard schedulers usually step every epoch
                nn_handler._scheduler.step()
                # Log scheduler step debug on rank 0 (once)
                if epoch == start_epoch:
                    nn_handler.log(f"Stepped scheduler {type(nn_handler._scheduler).__name__}.", logging.DEBUG)

        # --- Log LR & Epoch Summary (Rank 0) ---
        epoch_time = time.time() - epoch_start_time
        # Store epoch time on rank 0 logs before broadcasting
        if nn_handler._rank == 0:
            epoch_logs['epoch_time'] = epoch_time
            # Add current learning rate(s) to logs (rank 0)
            if _optimizer:
                for i, pg in enumerate(_optimizer.param_groups):
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

            nn_handler.log(log_msg)
            # Update outer progress bar postfix (rank 0)
            if pbar_outer:
                # Extract summary part for postfix
                summary_postfix = log_msg[log_msg.find("Train Loss:"):]
                pbar_outer.set_postfix_str(summary_postfix)

        # --- Callback: on_epoch_end ---
        # Broadcast logs from Rank 0 to all other ranks so callbacks have consistent info
        logs_list_broadcast = [epoch_logs if nn_handler._rank == 0 else None]
        if _is_distributed:
            dist.broadcast_object_list(logs_list_broadcast, src=0)
        # All ranks receive the aggregated logs from rank 0
        final_epoch_logs_agg = logs_list_broadcast[0]

        # Run on_epoch_end on all ranks with the *same* aggregated logs
        nn_handler._run_callbacks('on_epoch_end', epoch=epoch, logs=final_epoch_logs_agg)

        # --- Auto Saving (Rank 0 Only) ---
        # Pass aggregated logs for filename formatting
        auto_save_epoch(nn_handler, epoch, total_epochs, save_on_last_epoch, final_epoch_logs_agg)

        # --- Check for Early Stopping Signal ---
        # The flag _stop_training might be set by a callback (e.g., EarlyStopping on rank 0)
        # We need to broadcast this decision from rank 0 to all ranks
        stop_tensor = torch.tensor(int(nn_handler._stop_training), device=_device, dtype=torch.int)
        if _is_distributed:
            dist.broadcast(stop_tensor, src=0)  # Broadcast the decision from rank 0
        nn_handler._stop_training = bool(stop_tensor.item())  # Update flag on all ranks

        if nn_handler._stop_training:
            nn_handler.log(f"Early stopping triggered after epoch {current_epoch_1_based}.")
            # Ensure all ranks know about stopping before breaking
            if _is_distributed:
                dist.barrier()
            break  # Exit the main training loop

        # Update outer progress bar (rank 0)
        if pbar_outer:
            pbar_outer.update(1)

        # Final barrier at end of epoch loop for synchronization
        if _is_distributed:
            dist.barrier()

    # =========================== End of Training ===========================
    if pbar_outer:
        pbar_outer.close()

    # Apply final EMA weights if used (on all ranks)
    if nn_handler._ema:
        nn_handler.log("Applying final EMA weights to the model.")
        try:
            # Ensure EMA update happens on the parameters of the wrapped model
            nn_handler._ema.copy_to(nn_handler.model.parameters())
        except Exception as e:
            nn_handler.log(f"Failed to apply final EMA weights: {e}", logging.ERROR)

    # --- Callback: on_train_end ---
    # Pass the final aggregated logs from the last completed epoch
    final_logs = {'final_epoch': epoch + 1, 'world_size': nn_handler._world_size}
    # final_epoch_logs_agg should contain the aggregated logs from the last epoch
    final_logs.update(final_epoch_logs_agg if final_epoch_logs_agg else {})
    # Run on all ranks with consistent final logs
    nn_handler._run_callbacks('on_train_end', logs=final_logs)

    nn_handler.log("--- Training Run Finished ---")

    # Final barrier to ensure all processes finish cleanly
    if _is_distributed:
        dist.barrier()
