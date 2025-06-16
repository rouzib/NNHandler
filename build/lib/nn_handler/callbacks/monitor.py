import warnings
import time
from typing import Any, Dict, Optional, Union

import torch

from .base import Callback


class LearningRateMonitor(Callback):
    """Logs the learning rate(s) used by the optimizer."""

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        logs = logs or {}
        if self.handler and self.handler.optimizer:
            for i, param_group in enumerate(self.handler.optimizer.param_groups):
                lr = param_group['lr']
                logs[f'lr_group_{i}'] = lr
                if i == 0: logs['lr'] = lr  # Common case alias
        # Logs are handled by other callbacks (e.g., TensorBoardLogger) or printed by handler


try:
    from tensorboardX import SummaryWriter

    _tensorboard_available = True
except ImportError:
    _tensorboard_available = False


    # Dummy SummaryWriter
    class SummaryWriter:
        def __init__(self, log_dir=None, comment='', **kwargs): pass

        def add_scalar(self, tag, scalar_value, global_step=None, walltime=None): pass

        def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None): pass

        def close(self): pass


class TensorBoardLogger(Callback):
    """Callback that streams epoch results to TensorBoard."""

    def __init__(self, log_dir: str = './logs', comment: str = ''):
        super().__init__()
        if not _tensorboard_available:
            raise ImportError("TensorBoardLogger requires tensorboardX. Install with 'pip install tensorboardX'")
        self.log_dir = log_dir
        self.comment = comment
        self.writer: Optional[SummaryWriter] = None

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        self.writer = SummaryWriter(log_dir=self.log_dir, comment=self.comment)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        logs = logs or {}
        if self.writer:
            for name, value in logs.items():
                if isinstance(value, (int, float, torch.Tensor)):
                    self.writer.add_scalar(name, value, epoch + 1)  # TensorBoard uses 1-based steps

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        if self.writer:
            self.writer.close()
            self.writer = None

    # No state needed for this callback


# --- Weights & Biases Logger ---
try:
    import wandb

    _wandb_available = True
except ImportError:
    _wandb_available = False
    wandb = None  # Define wandb as None if not installed


class WandbLogger(Callback):
    """Logs metrics, parameters, and optionally model topology/gradients to Weights & Biases.

    Args:
        project (str): The name of the W&B project.
        entity (Optional[str]): The W&B entity (username or team). If None, uses default.
        run_name (Optional[str]): A specific name for this run. If None, W&B generates one.
        config (Optional[Dict[str, Any]]): A dictionary of hyperparameters to log.
            If None, tries to automatically log handler's model_kwargs, optimizer_kwargs, etc.
        log_freq_epoch (int): Log metrics this often (in epochs). Defaults to 1 (every epoch).
        log_model_topology (bool): If True, logs the model graph to W&B. Defaults to True.
        log_model_weights (bool): If True, logs model weights periodically (can be large). Defaults to False.
        log_weight_freq_epoch (int): Frequency (in epochs) to log weights if log_model_weights is True.
        # log_gradients (bool): If True, logs gradient norms periodically (requires specific hook point). Defaults to False. (Currently difficult to implement reliably with existing hooks)
    """

    def __init__(self,
                 project: str,
                 entity: Optional[str] = None,
                 run_name: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None,
                 log_freq_epoch: int = 1,
                 log_model_topology: bool = True,
                 log_model_weights: bool = False,
                 log_weight_freq_epoch: int = 10):
        # log_gradients: bool = False): # Gradient logging disabled for now
        super().__init__()
        if not _wandb_available:
            raise ImportError("WandbLogger requires the 'wandb' library. Install with 'pip install wandb'")

        self.project = project
        self.entity = entity
        self.run_name = run_name
        self.config = config
        self.log_freq_epoch = log_freq_epoch
        self.log_model_topology = log_model_topology
        self.log_model_weights = log_model_weights
        self.log_weight_freq_epoch = log_weight_freq_epoch
        # self.log_gradients = log_gradients # Disabled

        self.run: Optional[wandb.sdk.wandb_run.Run] = None
        self._run_id: Optional[str] = None  # For resuming

    def _get_handler_config(self) -> Dict[str, Any]:
        """Extracts relevant configuration from the handler."""
        if not self.handler: return {}
        cfg = {}
        cfg.update(self.handler._model_kwargs or {})
        cfg.update({'model_class': self.handler._model_class.__name__ if self.handler._model_class else None})
        cfg.update({'model_type': self.handler._model_type.name if self.handler._model_type else None})
        cfg.update(self.handler._optimizer_kwargs or {})
        cfg.update({'optimizer_class': self.handler._optimizer.__class__.__name__ if self.handler._optimizer else None})
        cfg.update(self.handler._scheduler_kwargs or {})
        cfg.update({'scheduler_class': self.handler._scheduler.__class__.__name__ if self.handler._scheduler else None})
        cfg.update({'loss_fn': self.handler._loss_fn.__name__ if self.handler._loss_fn else None})
        cfg.update(self.handler._loss_fn_kwargs or {})
        cfg.update({'device': str(self.handler.device)})
        cfg.update({'seed': self.handler.seed})
        # Add training args if available (not directly stored in handler, maybe pass manually)
        # cfg.update({'epochs': total_epochs_from_handler?})
        return cfg

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        config_to_log = self.config if self.config is not None else self._get_handler_config()

        try:
            if self.run is None:  # Avoid re-initializing if already loaded/resumed
                self.run = wandb.init(
                    project=self.project,
                    entity=self.entity,
                    name=self.run_name,
                    config=config_to_log,
                    id=self._run_id,  # Use stored ID if resuming
                    resume="allow"  # Allow resuming if run_id exists
                )
                self._run_id = self.run.id  # Store the run ID after init
                print(f"W&B Run initialized: {self.run.get_url()}")

            # Log model topology (graph) if requested and model exists
            if self.log_model_topology and self.handler and self.handler.model:
                try:
                    # wandb.watch might fail with DataParallel or complex models
                    # Using handler.module to get unwrapped model
                    wandb.watch(self.handler.module, log="all", log_freq=100)  # Log gradients/weights less freq
                    print("W&B watching model topology.")
                except Exception as e:
                    warnings.warn(f"wandb.watch failed: {e}. Model topology might not be logged.", RuntimeWarning)

        except Exception as e:
            warnings.warn(f"Failed to initialize W&B run: {e}", RuntimeWarning)
            self.run = None  # Ensure run is None if init fails

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        if not self.run: return  # Don't log if run init failed
        logs = logs or {}
        current_epoch_1_based = epoch + 1

        if current_epoch_1_based % self.log_freq_epoch == 0:
            # Prepare metrics for logging (remove non-scalar if any)
            metrics_to_log = {k: v for k, v in logs.items() if isinstance(v, (int, float, torch.Tensor))}
            # Make sure epoch is included for x-axis
            metrics_to_log['epoch'] = current_epoch_1_based

            try:
                self.run.log(metrics_to_log, step=current_epoch_1_based)
            except Exception as e:
                warnings.warn(f"W&B logging failed at epoch {current_epoch_1_based}: {e}", RuntimeWarning)

        # Log model weights periodically
        if self.log_model_weights and (current_epoch_1_based % self.log_weight_freq_epoch == 0):
            # This can be very slow and generate large artifacts
            # Consider logging only specific layers or norms instead
            warnings.warn(f"Logging model weights at epoch {current_epoch_1_based}. This can be slow/large.",
                          UserWarning)
            try:
                # Log state_dict as artifact? Or directly? Check wandb docs.
                # For simplicity, log norms or simple stats
                if self.handler and self.handler.model:
                    weight_norm = torch.norm(
                        torch.stack([torch.norm(p.data.detach(), 2) for p in self.handler.module.parameters()]), 2)
                    self.run.log({'model_weight_L2_norm': weight_norm.item()}, step=current_epoch_1_based)
                # To log the full model:
                # model_artifact = wandb.Artifact(f'model-epoch-{current_epoch_1_based}', type='model')
                # torch.save(self.handler.module.state_dict(), "temp_model.pth")
                # model_artifact.add_file("temp_model.pth")
                # self.run.log_artifact(model_artifact)
                # os.remove("temp_model.pth")
            except Exception as e:
                warnings.warn(f"Failed to log model weights/norms at epoch {current_epoch_1_based}: {e}",
                              RuntimeWarning)

        # Gradient Logging (placeholder - needs different hook)
        # if self.log_gradients:
        #     # Requires access to gradients *before* optimizer.step() or zero_grad()
        #     pass

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        if self.run:
            try:
                wandb.finish()
                print("W&B run finished.")
            except Exception as e:
                warnings.warn(f"W&B finish failed: {e}", RuntimeWarning)
            self.run = None  # Reset run object

    def state_dict(self) -> Dict[str, Any]:
        # Save run ID for potential resuming
        return {'wandb_run_id': self._run_id}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._run_id = state_dict.get('wandb_run_id')
        # wandb.init will handle resuming in on_train_begin if _run_id is set


# --- Performance Timer Callback ---

class EpochTimer(Callback):
    """Logs the time taken for each epoch."""

    def __init__(self):
        super().__init__()
        self._epoch_start_time: Optional[float] = None

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        self._epoch_start_time = time.perf_counter()  # Use perf_counter for more precise timing

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        if self._epoch_start_time is not None and logs is not None:
            epoch_time = time.perf_counter() - self._epoch_start_time
            logs['epoch_time_sec'] = epoch_time
            # The handler's default logging already prints this, but adding to logs
            # makes it available to other callbacks (like WandbLogger).
        self._epoch_start_time = None  # Reset timer


class ParameterNormMonitor(Callback):
    """Monitors and logs the norm of model parameters (weights).

    Args:
        log_freq_epoch (int): Log norm this often (in epochs). Defaults to 1.
        norm_type (float or int or 'inf'): Type of the norm to compute (e.g., 2 for L2 norm).
                                            Defaults to 2.
    """

    def __init__(self, log_freq_epoch: int = 1, norm_type: Union[float, int, str] = 2):
        super().__init__()
        self.log_freq_epoch = log_freq_epoch
        self.norm_type = norm_type
        if isinstance(self.norm_type, str) and self.norm_type.lower() != 'inf':
            warnings.warn(f"String norm_type '{self.norm_type}' is not 'inf'. Defaulting to L2 norm (2).")
            self.norm_type = 2
        elif isinstance(self.norm_type, str):
            self.norm_type = float('inf')

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        current_epoch_1_based = epoch + 1
        if not self.handler or not self.handler.model or logs is None:
            return
        if current_epoch_1_based % self.log_freq_epoch == 0:
            try:
                # Use handler.module to get unwrapped model parameters
                parameters = [p for p in self.handler.module.parameters() if p.data is not None]
                if not parameters:
                    return

                # Calculate the global norm across all parameters
                total_norm = torch.norm(
                    torch.stack([torch.norm(p.data.detach(), self.norm_type) for p in parameters]),
                    self.norm_type
                )
                norm_name = f"param_norm_{str(self.norm_type).lower()}"
                logs[norm_name] = total_norm.item()
            except Exception as e:
                warnings.warn(f"Failed to compute parameter norm: {e}", RuntimeWarning)
