import copy
import warnings
from typing import Any, Dict, Optional

import torch

from .base import Callback
from ..model_utils.scheduler import Schedule

class EarlyStopping(Callback):
    """Stop training when a monitored quantity has stopped improving.

    Args:
        monitor (str): Quantity to be monitored.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        mode (str): One of {"min", "max"}.
        verbose (int): Verbosity mode.
        restore_best_weights (bool): Whether to restore model weights from the epoch
            with the best value of the monitored quantity. If False, the model weights
            obtained at the last step of training are used. Requires the handler state
            to be saved (e.g., via ModelCheckpoint).
    """

    def __init__(self, monitor: str = 'val_loss', min_delta: float = 0.0, patience: int = 10,
                 mode: str = 'min', verbose: int = 0, restore_best_weights: bool = False):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None  # To store weights if restore_best_weights is True

        if mode not in ['min', 'max']:
            warnings.warn(f"EarlyStopping mode '{mode}' is unknown, fallback to 'min'.", RuntimeWarning)
            mode = 'min'
        self.mode = mode
        if self.mode == 'min':
            self.monitor_op = torch.lt
            self.min_delta *= -1  # Adjust delta for minimization
            self.best = torch.tensor(torch.inf)
        else:
            self.monitor_op = torch.gt
            self.best = torch.tensor(-torch.inf)

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        self.wait = 0  # Reset wait counter at the beginning of training
        self.stopped_epoch = 0
        self.best = torch.tensor(torch.inf) if self.mode == 'min' else torch.tensor(-torch.inf)
        self.best_weights = None

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(f"Early stopping requires {self.monitor} available!, skipping.", RuntimeWarning)
            return

        current_tensor = torch.tensor(current)
        if self.monitor_op(current_tensor - self.min_delta, self.best):
            self.best = current_tensor
            self.wait = 0
            if self.restore_best_weights:
                # Need deep copy if model can be modified later
                self.best_weights = copy.deepcopy(self.handler.model.state_dict())
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch + 1  # Store 1-based epoch
                self.handler._stop_training = True  # Signal handler to stop
                if self.restore_best_weights and self.best_weights is not None:
                    if self.verbose > 0:
                        print("Restoring model weights from the end of the best epoch.")
                    self.handler.model.load_state_dict(self.best_weights)
                if self.verbose > 0:
                    print(f"Epoch {self.stopped_epoch}: early stopping")

    def state_dict(self) -> Dict[str, Any]:
        state = {
            'wait': self.wait,
            'stopped_epoch': self.stopped_epoch,
            'best': self.best.item(),
            # Don't save best_weights here, rely on ModelCheckpoint or handler save
        }
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.wait = state_dict.get('wait', 0)
        self.stopped_epoch = state_dict.get('stopped_epoch', 0)
        self.best = torch.tensor(state_dict.get('best', self.best.item()))
        # best_weights are restored via handler load or ModelCheckpoint


class ParamScheduler(Callback):
    """
    The ParamScheduler class is a callback for dynamically updating a specified parameter of a
    module based on a provided schedule during training epochs.

    This class allows flexible parameter adjustment during model training by utilizing a defined
    schedule for the parameter. Users can specify the name of the parameter to update, a schedule
    object that dictates the parameter's values at different epochs, and verbosity settings for
    logging parameter updates. The updated values of the parameter are applied to the corresponding
    module when an epoch begins.

    Args:
        parameter_name (str): The name of the parameter to be updated.
        schedule (Schedule): The schedule to apply to the parameter.
        verbose (int): Verbosity mode. >0 -> prints msg
    """
    def __init__(self, parameter_name: str, schedule: Schedule, verbose: int = 0):
        super().__init__()
        self.parameter_name = parameter_name
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch: int, logs=None):
        value = self.schedule.get_value(epoch)
        if hasattr(self.handler.module, self.parameter_name):
            setattr(self.handler.module, self.parameter_name, value)
            msg = f"Epoch {epoch}: set '{self.parameter_name}' to {value:.6f}"
            if self.verbose > 0:
                print(msg)
            if self.handler.logger:
                self.handler.logger.info(msg)