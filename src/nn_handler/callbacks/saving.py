import os
import warnings
from typing import Optional, Dict, Any

import torch

from .base import Callback


class ModelCheckpoint(Callback):
    """Callback to save the model or weights at some frequency.

    Args:
        filepath (str): Path to save the model file. Can contain formatting
            options like `{epoch:02d}` or `{val_loss:.2f}`.
        monitor (str): Quantity to monitor (e.g., 'val_loss', 'val_accuracy').
        mode (str): One of {'min', 'max'}. If `save_best_only=True`, the decision
            to overwrite the current save file is made based on either the
            maximization or the minimization of the monitored quantity.
        save_best_only (bool): If True, only saves when the model is considered
            the "best" according to the monitored quantity and mode.
        save_weights_only (bool): If True, then only the model's weights are saved
            (`model.state_dict()`), else the full handler state is saved.
        save_freq (int): Frequency (in epochs) at which to save the model.
            If `save_best_only` is True, this parameter is ignored.
        verbose (int): Verbosity mode, 0 or 1.
    """

    def __init__(self, filepath: str = ".", monitor: str = 'val_loss', mode: str = 'min',
                 save_best_only: bool = True, save_weights_only: bool = False, save_freq: int = 1, verbose: int = 0):
        super().__init__()
        self._mode = None
        self.monitor_op = None
        self.best = None
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.save_freq = save_freq
        self.verbose = verbose

        if mode not in ['min', 'max']:
            warnings.warn(f"ModelCheckpoint mode '{mode}' is unknown, "
                          f"fallback to 'min'.", RuntimeWarning)
            mode = 'min'
        self.mode = mode

        self._current_epoch = 0  # Track epoch internally

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode: str):
        if mode not in ['min', 'max']:
            warnings.warn(f"ModelCheckpoint mode '{mode}' is unknown, "
                          f"fallback to 'min'.", RuntimeWarning)
            mode = 'min'
        if mode == 'min':
            self.monitor_op = torch.lt  # Use torch ops for tensor comparison
            self.best = torch.tensor(torch.inf)
        else:
            self.monitor_op = torch.gt
            self.best = torch.tensor(-torch.inf)
        self._mode = mode

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        self._current_epoch = logs["epoch"]  # epochs are 1-based in format string

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        logs = logs or {}
        base_filename = self.filepath.format(**logs)
        filepath = base_filename if base_filename.endswith(".pth") else f"{base_filename}.pth"
        # filepath = self.filepath.format(epoch=self._current_epoch, **logs)
        if self.save_best_only:
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn(f"Can save best model only with {self.monitor} available, "
                              f"skipping.", RuntimeWarning)
            else:
                current_tensor = torch.tensor(current)  # Ensure tensor for comparison
                if self.monitor_op(current_tensor, self.best):
                    if self.verbose > 0:
                        print(f'Epoch {self._current_epoch}: {self.monitor} improved '
                              f'from {self.best.item():.5f} to {current_tensor.item():.5f}, saving model to {filepath}')
                    if self.handler.logger:
                        self.handler.logger.info(
                            f'{self.__class__.__name__}: (Epoch {self._current_epoch}) {self.monitor} improved '
                            f'from {self.best.item():.5f} to {current_tensor.item():.5f}, saving model to {filepath}')
                    self.best = current_tensor
                    self._save_model(filepath)
                elif self.verbose > 1:
                    print(f'Epoch {self._current_epoch}: {self.monitor} did not improve from {self.best.item():.5f}')
                    if self.handler.logger:
                        self.handler.logger.debug(
                            f'{self.__class__.__name__}: (Epoch {self._current_epoch}) {self.monitor} did not improve from '
                            f'{self.best.item():.5f}')

        else:
            if self._current_epoch % self.save_freq == 0:
                if self.verbose > 0:
                    print(f'Epoch {self._current_epoch}: saving model to {filepath}')
                if self.handler.logger:
                    self.handler.logger.info(
                        f'{self.__class__.__name__}: (Epoch {self._current_epoch}) saving model to {filepath}')
                self._save_model(filepath)

    def _save_model(self, filepath: str):
        if self.handler._distributed and self.handler._rank != 0:
            return

        if self.handler is None:
            warnings.warn("Handler not set in ModelCheckpoint, cannot save.", RuntimeWarning)
            return
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if self.save_weights_only:
            torch.save(self.handler.model.state_dict(), filepath)
        else:
            self.handler.save(filepath)  # Save full handler state

    def state_dict(self) -> Dict[str, Any]:
        return {'best': self.best.item(),
                'mode': self.mode,
                'filepath': self.filepath,
                'current_epoch': self._current_epoch,
                'monitor': self.monitor,
                'save_best_only': self.save_best_only,
                'save_weights_only': self.save_weights_only,
                'save_freq': self.save_freq,
                'verbose': self.verbose,
                }  # Save best value as standard python type

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.best = torch.tensor(state_dict.get('best', self.best.item()))  # Load as tensor
        self.mode = state_dict.get('mode', self.mode)
        self.filepath = state_dict.get('filepath', self.filepath)
        self._current_epoch = state_dict.get('current_epoch', self._current_epoch)
        self.monitor = state_dict.get('monitor', self.monitor)
        self.save_best_only = state_dict.get('save_best_only', self.save_best_only)
        self.save_weights_only = state_dict.get('save_weights_only', self.save_weights_only)
        self.save_freq = state_dict.get('save_freq', self.save_freq)
        self.verbose = state_dict.get('verbose', self.verbose)
