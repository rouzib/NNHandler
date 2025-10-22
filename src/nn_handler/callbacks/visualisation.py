import abc
import os
import warnings
from typing import Optional, Any, Tuple, Dict, Callable
import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from .base import Callback
from ..trainer.batch_utils import _prepare_batch
from ..utils import autocast, on_rank

try:
    import matplotlib.pyplot as plt

    _matplotlib_available = True
except ImportError:
    _matplotlib_available = False

try:
    from IPython.core.display_functions import clear_output, display

    _ipython_available = True
except ImportError:
    _ipython_available = False


class BasePredictionVisualizer(Callback):
    """Base class for callbacks that visualize model predictions periodically.

    Users should inherit from this and implement the `_visualize_batch` method
    for their specific data type.

    Args:
        val_loader (DataLoader): A DataLoader to draw validation samples from.
                                 Must be provided if `use_fixed_batch` is False.
        log_freq_epoch (int): Visualize predictions every N epochs. Defaults to 5.
        num_samples (int): Number of samples to visualize in the batch. Defaults to 8.
        save_dir (str): Directory to save the visualizations. Defaults to None (no saving).
        use_fixed_batch (bool): If True, uses the same batch every time. Requires `fixed_batch` to be set.
        fixed_batch (Optional[Any]): A specific batch of data (input, Optional[target]) to use for visualization if `use_fixed_batch` is True.
    """

    @staticmethod
    def default_prediction_fn(model, model_inputs, **additional_params):
        return model(model_inputs, **additional_params)

    def __init__(self,
                 log_freq_epoch: int = 5,
                 num_samples: int = 8,
                 save_dir: str = None,
                 val_loader: Optional[DataLoader] = None,
                 use_fixed_batch: bool = False,
                 fixed_batch: Optional[Any] = None,
                 get_predictions_fn: Optional[Callable] = None, ):
        super().__init__()
        self.log_freq_epoch = log_freq_epoch
        self.num_samples = num_samples
        self.save_dir = save_dir
        self.val_loader = val_loader
        self.use_fixed_batch = use_fixed_batch
        self._fixed_batch_prepared: Optional[Any] = None
        self.get_predictions_fn = get_predictions_fn

        if use_fixed_batch:
            if fixed_batch is None:
                raise ValueError("`fixed_batch` must be provided if `use_fixed_batch` is True.")
            self._fixed_batch_prepared = fixed_batch  # Store the raw batch
        elif val_loader is None:
            raise ValueError("`val_loader` must be provided if `use_fixed_batch` is False.")

        if self.get_predictions_fn is None:
            self.get_predictions_fn = self.default_prediction_fn

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

    def _get_visualization_batch(self) -> Optional[Tuple[Any, Any, Any]]:
        """Gets a batch suitable for visualization."""
        if not self.handler or not self.handler.model: return None

        batch_data = None
        additional_params = {}
        if self.use_fixed_batch:
            # Prepare the fixed batch if not already done (move to device)
            if isinstance(self._fixed_batch_prepared, tuple) and len(
                    self._fixed_batch_prepared) >= 2:  # Assume (input, target)
                inputs = self._fixed_batch_prepared[0][:self.num_samples].to(self.handler.device)
                targets = self._fixed_batch_prepared[1][:self.num_samples].to(self.handler.device)
                batch_data = (inputs, targets)
            else:  # Assume input only
                if isinstance(self._fixed_batch_prepared, tuple):
                    inputs = self._fixed_batch_prepared[0][:self.num_samples].to(self.handler.device)
                else:
                    inputs = self._fixed_batch_prepared[:self.num_samples].to(self.handler.device)
                batch_data = (inputs, None)
        elif self.val_loader:
            try:
                # Get a batch from the validation loader
                batch_data_raw = next(iter(self.val_loader))
                batch_data = _prepare_batch(self.handler, batch_data_raw)  # Use handler's prep
                # Extract inputs, targets, and additional parameters
                inputs = batch_data["inputs"]
                targets = batch_data["targets"]
                additional_params = {k: v for k, v in batch_data.items() if k not in ["inputs", "targets"]}

                # Select subset
                inputs = inputs[:self.num_samples]
                targets = targets[:self.num_samples] if targets is not None else None
                batch_data = (inputs, targets)
            except StopIteration:
                warnings.warn("Validation loader exhausted, cannot get visualization batch.", RuntimeWarning)
                return None
            except Exception as e:
                warnings.warn(f"Error getting batch from val_loader for visualization: {e}", RuntimeWarning)
                return None
        else:
            return None  # Should not happen based on __init__ checks

        # Run prediction (using handler's __call__ or predict logic)
        self.handler.model.eval()  # Ensure eval mode
        with torch.no_grad():
            with autocast(device_type=self.handler._device.type, enabled=True):
                # Apply EMA context if handler uses it? Maybe add flag? For simplicity, don't apply EMA here.
                # Use handler's __call__ which uses the *current* model state (could be EMA or not)
                model_inputs = batch_data[0]
                predictions = self.get_predictions_fn(self.handler, model_inputs, **additional_params)

        return batch_data[0], batch_data[1], predictions  # inputs, targets, predictions

    @abc.abstractmethod
    def _visualize_batch(self, inputs: Any, targets: Optional[Any], predictions: Any, epoch: int,
                         logs: Optional[Dict[str, Any]] = None):
        """User-defined method to create and save the visualization."""
        raise NotImplementedError

    @on_rank(0, barrier=True)
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        current_epoch_1_based = epoch + 1
        if current_epoch_1_based % self.log_freq_epoch == 0:
            batch_info = self._get_visualization_batch()
            if batch_info:
                inputs, targets, predictions = batch_info
                try:
                    self._visualize_batch(inputs.cpu().to(torch.float32),
                                          targets.cpu().to(torch.float32) if targets is not None else None,
                                          predictions.cpu().to(torch.float32) if isinstance(predictions,
                                                                                            torch.Tensor) else predictions,
                                          # Move tensors to CPU
                                          current_epoch_1_based,
                                          logs)
                except Exception as e:
                    warnings.warn(f"Failed to visualize predictions at epoch {current_epoch_1_based}: {e}",
                                  RuntimeWarning)
                    import traceback
                    traceback.print_exc()  # Print stack trace for easier debugging


# --- Example Implementation for Images ---

class ImagePredictionVisualizer(BasePredictionVisualizer):
    """Visualizes image predictions (e.g., for Autoencoders, Segmentation).

    Assumes inputs, targets (optional), and predictions are image tensors
    (e.g., B, C, H, W) that can be plotted with matplotlib.
    """

    def __init__(self, show=False, clear_cell=False, vertical=False, log_scale=False, dpi=300, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not _matplotlib_available:
            raise ImportError("ImagePredictionVisualizer requires 'matplotlib'. Install with 'pip install matplotlib'")
        self.show = show
        self.clear_cell = clear_cell
        if not _ipython_available:
            raise ImportError("clear_cell=True requires 'IPython'. Install with 'pip install ipython'")
        self.vertical = vertical
        self.log_scale = log_scale
        self.dpi = dpi

    def _visualize_batch(self, inputs: torch.Tensor, targets: Optional[torch.Tensor], predictions: torch.Tensor,
                         epoch: int, logs: Optional[Dict[str, Any]] = None):
        # Determine grid size
        num_cols = 3 if targets is not None else 2  # Input, Target, Pred | Input, Pred
        num_rows = self.num_samples
        if self.vertical:
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
        else:
            fig, axes = plt.subplots(num_cols, num_rows, figsize=(num_rows * 3, num_cols * 3))
        fig.suptitle(f"Predictions - Epoch {epoch}, loss={logs.get('loss', math.nan):.2e}", fontsize=16)

        # Ensure axes is always a 2D array for consistent indexing
        if num_rows == 1 and num_cols == 1:
            axes = np.array([[axes]])
        elif num_rows == 1:
            axes = np.array([axes])
        elif num_cols == 1:
            axes = np.array([[ax] for ax in axes])

        axes = np.atleast_2d(axes)
        if not self.vertical:
            axes = axes.T

        for i in range(self.num_samples):
            # Handle case where batch size might be smaller than num_samples
            if i >= inputs.shape[0]: break

            img_in = inputs[i].permute(1, 2, 0).squeeze()  # H, W, C or H, W
            img_pred = predictions[i].permute(1, 2, 0).squeeze()

            if self.log_scale:
                img_in = torch.log(img_in)
                img_pred = torch.log(img_pred)

            use_colorbar_in = False
            use_colorbar_pred = False
            if img_in.ndim == 2:
                use_colorbar_in = True
            if img_pred.ndim == 2:
                use_colorbar_pred = True

            # --- Plot Input ---
            ax = axes[i][0]
            ax.imshow(img_in.numpy(), cmap='gray')  # Assuming grayscale or RGB suitable for imshow
            ax.set_title(f"Input {i}")
            ax.axis('off')
            if use_colorbar_in:
                fig.colorbar(ax.get_children()[0], ax=ax)

            # --- Plot Target (if available) ---
            if targets is not None and i < targets.shape[0]:
                img_target = targets[i].permute(1, 2, 0).squeeze()
                if self.log_scale:
                    img_target = torch.log(img_target)
                ax = axes[i][1]
                ax.imshow(img_target.numpy(), cmap='gray')
                ax.set_title(f"Target {i}")
                ax.axis('off')
                if img_target.ndim == 2:
                    fig.colorbar(ax.get_children()[0], ax=ax)

            # --- Plot Prediction ---
            ax = axes[i][num_cols - 1]  # Last column is prediction
            ax.imshow(img_pred.numpy(), cmap='gray')
            ax.set_title(f"Prediction {i}")
            ax.axis('off')
            if use_colorbar_pred:
                fig.colorbar(ax.get_children()[0], ax=ax)

        plt.tight_layout(rect=(0, 0.03, 1, 0.95))  # Adjust layout to prevent title overlap
        if self.clear_cell:
            clear_output(wait=True)
            display(plt.gcf())
        if self.show:
            plt.show(block=False)
        if self.save_dir:
            save_name = os.path.join(self.save_dir, f"epoch_{epoch:04d}_predictions.png")
            plt.savefig(save_name, dpi=self.dpi)
            print(f"Saved prediction visualization to {save_name}")

        plt.close(fig)
