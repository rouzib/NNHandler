import logging
import os
from typing import Dict, Any

from ..utils import on_rank


@on_rank(0, barrier=True)
def auto_save_epoch(nn_handler: 'NNHandler', epoch: int, total_epochs: int, save_on_last_epoch: bool,
                     logs: Dict[str, Any]):
    """Handles the logic for auto-saving the model state (only executed on rank 0)."""
    _auto_saver = nn_handler._auto_saver

    if _auto_saver.save_interval is None or _auto_saver.save_path is None:
        return  # Auto-save disabled

    current_epoch_1_based = epoch + 1
    should_save = False

    # Interval saving
    if _auto_saver.save_interval > 0 and (current_epoch_1_based % _auto_saver.save_interval == 0):
        should_save = True

    # Save on last epoch requested
    is_last_epoch = (current_epoch_1_based == total_epochs)
    if save_on_last_epoch and is_last_epoch:
        should_save = True

    # Save if interval is -1 (only last epoch)
    if _auto_saver.save_interval == -1 and is_last_epoch:
        should_save = True

    if should_save:
        # Use logs dict for formatting filename keys (e.g., val_loss, val_accuracy)
        format_dict = logs.copy()
        format_dict['epoch'] = current_epoch_1_based  # Ensure epoch is available
        try:
            # Add .pth extension if not included in the format name
            base_filename = _auto_saver.save_model_name.format(**format_dict)
            filename = base_filename if base_filename.endswith(".pth") else f"{base_filename}.pth"
        except KeyError as e:
            # Fallback if format key is missing in logs (e.g., no validation ran)
            filename = f"{_auto_saver.save_model_name}_epoch{current_epoch_1_based}.pth"
            nn_handler.warn(
                f"Auto-save filename formatting failed (KeyError: {e}). Using fallback: {filename}", Warning)
        except Exception as e:
            filename = f"{_auto_saver.save_model_name}_epoch{current_epoch_1_based}.pth"
            nn_handler.log(
                f"Auto-save filename formatting failed unexpectedly ({type(e).__name__}: {e}). Using fallback: {filename}",
                logging.ERROR)

        save_path = os.path.join(_auto_saver.save_path, filename)

        nn_handler.log(f"Auto-saving handler state to: {save_path}")

        # Perform save using the main save method
        nn_handler.save(save_path)

        # Handle overwriting previous auto-save file
        # Check if overwrite enabled, if a previous file exists, and if it's different from the current save
        if _auto_saver.overwrite_last_saved and \
                _auto_saver.last_saved_model and \
                _auto_saver.last_saved_model != save_path:
            try:
                os.remove(_auto_saver.last_saved_model)
                nn_handler.log(f"Removed previous auto-saved state: {_auto_saver.last_saved_model}", logging.DEBUG)
            except OSError as e:
                # Warn if removal fails but continue
                nn_handler.warn(f"Could not remove previous auto-saved state '{_auto_saver.last_saved_model}': {e}",
                                RuntimeWarning)

        # Update the last saved model path
        _auto_saver.last_saved_model = save_path