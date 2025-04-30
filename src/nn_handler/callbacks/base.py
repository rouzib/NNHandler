import abc
from typing import Optional, Dict, Any


class Callback(abc.ABC):
    """Abstract base class used to build new callbacks.

    Callbacks can be used to customize the behavior of the `NNHandler` during
    training, evaluation, or prediction.
    """

    def __init__(self):
        self.handler: Optional['NNHandler'] = None  # Handler will be set when added

    def set_handler(self, handler: 'NNHandler'):
        self.handler = handler

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None): pass

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None): pass

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None): pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None): pass

    def on_train_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None): pass

    def on_train_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None): pass

    def on_val_begin(self, logs: Optional[Dict[str, Any]] = None): pass

    def on_val_end(self, logs: Optional[Dict[str, Any]] = None): pass

    def on_val_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None): pass

    def on_val_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None): pass

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the callback."""
        return {}  # Base implementation returns empty dict

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Loads the state of the callback."""
        pass  # Base implementation does nothing
