import contextlib


class GradScaler:
    """
    Provides a dummy implementation of the GradScaler class.

    The GradScaler class is a placeholder used when automatic mixed precision
    (AMP) is not available. It mimics the interface of an actual GradScaler to
    enable seamless integration with training workflows where AMP could be optional.
    This dummy implementation does not modify or scale gradients but provides the
    basic methods required for compatibility.

    :ivar _enabled: Determines if the GradScaler is enabled. In this dummy
        implementation, it is set to a boolean value indicating whether AMP
        functionality is to be considered enabled.
    :type _enabled: bool
    """
    def __init__(self, enabled=False):
        self._enabled = enabled  # Store enabled state

    def scale(self, loss): return loss

    def step(self, optimizer): optimizer.step()

    def update(self): pass

    def __call__(self, *args, **kwargs): pass  # make it callable for load/save state_dict logic

    def state_dict(self): return {}

    def load_state_dict(self, state_dict): pass

    # Add is_enabled to match real GradScaler
    def is_enabled(self): return self._enabled


@contextlib.contextmanager
def autocast(device_type="cpu", enabled=False, **kwargs):
    # Simplified context manager that just yields
    yield


class ExponentialMovingAverage:
    """
    Implements Dummy Exponential Moving Average (EMA) of model parameters or similar objects.

    :ivar parameters: The model parameters tracked by this EMA instance.
    :type parameters: Any
    :ivar decay: The decay factor for updating EMA values.
    :type decay: float
    """
    def __init__(self, parameters, decay): pass

    def update(self): pass

    @contextlib.contextmanager  # Ensure it's a context manager
    def average_parameters(self): yield  # Dummy context manager

    def copy_to(self, parameters=None): pass

    def state_dict(self): return {}

    def load_state_dict(self, state_dict): pass