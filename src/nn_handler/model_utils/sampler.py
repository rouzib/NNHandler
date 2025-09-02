import abc
from typing import Dict, Any


class Sampler(abc.ABC):
    """
    Abstract base class that defines the interface for sampling strategies.

    This class is intended to be subclassed to implement various sampling
    strategies. It ensures all subclasses provide implementations for
    the `sample`, `save`, and `load` methods, which are crucial for
    sampling, serialization, and deserialization processes.
    """
    def __init__(self, **kwargs):
        super().__init__()

    @abc.abstractmethod
    def sample(self, N, device, **kwargs):
        pass

    @abc.abstractmethod
    def save(self) -> Dict[str, Any]:  # Specify return type
        pass

    @abc.abstractmethod
    def load(self, **state_dict: Dict[str, Any]):  # Specify argument type
        pass