import abc
from typing import Dict, Any


class Sampler(abc.ABC):
    @abc.abstractmethod
    def sample(self, N, device, **kwargs):
        pass

    @abc.abstractmethod
    def save(self) -> Dict[str, Any]:  # Specify return type
        pass

    @abc.abstractmethod
    def load(self, **state_dict: Dict[str, Any]):  # Specify argument type
        pass