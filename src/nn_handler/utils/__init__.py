from .import_utils import import_with_fallback
from .ddp_init import _resolve_device, initialize_ddp, _should_use_distributed, _initialize_distributed
from .ddp_decorators import parallel_on_all_devices, parallelize_on_gpus, on_rank
from .dd_helpers import aggregate_loss, aggregate_metrics
from .enums import ModelType, LoggingMode, DataLoaderType

__all__ = ["initialize_ddp", "on_rank", "parallelize_on_gpus", "import_with_fallback", "aggregate_loss",
           "aggregate_metrics", "ModelType", "LoggingMode", "DataLoaderType"]

(GradScaler, autocast), _amp_available = import_with_fallback("torch.amp", ["GradScaler", "autocast"])
(ExponentialMovingAverage,), _ema_available = import_with_fallback("torch_ema", ["ExponentialMovingAverage"])
