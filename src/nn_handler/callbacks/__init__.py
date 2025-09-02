from .base import Callback
from .saving import ModelCheckpoint
from .monitor import LearningRateMonitor, TensorBoardLogger, EpochTimer, WandbLogger
from .training import EarlyStopping
from .utils import SimpleGarbageCollector, CudaGarbageCollector
from .visualisation import BasePredictionVisualizer, ImagePredictionVisualizer

__all__ = ['Callback', 'ModelCheckpoint', 'LearningRateMonitor', 'TensorBoardLogger', 'EpochTimer', 'WandbLogger',
           'EarlyStopping', 'SimpleGarbageCollector', 'CudaGarbageCollector', 'BasePredictionVisualizer',
           'ImagePredictionVisualizer']

str_to_callback = {'ModelCheckpoint': ModelCheckpoint,
                   'LearningRateMonitor': LearningRateMonitor,
                   'TensorBoardLogger': TensorBoardLogger,
                   'EpochTimer': EpochTimer,
                   'WandbLogger': WandbLogger,
                   'EarlyStopping': EarlyStopping,
                   'SimpleGarbageCollector': SimpleGarbageCollector,
                   'CudaGarbageCollector': CudaGarbageCollector,
                   'BasePredictionVisualizer': BasePredictionVisualizer,
                   'ImagePredictionVisualizer': ImagePredictionVisualizer}
