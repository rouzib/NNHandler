# Enumerations Module

The `enums` module provides standardized enumeration types used throughout the NNHandler framework. These enumerations help ensure consistency and type safety when working with different model types, logging modes, and dataloader configurations.

## Overview

This module defines three main enumeration classes:

1. **ModelType**: Defines different types of machine learning models (classification, generative, etc.).
2. **LoggingMode**: Specifies where logging output should be directed (console, file, or both).
3. **DataLoaderType**: Identifies different dataloader configurations for various use cases.

## Enumerations

### `ModelType`

```python
class ModelType(Enum):
    CLASSIFICATION = "classification"
    GENERATIVE = "generative"
    REGRESSION = "regression"
    SCORE_BASED = "score_based"
```

An enumeration representing different types of machine learning models. Used to configure the behavior of the `NNHandler` class based on the type of model being used.

#### Methods:

- **`from_string(cls, s: str) -> ModelType`**: Converts a string to a ModelType enum member.
  - Raises `ValueError` if the string doesn't match any valid model type.

- **`parse(cls, model_type: Union[str, ModelType, Enum]) -> ModelType`**: Parses and validates a model type from various input formats.
  - Handles string inputs, ModelType instances, or other enum members with string values.
  - Raises `TypeError` if the input cannot be converted to a valid ModelType.

#### Example:

```python
from src.nn_handler.utils.enums import ModelType

# Direct usage
model_type = ModelType.CLASSIFICATION

# From string
model_type = ModelType.from_string("generative")

# Parse from various formats
model_type = ModelType.parse("regression")  # From string
model_type = ModelType.parse(ModelType.SCORE_BASED)  # From enum instance
```

### `LoggingMode`

```python
class LoggingMode(Enum):
    CONSOLE = "console"
    FILE = "file"
    BOTH = "both"
```

Defines different modes for directing logging output. Used by the logger module to configure where log messages are sent.

#### Example:

```python
from src.nn_handler.utils.enums import LoggingMode
from src.nn_handler.logger import initialize_logger

# Configure logging to both console and file
logger = initialize_logger(
    logger_name="MyLogger",
    mode=LoggingMode.BOTH,
    filename="logs/training.log"
)
```

### `DataLoaderType`

```python
class DataLoaderType(Enum):
    STANDARD = "standard"
    RANK_CACHED = "rank_cached"
```

Specifies different types of dataloaders used in the framework. This helps determine the behavior or method used for loading datasets in different contexts.

- **STANDARD**: The default dataloader configuration.
- **RANK_CACHED**: A specialized dataloader that caches data per rank, optimized for distributed training scenarios.

#### Example:

```python
from src.nn_handler.utils.enums import DataLoaderType
from src.nn_handler import NNHandler
from torch.utils.data import Dataset

# Create a handler instance
handler = NNHandler(...)

# Configure a dataloader with rank caching for better performance in distributed training
my_dataset = Dataset(...)
handler.set_train_loader(
    dataset=my_dataset,
    dataloader_type=DataLoaderType.RANK_CACHED,
    batch_size=32
)
```

## Usage Notes

- When configuring a `NNHandler` instance, the `model_type` parameter accepts either a `ModelType` enum member or a string that can be converted to one.
- The `LoggingMode` enum is used with the `initialize_logger` function to control logging behavior.
- The `DataLoaderType` enum is used when setting up dataloaders with the `set_train_loader` and `set_val_loader` methods to optimize data loading for different scenarios.