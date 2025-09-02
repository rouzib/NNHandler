from enum import Enum


class ModelType(Enum):
    """
    An enumeration representing different types of machine learning models.

    This class defines a set of predefined model types such as `CLASSIFICATION`,
    `GENERATIVE`, `REGRESSION`, and `SCORE_BASED`. It provides methods to
    convert from strings to enum members and to parse/validate model types.
    This ensures consistency and standardization across the usage of model
    type definitions.

    Attributes:
        CLASSIFICATION (str): Represents a classification model type.
        GENERATIVE (str): Represents a generative model type.
        REGRESSION (str): Represents a regression model type.
        SCORE_BASED (str): Represents a score-based model type.
    """
    CLASSIFICATION = "classification"
    GENERATIVE = "generative"
    REGRESSION = "regression"
    SCORE_BASED = "score_based"

    @classmethod
    def from_string(cls, s: str) -> 'ModelType':
        """Converts a string to a ModelType enum member."""
        try:
            return cls(s.lower())
        except ValueError:
            valid_types = [mt.value for mt in cls]
            raise ValueError(f"Invalid model_type string '{s}'. Valid options are: {valid_types}")

    @classmethod
    def parse(cls, model_type: str) -> 'ModelType':
        """
        Parses the provided model type and converts it to an instance of `ModelType`. This
        method handles various input cases where `model_type` can be a string, an instance
        of `ModelType`, or an enum member with a string value. Validation is performed to
        ensure the input is compatible with `ModelType`. If the input cannot be parsed,
        an appropriate exception is raised.

        Args:
            model_type (str | ModelType | Enum): The model type to parse. This can be a
                string representation, an instance of `ModelType`, or an enum member
                where the value is a string.

        Returns:
            ModelType: The parsed and validated `ModelType` instance.

        Raises:
            TypeError: If the input `model_type` is invalid or its type is not supported.
        """
        if isinstance(model_type, str):
            _model_type = ModelType.from_string(model_type)
        elif isinstance(model_type, ModelType):
            _model_type = model_type
        # Handle case where it might be an enum member from another definition
        elif issubclass(model_type.__class__, Enum) and hasattr(model_type, 'value') and isinstance(
                model_type.value, str):
            try:
                _model_type = ModelType.from_string(model_type.value)
            except ValueError:
                raise TypeError(f"Invalid model_type enum value: {model_type.value}")
        else:
            raise TypeError(f"model_type must be utils.ModelType or str, got {type(model_type)}")
        return _model_type


class LoggingMode(Enum):
    """
    Enumeration for defining different logging modes.

    This enumeration provides three modes for logging: console logging, file
    logging, and both. These modes can be used to control where log outputs
    are directed in logging frameworks.

    Attributes:
        CONSOLE: Logging output is directed to the console.
        FILE: Logging output is directed to a file.
        BOTH: Logging output is directed to both the console and a file.
    """
    CONSOLE = "console"
    FILE = "file"
    BOTH = "both"


class DataLoaderType(Enum):
    """
    Defines an enumeration for different types of dataloaders.

    This class provides distinct identifiers for various dataloader types used in
    data preprocessing and loading workflows. These types can be utilized to
    determine the behavior or method used for loading datasets in different
    contexts, such as memory optimization or rank-based data access.

    Attributes:
        STANDARD (str): Represents a standard dataloader configuration.
        RANK_CACHED (str): A dataloader configuration optimized for rank--cached datasets.
    """
    STANDARD = "standard"
    RANK_CACHED = "rank_cached"