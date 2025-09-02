import inspect
from typing import Any, Optional

from torch import Tensor

from ..utils import ModelType


def _prepare_batch(nn_handler: 'NNHandler', batch: Any) -> dict[str, Any | None]:
    """
    Prepares an individual batch for processing by a neural network handler. The method organizes
    inputs, targets, and any additional parameters to match the requirements of the model's
    forward function.

    The batch is processed based on the model's type (e.g., classification, regression, generative,
    score-based) and may include additional parameters derived dynamically using the forward
    method's signature. Data is moved to the specified device, and elements in the batch sequence
    are assigned as inputs, targets, and extra arguments as needed.

    :param nn_handler: The neural network handler that manages the model and runtime configuration.
    :param batch: The raw batch data containing inputs, targets, and potentially additional
        parameters, which will be structured for the model's forward function.
    :return: A dictionary containing the prepared inputs, targets, and any additional parameters.
    """
    inputs: Any
    targets: Optional[Any] = None
    additional_params = {}
    _model_type = nn_handler._model_type

    # Get signature from the underlying module
    model_to_inspect = nn_handler.module
    try:
        model_sig = inspect.signature(model_to_inspect.forward)
        valid_param_names = list(model_sig.parameters.keys())[1:]  # Skip 'self', get names of forward args
    except (TypeError, ValueError):
        # Handle models without standard forward signature (e.g., functional) gracefully
        valid_param_names = []
        nn_handler.warn(
            f"Could not inspect forward signature of {type(model_to_inspect).__name__}. Extra batch items may not be passed correctly.",
            Warning)

    # --- Helper to move data recursively ---
    def _to_device(data):
        if isinstance(data, Tensor):
            return data.to(nn_handler._device, non_blocking=True)  # Use non_blocking for potential speedup
        elif isinstance(data, (list, tuple)):
            return type(data)(_to_device(d) for d in data)  # Preserve original type (list/tuple)
        elif isinstance(data, dict):
            return {k: _to_device(v) for k, v in data.items()}
        else:
            return data  # Keep non-tensor data as is

    # --- Process Batch based on structure and model type ---
    if isinstance(batch, (list, tuple)):
        if not batch:
            raise ValueError("Batch is empty.")

        inputs = _to_device(batch[0])
        extra_items_idx_start = 1  # Index of items after input

        # Determine targets based on model type
        if _model_type in [ModelType.CLASSIFICATION, ModelType.REGRESSION]:
            if len(batch) < 2:
                raise ValueError(
                    f"{_model_type.name} model type expects batch to be a sequence (inputs, targets, ...). Got sequence of length {len(batch)}.")
            targets = _to_device(batch[1])
            extra_items_idx_start = 2
        elif _model_type == ModelType.GENERATIVE:
            # Assumes input is also the target (e.g., autoencoder)
            targets = inputs  # Target is a reference to the input tensor on the device
        elif _model_type == ModelType.SCORE_BASED:
            # Score-based models often don't need explicit targets here; loss handles it
            targets = None
        else:
            # Should be unreachable if model_type validation is correct
            raise ValueError(f"Unsupported ModelType: {_model_type}")

        # Process remaining items in the batch as additional forward parameters
        extra_batch_items = batch[extra_items_idx_start:]
        num_extra_params_needed = len(valid_param_names)
        num_extra_items_given = len(extra_batch_items)

        if num_extra_items_given > num_extra_params_needed:
            # Log warning
            nn_handler.warn(
                f"Batch contains {num_extra_items_given} extra items, but model forward() expects {num_extra_params_needed} after input. Ignoring excess items.",
                Warning)
            # Trim excess items
            extra_batch_items = extra_batch_items[:num_extra_params_needed]
            num_extra_items_given = len(extra_batch_items)  # Update count

        # Check if enough extra items were provided for required params
        # This check might be too strict if some params are optional with defaults
        # if num_extra_items_given < num_extra_params_needed:
        #    missing_params = valid_param_names[num_extra_items_given:]
        #    warnings.warn(f"Model forward() expects parameters {missing_params} which were not found in the batch.", RuntimeWarning)

        # Assign extra items to parameter names in order
        for i, item in enumerate(extra_batch_items):
            param_name = valid_param_names[i]
            additional_params[param_name] = _to_device(item)

    else:  # Assume batch is a single tensor (input)
        inputs = _to_device(batch)
        if _model_type == ModelType.GENERATIVE:
            targets = inputs
        elif _model_type in [ModelType.CLASSIFICATION, ModelType.REGRESSION]:
            raise ValueError(
                f"{_model_type.name} model type expects batch to be a sequence (inputs, targets, ...). Got a single tensor.")
        # Score-based handles single tensor input correctly (targets=None)

    return {"inputs": inputs, "targets": targets, **additional_params}
