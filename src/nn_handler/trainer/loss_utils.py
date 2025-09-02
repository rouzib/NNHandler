import inspect
import logging
from typing import Any, Optional, Union, Tuple, List, Dict

import torch
from torch import Tensor

from ..utils import ModelType


def _calculate_loss(nn_handler: 'NNHandler', model_output: Any, targets: Optional[Any],
                    inputs: Any, current_epoch: int) -> Union[
    Tensor, Tuple[Tensor, List[Tensor]]]:
    """
    Calculate the loss for a given model type and configuration. This function determines
    the appropriate arguments based on the model type and calls the configured loss
    function. If required by the loss function and compatible with its signature, the
    current epoch number is passed as a keyword argument.

    :param nn_handler: The handler object responsible for managing the neural network's
        configuration, loss function, and other associated components.
    :type nn_handler: NNHandler
    :param model_output: The output produced by the model, which will be used for loss
        calculation.
    :type model_output: Any
    :param targets: The ground truth data against which the model output will be
        compared. Can be None for specific model types, but not permissible for
        classification, regression, or generative models.
    :type targets: Optional[Any]
    :param inputs: The input data for the model, necessary for specific loss calculations,
        such as for score-based models.
    :type inputs: Any
    :param current_epoch: The current training epoch number, which might be passed to
        the loss function if allowed by its signature.
    :type current_epoch: int
    :return: A tensor representing the calculated loss, or a tuple containing the loss
        tensor and additional information.
    :rtype: Union[Tensor, Tuple[Tensor, List[Tensor]]]
    :raises RuntimeError: If the loss function is not set in the handler or mandatory
        attributes are missing (like targets for certain model types or an SDE for
        score-based models).
    :raises ValueError: If the model type is unsupported for loss calculation.
    """
    if nn_handler._loss_fn is None:
        raise RuntimeError("Loss function not set.")

    _model_type = nn_handler._model_type
    loss_args = []
    _loss_fn = nn_handler._loss_fn
    # Start with configured kwargs, potentially add epoch later
    loss_kwargs = nn_handler._loss_fn_kwargs.copy()

    if _model_type in [ModelType.CLASSIFICATION, ModelType.REGRESSION, ModelType.GENERATIVE]:
        if targets is None:  # Should not happen if _prepare_batch is correct
            raise RuntimeError(f"{_model_type.name} requires targets for loss calculation.")
        loss_args = [model_output, targets]
    elif _model_type == ModelType.SCORE_BASED:
        if nn_handler.sde is None:
            raise RuntimeError("Score-based model requires an SDE to be set for loss calculation.")
        # Assumed signature: loss(data, sde, model, device, **kwargs)
        # Pass the unwrapped model for score-based loss functions
        loss_args = [inputs, nn_handler.sde, nn_handler.module, nn_handler._device]
    else:
        raise ValueError(f"Unsupported ModelType for loss: {_model_type}")

    # Add epoch if required and loss function signature allows it
    if nn_handler._pass_epoch_to_loss:
        try:
            # Attempt to bind 'epoch' as a keyword argument
            inspect.signature(_loss_fn).bind_partial(**{'epoch': current_epoch})
            # If bind doesn't raise TypeError, add 'epoch' to kwargs
            loss_kwargs['epoch'] = current_epoch
        except TypeError:
            # Signature doesn't accept 'epoch' kwarg, issue warning on rank 0 once
            nn_handler.warn(
                f"Loss function {getattr(_loss_fn, '__name__', repr(_loss_fn))} requested epoch passing (pass_epoch_to_loss=True), "
                "but it does not accept 'epoch' as a keyword argument. Epoch will not be passed.")
        except Exception as e:  # Catch other potential signature errors
            nn_handler.raise_error(Exception,
                                   f"Error inspecting signature of loss function {getattr(_loss_fn, '__name__', repr(_loss_fn))}: {e}",
                                   e)

    # Call the loss function
    loss = _loss_fn(*loss_args, **loss_kwargs)
    return loss


def _calculate_metrics(nn_handler: 'NNHandler', model_output: Any, targets: Optional[Any]) -> Dict[str, float]:
    """
    Calculates metrics for a given model output and corresponding targets using pre-defined metric
    functions within the provided neural network handler. It ensures all metric computations are
    free from impact on gradient calculations. Supports error handling by logging errors and
    storing NaN for metrics that cannot be computed.

    :param nn_handler: The neural network handler object containing the model type, list of
        metrics, and logging functionality.
    :type nn_handler: NNHandler

    :param model_output: The output of the model to be evaluated against the targets.
    :type model_output: Any

    :param targets: The target values for evaluating the model output. Can be None in certain
        scenarios, such as score-based validation without targets.
    :type targets: Optional[Any]

    :return: A dictionary with metric names as keys and their computed values as floats. If any
        metric encounters an error during computation, its value will be stored as NaN.
    :rtype: Dict[str, float]
    """
    _model_type = nn_handler._model_type
    batch_metrics = {}
    _metrics = nn_handler._metrics
    if not _metrics:
        return batch_metrics
    # Cannot calculate metrics requiring targets if targets are None (e.g., score-based validation)
    # if targets is None and _model_type in [ModelType.CLASSIFICATION, ModelType.REGRESSION]:
    #     return batch_metrics

    with torch.no_grad():  # Ensure metrics don't track gradients
        for name, metric_fn in _metrics.items():
            try:
                # Assume metric_fn takes (output, target)
                value = metric_fn(model_output, targets)
                # Ensure result is a float number
                if isinstance(value, Tensor):
                    batch_metrics[name] = value.item()
                else:
                    batch_metrics[name] = float(value)
            except Exception as e:
                nn_handler.log(f"Rank {nn_handler._rank}: Error calculating metric '{name}': {e}",
                               logging.ERROR)
                batch_metrics[name] = torch.nan  # Indicate error
    return batch_metrics
