import math
from enum import Enum

import torch


class TSchedule(Enum):
    """An enumeration representing different types of schedules.

    This class defines various scheduling types that can be used
    in scheduling contexts, such as learning rate schedules,
    event timelines, or other similar use cases. It provides
    predefined schedule types as enum members.

    Attributes:
        LINEAR (str): Represents a linear schedule.
        COSINE (str): Represents a cosine schedule.
    """
    LINEAR = "linear"
    COSINE = "cosine"


def get_schedule_type(sde):
    """
    Determines the type of schedule based on the class name of the provided object.

    This function checks if the provided object (sde) has an attribute '__class__'
    and verifies if its class name matches "VPSDE". If it matches, the function
    returns `TSchedule.COSINE`; otherwise, it returns `TSchedule.LINEAR`.

    Args:
        sde: The object whose schedule type is to be determined. The object
            should have a '__class__' attribute, and its class name is
            evaluated to determine the schedule type.

    Returns:
        TSchedule: A schedule type, which can be either `TSchedule.COSINE` or
            `TSchedule.LINEAR`, based on the evaluation of the object's class name.
    """
    if hasattr(sde, '__class__') and sde.__class__.__name__ == "VPSDE":
        return TSchedule.COSINE
    else:
        return TSchedule.LINEAR


def get_t_schedule(sde, steps, device):
    """
    Computes a t-schedule tensor based on the specified SDE, number of steps, and
    device.

    A t-schedule is a sequence of time values which are computed differently based
    on the schedule type of the provided SDE. This function supports both linear
    and cosine schedule types.

    Args:
        sde (SDE): The stochastic differential equation (SDE) instance, which
            contains the schedule type (linear or cosine), and parameters such
            as T (maximum time) and epsilon (minimum time).
        steps (int): The number of steps for the t-schedule, typically representing
            the discretization levels for solving the SDE.
        device (torch.device): The device where the resulting tensor will be
            allocated, e.g., CPU or GPU.

    Returns:
        torch.Tensor: A tensor of t-schedule values computed based on the schedule
        type of the given SDE.

    Raises:
        ValueError: If the schedule type is not recognized or unsupported.
    """
    schedule_type = get_schedule_type(sde)

    if schedule_type == TSchedule.LINEAR:
        return torch.linspace(sde.T, sde.epsilon, steps + 1, device=device)
    elif schedule_type == TSchedule.COSINE:
        return torch.tensor(
            [sde.epsilon + 0.5 * (sde.T - sde.epsilon) * (1 + math.cos(math.pi * i / steps)) for i in range(steps + 1)],
            device=device)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
