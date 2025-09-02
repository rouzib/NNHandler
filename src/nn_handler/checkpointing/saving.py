import os

import torch


def save_single_file(nn_handler: 'NNHandler', state, path):
    """
    Saves the provided state to a single file at the specified path. Ensures the directory exists
    before attempting to save the file. If saving fails, logs the error and raises an exception.

    :param nn_handler: The NNHandler object that manages error handling and logging operations.
    :param state: The state object to be serialized and saved. Typically includes model weights
        or other relevant data.
    :param path: The file path where the state should be saved. Includes the directory and file name.

    :return: None
    """
    # Ensure directory exists
    try:
        save_dir = os.path.dirname(path)
        if save_dir:  # Only create if path includes a directory
            os.makedirs(save_dir, exist_ok=True)
    except OSError as e:
        nn_handler.raise_error(OSError, f"Could not create directory for saving state to {path}: {e}", e)

    try:
        torch.save(state, path)
        nn_handler.log(f"NNHandler state saved successfully by Rank {nn_handler._rank} to: {path}")
    except Exception as e:
        nn_handler.raise_error(Exception, f"Rank {nn_handler._rank} failed to save NNHandler state to {path}: {e}", e)


def save_multi_files(nn_handler: 'NNHandler', state, path, keys_to_save=None):
    """
    Saves specific components of the neural network state to separate files.

    Each component of the state, as determined by `keys_to_save`, is saved to a
    separate file with `_key` appended to the base filename. The function also
    ensures that directories in the file path exist and logs the success or
    failure of the save operation. If keys like 'seed' or 'sde_class' are present
    in the state, they are used to update the `model_kwargs` attribute of the
    state.

    :param nn_handler: The neural network handler object, responsible for managing
        the operation of the neural network and its states.
    :type nn_handler: NNHandler
    :param state: A dictionary containing the state of the neural network. This
        includes data such as model parameters, optimizer state,
        and additional context-specific information.
    :type state: dict
    :param path: The base path where the state components will be saved. The
        component keys will be appended to the base path for saving individual
        files.
    :type path: str
    :param keys_to_save: A list of keys in the state dictionary to be saved as
        individual files. Defaults to `["model_state_dict", "optimizer_state_dict",
        "model_kwargs"]`.
    :type keys_to_save: list, optional
    :return: None
    """
    if keys_to_save is None:
        keys_to_save = ["model_state_dict", "optimizer_state_dict", "model_kwargs"]

    # Ensure directory exists
    try:
        save_dir = os.path.dirname(path)
        if save_dir:  # Only create if path includes a directory
            os.makedirs(save_dir, exist_ok=True)
    except OSError as e:
        nn_handler.raise_error(OSError, f"Could not create directory for saving state to {path}: {e}", e)

    if state.get("seed") is not None:
        state.get("model_kwargs", {}).update({"seed": state.get("seed")})
    if state.get("sde_class") is not None:
        state.get("model_kwargs", {}).update({"sde_class_name": state.get("sde_class").__name__})
        state.get("model_kwargs", {}).update({"sde_kwargs": state.get("sde_kwargs")})
    state.get("model_kwargs", {}).update({"nn_handler_version": state.get("nn_handler_version")})
    state.get("model_kwargs", {}).update({"pytorch_version": state.get("pytorch_version")})

    try:
        base, ext = os.path.splitext(path)
        for key in keys_to_save:
            value = state.get(key)
            if value is not None:
                save_path = f"{base}_{key}{ext}"
                torch.save(value, save_path)
        nn_handler.log(f"NNHandler state saved successfully by Rank {nn_handler._rank} to: {path}")
    except Exception as e:
        nn_handler.raise_error(Exception, f"Rank {nn_handler._rank} failed to save NNHandler state to {path}: {e}", e)


def save_multi_from_single(path, keys_to_save=None):
    """
    Saves multiple components of a PyTorch model state to separate files, extracting
    and organizing crucial components such as model state, optimizer state, and
    additional metadata. The function allows flexibility in selecting which parts
    of the state to save while ensuring compatibility and directory safety.

    :param path: The file path from which the state will be loaded and also serves
                 as the base for saving separated components.
    :type path: str
    :param keys_to_save: List of specific keys in the state to save. Defaults to
                         ["model_state_dict", "optimizer_state_dict",
                         "model_kwargs"] if not provided.
    :type keys_to_save: list or None
    :return: None
    :rtype: NoneType
    """
    state = torch.load(path, weights_only=False)
    if keys_to_save is None:
        keys_to_save = ["model_state_dict", "optimizer_state_dict", "model_kwargs"]

    # Ensure directory exists
    try:
        save_dir = os.path.dirname(path)
        if save_dir:  # Only create if path includes a directory
            os.makedirs(save_dir, exist_ok=True)
    except OSError as e:
        raise e

    if state.get("seed") is not None:
        state.get("model_kwargs", {}).update({"seed": state.get("seed")})
    if state.get("sde_class") is not None:
        state.get("model_kwargs", {}).update({"sde_class_name": state.get("sde_class").__name__})
        state.get("model_kwargs", {}).update({"sde_kwargs": state.get("sde_kwargs")})
    state.get("model_kwargs", {}).update({"nn_handler_version": state.get("nn_handler_version")})
    state.get("model_kwargs", {}).update({"pytorch_version": state.get("pytorch_version")})

    try:
        base, ext = os.path.splitext(path)
        for key in keys_to_save:
            value = state.get(key)
            if value is not None:
                save_path = f"{base}_{key}{ext}"
                torch.save(value, save_path)
        print(f"NNHandler state saved successfully to: {path}")
    except Exception as e:
        raise e
