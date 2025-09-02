import importlib


def import_with_fallback(module_path, class_names, fallback_path="dummy_classes"):
    """
    Dynamically imports classes from the specified module. If the import fails, imports from the fallback module.

    Args:
        module_path (str): The module path to import from, e.g., 'torch.amp' or 'torch_ema'.
        class_names (list[str]): List of class or function names to import.
        fallback_path (str): The fallback (dummy) module, default is 'dummy_classes'.

    Returns:
        Tuple: Tuple containing imported objects in the order of class_names.
        Boolean: True if the import was successful, False otherwise.
    """
    try:
        mod = importlib.import_module(module_path)
        return tuple(getattr(mod, name) for name in class_names), True
    except ImportError:
        dummy_mod = importlib.import_module('.' + fallback_path, package=__package__)
        return tuple(getattr(dummy_mod, name) for name in class_names), False
