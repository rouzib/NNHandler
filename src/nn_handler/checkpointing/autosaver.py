import ast
import os
import sys
import inspect
import warnings
from typing import Optional, Dict, Any, Tuple


class AutoSaver:
    """
    Facilitates auto-saving and managing the saving of model states during training or operation.
    """
    _overwrite_last_saved: bool = False
    _save_interval: Optional[int] = None
    _save_path: Optional[str] = None
    _save_model_name: str = "model_state_epoch"
    _save_model_code: bool = False
    _last_saved_model: Optional[str] = None

    __model_code: Optional[str] = None
    __module_name: Optional[str] = None

    # __module_code removed as it wasn't used and complicates things

    def __init__(self, save_interval: Optional[int] = None, save_path: str = ".",
                 save_model_name: str = "model_state_epoch",
                 overwrite_last_saved: bool = False, save_model_code: bool = False):
        self.save_interval = save_interval  # Use setter for validation/logic
        self.save_path = save_path  # Use setter for validation/logic
        self.save_model_name = save_model_name
        self.overwrite_last_saved = overwrite_last_saved
        self.save_model_code = save_model_code
        self._last_saved_model = None  # Reset on init
        self.__model_code = None
        self.__module_name = None

    # Getter and Setter for _save_interval
    @property
    def save_interval(self) -> Optional[int]:
        return self._save_interval

    @save_interval.setter
    def save_interval(self, value: Optional[int]):
        if value is not None and not isinstance(value, int):
            raise TypeError("save_interval must be an integer or None.")
        if value is not None and value < -1:
            raise ValueError("save_interval must be a non-negative integer, -1, or None.")
        if value == 0:
            self._save_interval = None  # Treat 0 as disabling
        else:
            self._save_interval = value

    # Getter and Setter for _save_path
    @property
    def save_path(self) -> Optional[str]:
        return self._save_path

    @save_path.setter
    def save_path(self, value: Optional[str]):
        if value is None:
            self._save_path = None
            self._save_interval = None  # Disable saving if path is None
        elif isinstance(value, str):
            self._save_path = value
        else:
            raise TypeError("save_path must be a string or None.")

    # Other properties (save_model_name, save_model_code, overwrite_last_saved, last_saved_model) remain similar

    @property
    def save_model_name(self) -> str:
        return self._save_model_name

    @save_model_name.setter
    def save_model_name(self, value: str):
        if not isinstance(value, str):
            raise TypeError("save_model_name must be a string.")
        self._save_model_name = value

    @property
    def save_model_code(self) -> bool:
        return self._save_model_code

    @save_model_code.setter
    def save_model_code(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("save_model_code must be a boolean.")
        self._save_model_code = value

    @property
    def overwrite_last_saved(self) -> bool:
        return self._overwrite_last_saved

    @overwrite_last_saved.setter
    def overwrite_last_saved(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("overwrite_last_saved must be a boolean.")
        self._overwrite_last_saved = value

    @property
    def last_saved_model(self) -> Optional[str]:
        return self._last_saved_model

    @last_saved_model.setter
    def last_saved_model(self, value: Optional[str]):
        if value is not None and not isinstance(value, str):
            raise TypeError("last_saved_model must be a string or None.")
        self._last_saved_model = value

    @property
    def model_code(self) -> Optional[str]:
        return self.__model_code

    def try_save_model_code(self, model_class, logger):
        """Tries to save the source code of a specified model class."""
        if not self._save_model_code:
            return  # Don't warn if not enabled

        try:
            source_file = inspect.getsourcefile(model_class)
            if source_file is None:
                raise OSError("Source file not found (e.g., defined in REPL/notebook).")
            self.__model_code = self.get_imports_from_source_file(model_class) + "\n\n" + inspect.getsource(model_class)
            self.__module_name = model_class.__module__
            if logger:
                logger.info(f"Successfully saved model code from {source_file} for class {model_class.__name__}.")
        except (OSError, TypeError, ValueError) as e:  # Catch more specific errors
            error_message = (f"Model code could not be found or extracted for {model_class.__name__}. "
                             f"Reason: {e}. Loading this model with 'load_from_code=True' will fail.")
            if logger:
                logger.error(error_message)
            else:
                print(f"ERROR: {error_message}", file=sys.stderr)
            self._save_model_code = False  # Disable code saving if it failed
            self.__model_code = None
            self.__module_name = None

    @staticmethod
    def get_imports_from_source_file(class_obj):
        """Retrieves all import statements from the source file of the given class."""
        source_file = inspect.getsourcefile(class_obj)
        if source_file is None:
            raise ValueError(f"Source file not found for class {class_obj.__name__}.")

        with open(source_file, 'r', encoding='utf-8') as f:  # Specify encoding
            source_code = f.read()

        tree = ast.parse(source_code)
        imports = set()  # Use a set to avoid duplicate imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        imports.add(f"import {alias.name} as {alias.asname}")
                    else:
                        imports.add(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                level = node.level  # Handle relative imports
                module_prefix = "." * level
                for alias in node.names:
                    if alias.asname:
                        imports.add(f"from {module_prefix}{module} import {alias.name} as {alias.asname}")
                    else:
                        imports.add(f"from {module_prefix}{module} import {alias.name}")
        # Sort for consistent output
        return "\n".join(sorted(list(imports)))

    # module_code property removed

    def state_dict(self) -> Dict[str, Any]:  # Use state_dict convention
        """Returns the state of the AutoSaver."""
        return {
            "overwrite_last_saved": self._overwrite_last_saved,
            "save_interval": self._save_interval,
            "save_path": self._save_path,
            "save_model_name": self._save_model_name,
            "save_model_code": self._save_model_code,
            "last_saved_model": self._last_saved_model,
            "_model_code": self.__model_code,
            "_module_name": self.__module_name,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):  # Use state_dict convention
        """Loads the state of the AutoSaver."""
        self.overwrite_last_saved = state_dict.get("overwrite_last_saved", False)
        self.save_interval = state_dict.get("save_interval")
        self.save_path = state_dict.get("save_path")
        self.save_model_name = state_dict.get("save_model_name", "model_state_epoch")
        self.save_model_code = state_dict.get("save_model_code", False)
        self.last_saved_model = state_dict.get("last_saved_model")
        self.__model_code = state_dict.get("_model_code")
        self.__module_name = state_dict.get("_module_name")


def load_model_code(path: str) -> Tuple[Optional[str], Optional[str]]:
    """Loads model code and module name from a saved state file (Helper)."""
    # This method is static and doesn't interact with instance state directly.
    # It reads the file path provided.
    from io import BytesIO
    import zipfile
    import pickle

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    # Define a robust unpickler that tries to ignore missing classes/storages
    class RobustUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except (AttributeError, ModuleNotFoundError):
                warnings.warn(f"Could not find class {module}.{name} during code extraction. Replacing with dummy.",
                              RuntimeWarning)
                return type(name, (object,), {'__module__': module})

        def persistent_load(self, pid):
            if isinstance(pid, tuple) and len(pid) > 0 and pid[0] == 'storage':
                try:
                    storage_type, key, location, size = pid[1:]
                    warnings.warn(f"Ignoring PyTorch storage {storage_type} during code extraction.",
                                  RuntimeWarning)
                    return None
                except Exception:
                    warnings.warn(f"Failed to parse persistent_load storage pid: {pid}", RuntimeWarning)
                    return None
            warnings.warn(f"Ignoring unknown persistent_load pid: {pid}", RuntimeWarning)
            return None

    model_code: Optional[str] = None
    module_name: Optional[str] = None

    try:
        # Prefer zipfile format first
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path, 'r') as zip_file:
                if 'data.pkl' in zip_file.namelist():
                    data_bytes = zip_file.read('data.pkl')
                    data_file = BytesIO(data_bytes)
                    unpickler = RobustUnpickler(data_file)
                    obj = unpickler.load()
                    if isinstance(obj, dict) and "auto_saver_state" in obj:
                        auto_saver_state = obj.get("auto_saver_state", {})
                        model_code = auto_saver_state.get("_model_code")
                        module_name = auto_saver_state.get("_module_name")
                else:  # No data.pkl found
                    warnings.warn(f"Zip archive '{path}' does not contain 'data.pkl'. Cannot extract code.",
                                  RuntimeWarning)

        # Fallback to trying direct pickle load (for older formats)
        elif model_code is None:  # Only try if zip failed to yield code
            try:
                with open(path, 'rb') as f:
                    unpickler = RobustUnpickler(f)
                    obj = unpickler.load()
                    auto_saver_state = obj.get("auto_saver_state",
                                               obj.get("auto_saver_kwargs"))  # Check legacy key too
                    if isinstance(auto_saver_state, dict):
                        model_code = auto_saver_state.get("_model_code")
                        module_name = auto_saver_state.get("_module_name")
                    else:
                        warnings.warn(f"Could not find 'auto_saver_state' dict in pickle file '{path}'.",
                                      RuntimeWarning)
            except (pickle.UnpicklingError, EOFError, TypeError) as pickle_err:
                warnings.warn(
                    f"File '{path}' is not a valid zip archive or pickle file for code extraction: {pickle_err}",
                    RuntimeWarning)


    except Exception as e:  # Catch other potential errors during extraction
        warnings.warn(f"Unexpected error extracting code/module name from {path}: {e}. Returning None.",
                      RuntimeWarning)
        return None, None

    return model_code, module_name
