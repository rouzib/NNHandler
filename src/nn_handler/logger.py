import logging
import os
import sys
from enum import Enum

from .utils import on_rank
from .utils.enums import LoggingMode


@on_rank(0)
def initialize_logger(
        logger_name: str,
        mode: LoggingMode = LoggingMode.CONSOLE,
        filename: str = "NNHandler.log",
        file_mode: str = "a",
        level: int = logging.INFO):
    """
    Initializes and returns a logger instance for the given logger_name and configuration.
    Logger is only created if rank == 0. Returns None otherwise.
    """

    logger = logging.getLogger(logger_name)

    if logger.hasHandlers():
        return logger  # Prevent adding multiple handlers

    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter(
        "[%(levelname)s|%(asctime)s|%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    handlers_added = 0
    if mode in [LoggingMode.CONSOLE, LoggingMode.BOTH]:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        handlers_added += 1
    if mode in [LoggingMode.FILE, LoggingMode.BOTH]:
        try:
            log_dir = os.path.dirname(filename)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(filename, mode=file_mode)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            handlers_added += 1
        except OSError as e:
            print(f"WARNING): Failed to create log file handler for {filename}: {e}", file=sys.stderr)

    if handlers_added > 0:
        logger.info(f"Logger initialized (mode: {mode.name}, level: {logging.getLevelName(level)}).")
        return logger
    else:
        print(f"WARNING: Logger initialization requested but no handlers were added.", file=sys.stderr)
        return None
