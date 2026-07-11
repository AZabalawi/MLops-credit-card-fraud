"""
utils.py
--------
Shared helper functions used across pipeline stages (prepare, train,
evaluate). Keeping this logic in one place avoids code duplication and
keeps each stage script focused on its single responsibility.
"""

import os
import sys
import yaml
import logging


def load_params(params_path: str = "params.yaml") -> dict:
    """
    Load the params.yaml configuration file.

    Parameters
    ----------
    params_path : str
        Path to the params.yaml file (relative to project root).

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the params file does not exist.
    yaml.YAMLError
        If the file is not valid YAML.
    """
    if not os.path.exists(params_path):
        raise FileNotFoundError(
            f"Could not find '{params_path}'. Make sure you run this "
            f"script from the project root directory."
        )
    with open(params_path, "r") as f:
        try:
            params = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise yaml.YAMLError(f"Error parsing {params_path}: {exc}")
    return params


def get_logger(name: str) -> logging.Logger:
    """
    Create a consistent, readable console logger for pipeline stages.

    Parameters
    ----------
    name : str
        Name of the logger (typically __name__ of the calling module).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:  # avoid duplicate handlers on re-import
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def ensure_dir(path: str) -> None:
    """Create a directory (including parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)
