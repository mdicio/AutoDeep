"""Logging configuration: one ``autodeep`` logger, quiet third-party libraries."""

from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

LOGGER_NAME = "autodeep"
_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(LOGGER_NAME if not name else f"{LOGGER_NAME}.{name}")


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None, quiet_libraries: bool = True) -> logging.Logger:
    """Configure the ``autodeep`` logger (idempotent)."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    formatter = logging.Formatter(_FMT, datefmt="%H:%M:%S")

    if not any(getattr(h, "_autodeep_console", False) for h in logger.handlers):
        console = logging.StreamHandler(sys.stdout)
        console._autodeep_console = True
        console.setFormatter(formatter)
        logger.addHandler(console)
    for h in logger.handlers:
        if getattr(h, "_autodeep_console", False):
            h.setLevel(getattr(logging, str(level).upper(), logging.INFO))

    if log_file is not None:
        log_file = Path(log_file).resolve()
        for h in list(logger.handlers):
            if getattr(h, "_autodeep_file", False) and Path(h.baseFilename) != log_file:
                logger.removeHandler(h)
                h.close()
        if not any(getattr(h, "_autodeep_file", False) for h in logger.handlers):
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, mode="a")
            fh._autodeep_file = True
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    if quiet_libraries:
        quiet_third_party()
    return logger


def quiet_third_party() -> None:
    os.environ.setdefault("PT_LOGLEVEL", "ERROR")
    for name in ("pytorch_lightning", "lightning", "lightning.pytorch", "lightning_fabric", "pytorch_tabular", "hyperopt", "matplotlib"):
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*does not have many workers.*")
    warnings.filterwarnings("ignore", message=".*num_workers.*")
    warnings.filterwarnings("ignore", message=".*LeafSpec.*")
    warnings.filterwarnings("ignore", message=".*torch.nn.utils.weight_norm.*")
    warnings.filterwarnings("ignore", message=".*Trying to infer the `batch_size`.*")
    warnings.filterwarnings("ignore", message=".*GPU available but not used.*")
    warnings.filterwarnings("ignore", message=".*Checkpoint directory.*exists and is not empty.*")
    warnings.filterwarnings("ignore", message=".*but have no logger configured.*")
