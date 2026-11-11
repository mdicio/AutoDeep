"""Tiny event bus used to report progress (trial ended, model finished...).

The :class:`~autodeep.automl.AutoRunner` installs a reporter; model trainers call
:func:`emit`. A reporter can raise :class:`RunCancelled` to stop a run between trials.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

_local = threading.local()


class RunCancelled(Exception):
    """Raised to abort a run (e.g. the user pressed *Stop* in the web UI)."""


def set_reporter(fn: Optional[Callable[[dict], None]]) -> None:
    _local.reporter = fn


def get_reporter() -> Optional[Callable[[dict], None]]:
    return getattr(_local, "reporter", None)


def emit(event: str, **data) -> None:
    fn = get_reporter()
    if fn is None:
        return
    payload = {"event": event, "time": time.time(), **data}
    fn(payload)
