"""Tiny event bus used to report progress (trial ended, model finished...).

The :class:`~autodeep.automl.AutoRunner` installs a reporter; model trainers call
:func:`emit`. A reporter can raise :class:`RunCancelled` to stop a run between trials.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

_local = threading.local()


class RunCancelled(Exception):
    """Raised to abort a run (e.g. the user pressed *Stop* in the web UI)."""


def set_reporter(fn: Callable[[dict], None] | None) -> None:
    _local.reporter = fn


def get_reporter() -> Callable[[dict], None] | None:
    return getattr(_local, "reporter", None)


def emit(event: str, **data) -> None:
    fn = get_reporter()
    if fn is None:
        return
    payload = {"event": event, "time": time.time(), **data}
    fn(payload)
