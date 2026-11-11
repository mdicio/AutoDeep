"""Run an experiment described by a JSON job file, streaming progress to ``events.jsonl``.

Used by the web UI (one subprocess per job, so a crash or a cancel never takes the server
down) but also usable by hand::

    python -m autodeep.jobs path/to/job_dir      # reads job_dir/job.json
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict

JOB_FILE = "job.json"
EVENTS_FILE = "events.jsonl"
STATUS_FILE = "status.json"
LOG_FILE = "run.log"

# keys of job.json forwarded to AutoRunner
RUNNER_KEYS = {"max_evals", "random_state", "timeout_per_model", "max_epochs", "device", "save_models", "model_config"}


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, default=str))
    os.replace(tmp, path)


def run_job(job_dir) -> int:
    job_dir = Path(job_dir).resolve()
    job = json.loads((job_dir / JOB_FILE).read_text())
    events_path, status_path = job_dir / EVENTS_FILE, job_dir / STATUS_FILE
    status = {"state": "starting", "pid": os.getpid(), "started_at": time.time(), "job_id": job.get("id")}
    _write_json(status_path, status)

    cancel = {"requested": False}
    from autodeep.progress import RunCancelled

    def on_sigterm(signum, frame):
        # Training runs in the main thread: raising here interrupts it mid-epoch (like Ctrl+C would),
        # the runner then records the models finished so far and exits cleanly.
        if not cancel["requested"]:
            cancel["requested"] = True
            raise RunCancelled()

    signal.signal(signal.SIGTERM, on_sigterm)

    events = open(events_path, "a", buffering=1)

    def on_event(evt):
        events.write(json.dumps(evt, default=str) + "\n")
        snap = {
            k: evt[k]
            for k in ("eta_seconds", "eta_human", "elapsed_seconds", "progress", "jobs_done", "jobs_total", "trials_done", "trials_total")
            if k in evt
        }
        status.update(snap)
        status["state"] = "running"
        status["last_event"] = evt.get("event")
        if evt.get("event") == "model_start":
            status["current"] = {k: evt.get(k) for k in ("dataset", "model", "max_evals")}
        elif evt.get("event") in ("trial_start", "trial_end", "stage"):
            cur = status.setdefault("current", {})
            cur.update({k: evt.get(k) for k in ("dataset", "model", "trial", "max_evals", "stage", "message") if evt.get(k) is not None})
            if evt.get("event") != "stage":
                cur.pop("stage", None)
        _write_json(status_path, status)
        if cancel["requested"] and evt.get("event") in ("model_start", "trial_start", "trial_end", "stage"):
            raise RunCancelled()

    # everything a run writes (lightning logs, caches...) stays inside the job folder
    os.chdir(job_dir)
    code = 0
    try:
        from autodeep.automl import AutoRunner

        kwargs = {k: v for k, v in job.items() if k in RUNNER_KEYS and v not in (None, "")}
        runner = AutoRunner(datasets=job["datasets"], models=job["models"], output_folder=job_dir / "output", on_event=on_event, **kwargs)
        board = runner.run()
        status["state"] = "cancelled" if cancel["requested"] else "finished"
        status["n_ok"] = int((board["status"] == "ok").sum()) if not board.empty else 0
        status["n_failed"] = int((board["status"] == "failed").sum()) if not board.empty else 0
    except RunCancelled:  # SIGTERM arrived outside the training loop
        status["state"] = "cancelled"
    except Exception as e:  # configuration errors etc.
        status["state"] = "error"
        status["error"] = f"{type(e).__name__}: {e}"
        events.write(json.dumps({"event": "error", "time": time.time(), "message": status["error"], "traceback": traceback.format_exc()}) + "\n")
        traceback.print_exc()
        code = 1
    finally:
        status["finished_at"] = time.time()
        _write_json(status_path, status)
        events.close()
    return code


if __name__ == "__main__":
    sys.exit(run_job(sys.argv[1]))
