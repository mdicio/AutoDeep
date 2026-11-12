"""Local web app: ``autodeep ui``.

The server is deliberately light: it never imports torch. Each experiment runs in its
own subprocess (``python -m autodeep.jobs <job_dir>``) that streams events to disk.

Workspace layout (default ``./autodeep_workspace``)::

    datasets/            uploaded files
    datasets.json        registered datasets (name, path, target, options)
    runs/<job_id>/       job.json, status.json, events.jsonl, run.log, output/experiments.csv
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

from autodeep import __version__
from autodeep.datasets import SUPPORTED_EXTENSIONS, DatasetSpec, list_builtin_datasets, profile_dataframe, read_table
from autodeep.evaluation.generalevaluator import DEFAULT_EVAL_METRICS, DEFAULT_METRIC, PROBLEM_TYPES, metrics_for_problem
from autodeep.jobs import EVENTS_FILE, JOB_FILE, LOG_FILE, STATUS_FILE
from autodeep.registry import DEFAULT_MODELS, PRESETS, list_models, resolve_models

STATIC_DIR = Path(__file__).parent / "static"
MAX_PROFILE_ROWS = 200_000


# --------------------------------------------------------------------------------------
# request models
# --------------------------------------------------------------------------------------


class DatasetIn(BaseModel):
    name: str
    path: str
    target: str | None = None
    problem_type: str | None = None
    metric: str | None = None
    eval_metrics: list[str] | None = None
    test_size: float = 0.2
    split_col: str | None = None
    train_value: Any | None = None
    test_value: Any | None = None
    drop_columns: list[str] = Field(default_factory=list)
    max_rows: int | None = None


class JobIn(BaseModel):
    name: str | None = None
    datasets: list[DatasetIn]
    models: list[str]
    max_evals: int = 20
    max_epochs: int | None = None
    timeout_per_model: float | None = None
    device: str = "auto"
    random_state: int = 42
    save_models: bool = False
    save_best: bool = True


class PlanIn(BaseModel):
    datasets: list[DatasetIn]
    time_budget: float = 15 * 60  # seconds, whole run
    device: str = "auto"
    max_epochs: int | None = None


class ProfileIn(BaseModel):
    path: str
    target: str | None = None


# --------------------------------------------------------------------------------------
# workspace helpers
# --------------------------------------------------------------------------------------


class Workspace:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.datasets_dir = self.root / "datasets"
        self.runs_dir = self.root / "runs"
        self.registry_file = self.root / "datasets.json"
        for d in (self.datasets_dir, self.runs_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.procs: dict[str, subprocess.Popen] = {}
        self._profile_cache: dict[str, Any] = {}

    # datasets ----------------------------------------------------------------
    def registered(self) -> list[dict]:
        if not self.registry_file.exists():
            return []
        try:
            return json.loads(self.registry_file.read_text())
        except json.JSONDecodeError:
            return []

    def save_registered(self, items: list[dict]) -> None:
        with self.lock:
            self.registry_file.write_text(json.dumps(items, indent=2, default=str))

    def frame(self, path: str) -> pd.DataFrame:
        p = Path(path)
        key = f"{path}:{p.stat().st_mtime if p.exists() else 0}"
        if key not in self._profile_cache:
            df = read_table(path)
            if len(self._profile_cache) > 8:
                self._profile_cache.clear()
            self._profile_cache[key] = df
        return self._profile_cache[key]

    # jobs --------------------------------------------------------------------
    def job_dir(self, job_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", job_id):
            raise HTTPException(400, "invalid job id")
        d = self.runs_dir / job_id
        if not d.exists():
            raise HTTPException(404, f"job {job_id} not found")
        return d

    def job_status(self, job_id: str) -> dict:
        d = self.job_dir(job_id)
        job = json.loads((d / JOB_FILE).read_text())
        status = {}
        if (d / STATUS_FILE).exists():
            try:
                status = json.loads((d / STATUS_FILE).read_text())
            except json.JSONDecodeError:
                status = {}
        proc = self.procs.get(job_id)
        state = status.get("state", "queued")
        if state in ("starting", "running", "queued"):
            alive = proc.poll() is None if proc is not None else _pid_alive(status.get("pid"))
            if not alive:
                state = "crashed" if state != "queued" else "error"
                status["error"] = status.get("error") or "the job process exited unexpectedly - see the log"
        status["state"] = state
        return {"id": job_id, "name": job.get("name"), "created_at": job.get("created_at"), "job": job, "status": status}


def _pid_alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def _safe_filename(name: str) -> str:
    stem, ext = os.path.splitext(os.path.basename(name or "dataset.csv"))
    stem = re.sub(r"[^A-Za-z0-9_\-]+", "_", stem).strip("_") or "dataset"
    return f"{stem}{ext.lower()}"


def _read_events(path: Path, since: int = 0, limit: int = 500) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i < since:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            evt["seq"] = i
            out.append(evt)
            if len(out) >= limit:
                break
    return out


def _best_models(job_dir: Path) -> list[dict]:
    """Saved best model per dataset (read from info.json; the server never unpickles models)."""
    out = []
    for info_file in sorted((job_dir / "output" / "models").glob("*/best/info.json")):
        try:
            info = json.loads(info_file.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        keys = ("dataset", "model", "model_display_name", "problem_type", "target", "classes", "metric", "val_score", "test_score", "input_schema")
        out.append({**{k: info.get(k) for k in keys}, "path": str(info_file.parent.resolve())})
    return out


def _results(job_dir: Path) -> list[dict]:
    path = job_dir / "output" / "experiments.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    rows = []
    for rec in df.to_dict(orient="records"):
        for k in ("train_metrics", "validation_metrics", "test_metrics", "best_params"):
            try:
                rec[k] = json.loads(rec[k]) if isinstance(rec.get(k), str) else {}
            except json.JSONDecodeError:
                rec[k] = {}
        rows.append({k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in rec.items()})
    return rows


# --------------------------------------------------------------------------------------
# app
# --------------------------------------------------------------------------------------


def create_app(workspace: str | None = None) -> FastAPI:
    ws = Workspace(Path(workspace or os.environ.get("AUTODEEP_WORKSPACE", "autodeep_workspace")))
    app = FastAPI(title="AutoDeep", version=__version__)
    app.state.workspace = ws

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse((STATIC_DIR / "index.html").read_text())

    # ---------------------------------------------------------------- catalogue
    @app.get("/api/system")
    def system():
        info = {"version": __version__, "python": sys.version.split()[0], "cpu_count": os.cpu_count(), "workspace": str(ws.root)}
        try:  # torch import is slow-ish, but only on this endpoint
            from autodeep.device import gpu_info, resolve_device

            info["gpu"] = gpu_info()
            info["device"] = resolve_device()
        except Exception as e:  # pragma: no cover
            info["gpu"] = {"available": False, "error": str(e)}
            info["device"] = "cpu"
        return info

    @app.get("/api/models")
    def models():
        return {"models": list_models(), "presets": PRESETS, "default": DEFAULT_MODELS}

    @app.get("/api/metrics")
    def metrics():
        return {
            pt: {"default": DEFAULT_METRIC[pt], "default_eval": DEFAULT_EVAL_METRICS[pt], "available": metrics_for_problem(pt)}
            for pt in PROBLEM_TYPES
        }

    # ---------------------------------------------------------------- datasets
    @app.get("/api/datasets")
    def datasets():
        return {"builtin": list_builtin_datasets(), "registered": ws.registered(), "supported_extensions": list(SUPPORTED_EXTENSIONS)}

    @app.post("/api/datasets/upload")
    async def upload(file: UploadFile = File(...), name: str | None = Form(None)):
        filename = _safe_filename(file.filename)
        if not filename.lower().endswith(SUPPORTED_EXTENSIONS):
            raise HTTPException(400, f"Unsupported file type. Use one of {', '.join(SUPPORTED_EXTENSIONS)}")
        dest = ws.datasets_dir / filename
        i = 1
        while dest.exists():
            i += 1
            dest = ws.datasets_dir / f"{Path(filename).stem}_{i}{Path(filename).suffix}"
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        try:
            df = ws.frame(str(dest))
        except Exception as e:
            dest.unlink(missing_ok=True)
            raise HTTPException(400, f"Could not read the file: {e}")
        ds_name = re.sub(r"[^A-Za-z0-9_]+", "_", name or dest.stem).strip("_") or "dataset"
        return {"name": ds_name, "path": str(dest), "profile": profile_dataframe(df.head(MAX_PROFILE_ROWS))}

    @app.post("/api/datasets/profile")
    def profile(req: ProfileIn):
        try:
            df = ws.frame(req.path) if not req.path.startswith(("builtin:", "openml:")) else read_table(req.path)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        except Exception as e:
            raise HTTPException(400, f"Could not read dataset: {e}")
        return profile_dataframe(df.head(MAX_PROFILE_ROWS), target=req.target)

    @app.post("/api/datasets/validate")
    def validate(ds: DatasetIn):
        try:
            spec = DatasetSpec.from_dict(ds.name, ds.model_dump(exclude={"name"}))
            df = ws.frame(spec.path) if not spec.path.startswith(("builtin:", "openml:")) else read_table(spec.path)
            spec.resolve(df)
        except Exception as e:
            raise HTTPException(400, str(e))
        return {"ok": True, "spec": spec.to_dict()}

    @app.post("/api/datasets/register")
    def register(ds: DatasetIn):
        validate(ds)
        items = [d for d in ws.registered() if d["name"] != ds.name]
        items.append(ds.model_dump())
        ws.save_registered(items)
        return {"ok": True, "registered": items}

    @app.delete("/api/datasets/register/{name}")
    def unregister(name: str):
        items = [d for d in ws.registered() if d["name"] != name]
        ws.save_registered(items)
        return {"ok": True, "registered": items}

    # ---------------------------------------------------------------- estimate & jobs
    def _specs(job: JobIn):
        specs, shapes = [], []
        from autodeep.estimation import shape_from_spec

        for ds in job.datasets:
            spec = DatasetSpec.from_dict(ds.name, ds.model_dump(exclude={"name"}))
            df = ws.frame(spec.path) if not spec.path.startswith(("builtin:", "openml:")) else read_table(spec.path)
            if spec.max_rows and len(df) > spec.max_rows:
                df = df.sample(n=int(spec.max_rows), random_state=0)
            spec.resolve(df)
            specs.append(spec)
            shapes.append(shape_from_spec(spec, df))
        return specs, shapes

    @app.post("/api/estimate")
    def estimate(job: JobIn):
        from autodeep.device import resolve_device, set_device
        from autodeep.estimation import estimate_plan

        if not job.datasets or not job.models:
            raise HTTPException(400, "Select at least one dataset and one model")
        try:
            models = resolve_models(job.models)
            specs, shapes = _specs(job)
        except Exception as e:
            raise HTTPException(400, str(e))
        set_device(job.device)
        from autodeep.machine import current_speed_factor

        plan = estimate_plan(
            shapes,
            models,
            job.max_evals,
            timeout_per_model=job.timeout_per_model,
            max_epochs=job.max_epochs,
            device=resolve_device(),
            speed_factor=current_speed_factor(),
        )
        return {**plan.summary(), "datasets": [s.to_dict() for s in specs]}

    @app.post("/api/plan")
    def auto_plan(req: PlanIn):
        """Recommended models / trials / time limit for these datasets on this machine (models: auto)."""
        from autodeep.autoplan import recommend
        from autodeep.device import resolve_device, set_device
        from autodeep.machine import machine_profile

        if not req.datasets:
            raise HTTPException(400, "Add at least one dataset")
        try:
            _, shapes = _specs(JobIn(datasets=req.datasets, models=["xgb"]))
        except Exception as e:
            raise HTTPException(400, str(e))
        set_device(req.device)
        machine = machine_profile()
        plan = recommend(
            shapes, req.time_budget, device=resolve_device(), speed_factor=machine.speed_factor, max_epochs=req.max_epochs, machine=machine
        )
        return plan.to_dict()

    @app.post("/api/jobs")
    def create_job(job: JobIn):
        if not job.datasets or not job.models:
            raise HTTPException(400, "Select at least one dataset and one model")
        try:
            models = resolve_models(job.models)
            specs, _ = _specs(job)
        except Exception as e:
            raise HTTPException(400, str(e))
        job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        d = ws.runs_dir / job_id
        d.mkdir(parents=True)
        payload = job.model_dump()
        payload.update(
            id=job_id,
            name=job.name or f"{', '.join(s.name for s in specs)} x {len(models)} models",
            created_at=time.time(),
            models=models,
            datasets={s.name: {k: v for k, v in s.to_dict().items() if k != "name"} for s in specs},
        )
        (d / JOB_FILE).write_text(json.dumps(payload, indent=2, default=str))
        log = open(d / LOG_FILE, "w")
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(
            [sys.executable, "-m", "autodeep.jobs", str(d)], stdout=log, stderr=subprocess.STDOUT, cwd=str(d), env=env, start_new_session=True
        )
        ws.procs[job_id] = proc
        threading.Thread(target=proc.wait, daemon=True).start()  # reap the process when it exits (no zombies)
        return {"id": job_id}

    @app.get("/api/jobs")
    def jobs():
        out = []
        for d in sorted(ws.runs_dir.iterdir(), reverse=True):
            if (d / JOB_FILE).exists():
                try:
                    s = ws.job_status(d.name)
                    s.pop("job", None)
                    out.append(s)
                except Exception:
                    continue
        return {"jobs": out}

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str, since: int = 0):
        info = ws.job_status(job_id)
        d = ws.job_dir(job_id)
        info["events"] = _read_events(d / EVENTS_FILE, since=since)
        info["results"] = _results(d)
        info["best_models"] = _best_models(d)
        return info

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        info = ws.job_status(job_id)
        pid = info["status"].get("pid")
        if info["status"]["state"] not in ("running", "starting") or not pid:
            return {"ok": False, "state": info["status"]["state"]}
        try:
            os.kill(int(pid), signal.SIGTERM)  # stops cleanly after the current trial

            job_dir = ws.job_dir(job_id)

            def hard_kill():  # last resort if the process does not stop by itself
                for _ in range(30):
                    time.sleep(1)
                    if not _pid_alive(pid) or (ws.procs.get(job_id) and ws.procs[job_id].poll() is not None):
                        return
                try:
                    os.killpg(os.getpgid(int(pid)), signal.SIGKILL)
                except OSError:
                    pass
                path = job_dir / STATUS_FILE
                try:
                    st = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError):
                    st = {}
                st.update(state="cancelled", finished_at=time.time())
                path.write_text(json.dumps(st))

            threading.Thread(target=hard_kill, daemon=True).start()
        except OSError as e:
            raise HTTPException(500, str(e))
        return {"ok": True}

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str):
        info = ws.job_status(job_id)
        if info["status"]["state"] in ("running", "starting"):
            raise HTTPException(400, "Cancel the job before deleting it")
        shutil.rmtree(ws.job_dir(job_id))
        return {"ok": True}

    @app.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
    def log(job_id: str, tail: int = 300):
        d = ws.job_dir(job_id)
        path = d / "output" / "autodeep.log"
        if not path.exists():
            path = d / LOG_FILE
        if not path.exists():
            return ""
        lines = path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-tail:])

    @app.get("/api/jobs/{job_id}/results.csv")
    def results_csv(job_id: str):
        path = ws.job_dir(job_id) / "output" / "experiments.csv"
        if not path.exists():
            raise HTTPException(404, "no results yet")
        return FileResponse(path, media_type="text/csv", filename=f"autodeep_{job_id}.csv")

    def _model_dir(job_id: str, dataset: str) -> Path:
        base = (ws.job_dir(job_id) / "output" / "models").resolve()
        d = (base / dataset / "best").resolve()
        if base not in d.parents or not (d / "info.json").exists():
            raise HTTPException(404, f"No saved model for dataset '{dataset}'")
        return d

    @app.get("/api/jobs/{job_id}/models/{dataset}/download")
    def download_model(job_id: str, dataset: str):
        import io
        import zipfile

        from fastapi.responses import Response

        d = _model_dir(job_id, dataset)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for f in d.rglob("*"):
                if f.is_file():
                    z.write(f, Path(f"{dataset}_model") / f.relative_to(d))
        name = _safe_filename(f"autodeep_{dataset}_model.zip")
        return Response(buf.getvalue(), media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @app.post("/api/jobs/{job_id}/models/{dataset}/predict")
    async def predict(job_id: str, dataset: str, file: UploadFile = File(...)):
        d = _model_dir(job_id, dataset)
        suffix = Path(file.filename or "rows.csv").suffix.lower() or ".csv"
        if suffix not in SUPPORTED_EXTENSIONS:
            raise HTTPException(400, f"Unsupported file type '{suffix}'")
        work = ws.job_dir(job_id) / "predictions"
        work.mkdir(exist_ok=True)
        stem = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        src, dst = work / f"{stem}_input{suffix}", work / f"{stem}_{_safe_filename(dataset)}_predictions.csv"
        src.write_bytes(await file.read())
        # prediction loads torch: run it in a subprocess so the server stays light
        proc = subprocess.run(
            [sys.executable, "-m", "autodeep.cli", "predict", str(d), str(src), "-o", str(dst)], capture_output=True, text=True, timeout=1800
        )
        if proc.returncode != 0 or not dst.exists():
            lines = [ln for ln in (proc.stderr or "").strip().splitlines() if ln.strip()]
            raise HTTPException(400, lines[-1] if lines else "prediction failed")
        return FileResponse(dst, media_type="text/csv", filename=f"{_safe_filename(dataset)}_predictions.csv")

    return app


DEFAULT_PORT = 8765


def _port_free(host: str, port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def serve(host: str = "127.0.0.1", port: int | None = None, workspace: str | None = None, open_browser: bool = True):
    import uvicorn

    if port is None:  # default port busy (another app?) -> take the next free one
        port = next((p for p in range(DEFAULT_PORT, DEFAULT_PORT + 50) if _port_free(host, p)), DEFAULT_PORT)
    elif not _port_free(host, port):
        raise SystemExit(f"Port {port} is already in use. Pick another one with --port, or omit --port to choose a free one automatically.")
    app = create_app(workspace)
    url = f"http://{host if host not in ('0.0.0.0', '::') else '127.0.0.1'}:{port}"
    print(f"AutoDeep UI running at {url}  (workspace: {app.state.workspace.root})  - press Ctrl+C to stop", flush=True)
    if open_browser:
        import webbrowser

        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
