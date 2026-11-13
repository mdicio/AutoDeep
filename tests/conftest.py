import os

import numpy as np
import pandas as pd
import pytest

# keep test runs away from the user's timing history / caches and off a possibly busy GPU
os.environ.setdefault("AUTODEEP_DEVICE", "cpu")
os.environ.setdefault("AUTODEEP_SPEED_FACTOR", "1")  # deterministic estimates (no CPU benchmark)


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path_factory, monkeypatch):
    cache = tmp_path_factory.getbasetemp() / "autodeep_cache"
    monkeypatch.setenv("AUTODEEP_CACHE", str(cache))
    import autodeep.estimation as est

    monkeypatch.setattr(est, "HISTORY_FILE", cache / "timings.jsonl")
    import autodeep.automl as automl

    monkeypatch.setattr(automl, "DEFAULT_IGTD_DIR", cache / "igtd")
    yield


@pytest.fixture
def mixed_csv(tmp_path):
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "age [years]": rng.normal(40, 10, n),
            "income": rng.lognormal(10, 1, n),
            "city": rng.choice(["rome", "milan", "turin"], n),
            "member": rng.random(n) > 0.5,
            "split": np.where(np.arange(n) < 240, "train", "test"),
        }
    )
    df.loc[::17, "income"] = np.nan
    df.loc[::23, "city"] = np.nan
    logit = 0.05 * (df["age [years]"] - 40) + (df["city"] == "rome") * 1.0
    df["label"] = np.where(logit + rng.normal(0, 0.5, n) > 0.3, "yes", "no")
    path = tmp_path / "mixed.csv"
    df.to_csv(path, index=False)
    return path
