import numpy as np
from hyperopt import hp

from autodeep.modelutils.search import TrialOutcome, n_startup_trials, run_hyperopt


def test_startup_trials_leave_most_of_the_budget_to_tpe():
    assert n_startup_trials(5) == 5
    assert n_startup_trials(25) == 8
    assert n_startup_trials(100) == 10


def test_tpe_concentrates_after_the_random_startup():
    # 1-D bowl with its minimum at x=3 inside [-10, 10]: once TPE takes over, trials move towards it
    def objective(params):
        x = params["x"]
        return TrialOutcome(validation_metrics={"rmse": (x - 3.0) ** 2 + 0.1}, train_metrics={})

    result = run_hyperopt(objective, {"x": hp.uniform("x", -10, 10)}, metric="rmse", max_evals=25, random_state=0)
    xs = [r["params"]["x"] for r in result.trials]
    assert len(xs) == 25
    startup, guided = xs[: n_startup_trials(25)], xs[n_startup_trials(25) :]
    assert np.median(np.abs(np.array(guided) - 3)) < np.median(np.abs(np.array(startup) - 3))
    assert abs(result.best_params["x"] - 3) < 1.0
