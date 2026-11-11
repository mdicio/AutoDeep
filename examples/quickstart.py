"""AutoDeep from Python in ~20 lines.  Run:  python examples/quickstart.py"""

import pandas as pd

from autodeep import AutoRunner, ResultsAnalyzer

runner = AutoRunner(
    datasets={
        "iris": {"path": "builtin:iris"},  # multiclass, works offline
        "diabetes": {"path": "builtin:diabetes"},  # regression
        # "my_data": {"path": "data/my_data.csv", "target": "label", "drop_columns": ["id"]},
    },
    models=["xgb", "catboost", "mlp", "gandalf"],  # or "fast" / "default" / "modern" / "deep" / "all"
    max_evals=5,
    max_epochs=30,
    output_folder="autodeep_results",
)

print(runner.estimate())  # per-model runtime estimate, before training anything
leaderboard = runner.run()  # trains, tunes and evaluates everything
print(leaderboard)

# Dig into the saved results (also readable later, from another session)
analyzer = ResultsAnalyzer(runner.results_path)
pd.set_option("display.width", 200)
print(analyzer.view_performance(dataset_name="iris"))
print(analyzer.view_parameters(model_name="xgb"))
