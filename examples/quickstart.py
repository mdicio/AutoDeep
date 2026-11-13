"""AutoDeep from Python in ~25 lines.  Run:  python examples/quickstart.py"""

import pandas as pd

from autodeep import AutoRunner, ResultsAnalyzer

runner = AutoRunner(
    datasets={
        "iris": {"path": "builtin:iris"},  # multiclass, works offline
        "diabetes": {"path": "builtin:diabetes"},  # regression
        # "my_data": {"path": "data/my_data.csv", "target": "label", "drop_columns": ["id"]},
    },
    time_budget="3m",  # models="auto" (default): pick what fits 3 minutes on this machine
    # models=["xgb", "catboost", "mlp", "gandalf"], max_evals=5,   # ...or choose yourself
    output_folder="autodeep_results",
)

print(runner.plan().describe())  # which models, how many trials, and why
print(runner.estimate())  # per-model runtime estimate, before training anything
leaderboard = runner.run()  # trains, tunes and evaluates everything
print(leaderboard)

# The best model of each dataset is saved with its preprocessing, ready to use
model = runner.best_model("iris")  # later: autodeep.load_model("autodeep_results/models/iris/best")
new_rows = pd.DataFrame([{"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2}])
print(model.predict(new_rows), model.predict_proba(new_rows), sep="\n")

# Dig into the saved results (also readable later, from another session)
analyzer = ResultsAnalyzer(runner.results_path)
pd.set_option("display.width", 200)
print(analyzer.view_performance(dataset_name="iris"))
print(analyzer.view_parameters(model_name="xgb"))
