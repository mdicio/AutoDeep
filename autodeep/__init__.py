"""AutoDeep: benchmark gradient-boosting and deep-learning models on tabular data.

Quick start::

    from autodeep import AutoRunner

    runner = AutoRunner(datasets={"iris": {"path": "builtin:iris"}}, models="fast", max_evals=10)
    print(runner.estimate())
    print(runner.run())
"""

__version__ = "0.3.0"

from autodeep.datasets import BUILTIN_DATASETS, DatasetSpec, list_builtin_datasets  # noqa: E402
from autodeep.registry import DEFAULT_MODELS, PRESETS, list_models  # noqa: E402


def __getattr__(name):
    # heavy imports (torch, sklearn models...) only when actually needed
    if name == "AutoRunner":
        from autodeep.automl import AutoRunner

        return AutoRunner
    if name == "ResultsAnalyzer":
        from autodeep.outputhandler.resultanalyzer import ResultsAnalyzer

        return ResultsAnalyzer
    raise AttributeError(f"module 'autodeep' has no attribute '{name}'")


__all__ = ["AutoRunner", "DatasetSpec", "ResultsAnalyzer", "list_models", "list_builtin_datasets", "BUILTIN_DATASETS", "DEFAULT_MODELS", "PRESETS"]
