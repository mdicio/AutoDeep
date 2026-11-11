"""Explore the ``experiments.csv`` written by :class:`~autodeep.automl.AutoRunner`."""

from __future__ import annotations

import ast
import json
import re

import pandas as pd

DICT_COLUMNS = {"train_metrics": "train", "validation_metrics": "validation", "test_metrics": "test", "best_params": "param"}


def _parse(value):
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return {}
    s = str(value)
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        pass
    s = re.sub(r"<class '[^']*\.([^'.]+)'>", r"'\1'", s)  # files written by AutoDeep <= 0.1
    s = re.sub(r"<[^>]+>", "''", s)
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return {}


class ResultsAnalyzer:
    def __init__(self, results_file):
        self.results_df = pd.read_csv(results_file)
        self.key_cols = [c for c in ["run_id", "dataset", "model", "status", "metric", "best_score", "test_score"] if c in self.results_df.columns]
        self.normalize_experiments()

    def normalize_dict_column(self, df, col_name, prefix):
        if col_name not in df.columns:
            return df
        expanded = pd.json_normalize(df[col_name].map(_parse).tolist(), max_level=0)
        expanded.index = df.index
        expanded.columns = [f"{prefix}_{c}" for c in expanded.columns]
        return pd.concat([df.drop(columns=[col_name]), expanded], axis=1)

    def normalize_experiments(self):
        for col, prefix in DICT_COLUMNS.items():
            self.results_df = self.normalize_dict_column(self.results_df, col, prefix)

    def get_dynamic_columns(self, column_prefix):
        return [c for c in self.results_df.columns if c.startswith(column_prefix)]

    def filter_by_params(self, dataset_name=None, model_name=None):
        df = self.results_df
        if dataset_name:
            df = df[df["dataset"] == dataset_name]
        if model_name:
            df = df[df["model"] == model_name]
        return df

    def _perf_cols(self):
        return self.get_dynamic_columns("train_") + self.get_dynamic_columns("validation_") + self.get_dynamic_columns("test_")

    def view_results(self, dataset_name=None, model_name=None):
        return self.filter_by_params(dataset_name, model_name)[self.key_cols + self._perf_cols() + self.get_dynamic_columns("param_")]

    def view_performance(self, dataset_name=None, model_name=None):
        return self.filter_by_params(dataset_name, model_name)[self.key_cols + self._perf_cols()]

    def view_parameters(self, dataset_name=None, model_name=None):
        return self.filter_by_params(dataset_name, model_name)[self.key_cols + self.get_dynamic_columns("param_")]

    def leaderboard(self, split="test"):
        """Best run per (dataset, model), ranked by the dataset's optimisation metric."""
        from autodeep.evaluation.generalevaluator import METRICS

        df = self.results_df
        if "status" in df:
            df = df[df["status"] == "ok"]
        rows = []
        for (dataset, model), g in df.groupby(["dataset", "model"]):
            metric = g["metric"].iloc[0] if "metric" in g else None
            col = f"{split}_{metric}"
            if metric not in METRICS or col not in g:
                continue
            best = g.loc[g[col].idxmax() if METRICS[metric].maximize else g[col].idxmin()]
            rows.append({"dataset": dataset, "model": model, "metric": metric, f"{split}_score": best[col], "run_id": best.get("run_id")})
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        out["_rank_key"] = [(-s if METRICS[m].maximize else s) for s, m in zip(out[f"{split}_score"], out["metric"])]
        return out.sort_values(["dataset", "_rank_key"]).drop(columns="_rank_key").reset_index(drop=True)
