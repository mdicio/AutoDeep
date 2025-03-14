import ast
import re

import pandas as pd


class ResultsAnalyzer:

    def __init__(self, results_file):
        """__init__

        Args:
        self : type
            Description
        results_file : type
            Description

        Returns:
            type: Description
        """
        self.results_df = pd.read_csv(results_file)
        self.key_cols = ["run_id", "dataset", "model", "best_score"]
        self.normalize_experiments()

    def clean_dict_string(self, dict_string):
        """clean_dict_string

        Args:
        self : type
            Description
        dict_string : type
            Description

        Returns:
            type: Description
        """
        cleaned_string = re.sub("<class '.*'>", '""', dict_string)
        return cleaned_string

    def normalize_dict_column(self, df, col_name, prefix):
        """normalize_dict_column

        Args:
        self : type
            Description
        df : type
            Description
        col_name : type
            Description
        prefix : type
            Description

        Returns:
            type: Description
        """
        df[col_name] = df[col_name].apply(self.clean_dict_string)
        expanded = df[col_name].apply(lambda x: pd.Series(ast.literal_eval(x)))
        expanded.columns = [f"{prefix}_{col}" for col in expanded.columns]
        return pd.concat([df, expanded], axis=1).drop(columns=[col_name])

    def normalize_experiments(self):
        """normalize_experiments

        Args:
        self : type
            Description

        Returns:
            type: Description
        """
        self.results_df = self.normalize_dict_column(
            self.results_df, "train_metrics", "train"
        )
        self.results_df = self.normalize_dict_column(
            self.results_df, "validation_metrics", "validation"
        )
        self.results_df = self.normalize_dict_column(
            self.results_df, "test_metrics", "test"
        )
        self.results_df = self.normalize_dict_column(
            self.results_df, "best_params", "param"
        )

    def get_dynamic_columns(self, column_prefix):
        """get_dynamic_columns

        Args:
        self : type
            Description
        column_prefix : type
            Description

        Returns:
            type: Description
        """
        return [col for col in self.results_df.columns if col.startswith(column_prefix)]

    def filter_by_params(self, dataset_name=None, model_name=None):
        """filter_by_params

        Args:
        self : type
            Description
        dataset_name : type
            Description
        model_name : type
            Description

        Returns:
            type: Description
        """
        df_filtered = self.results_df
        if dataset_name:
            df_filtered = df_filtered[df_filtered["dataset"] == dataset_name]
        if model_name:
            df_filtered = df_filtered[df_filtered["model"] == model_name]
        return df_filtered

    def view_results(self, dataset_name=None, model_name=None):
        """view_results

        Args:
        self : type
            Description
        dataset_name : type
            Description
        model_name : type
            Description

        Returns:
            type: Description
        """
        performance_cols = (
            self.get_dynamic_columns("train_")
            + self.get_dynamic_columns("validation_")
            + self.get_dynamic_columns("test_")
        )
        param_cols = self.get_dynamic_columns("param_")
        df_filtered = self.filter_by_params(
            dataset_name=dataset_name, model_name=model_name
        )
        all_columns = self.key_cols + performance_cols + param_cols
        return df_filtered[all_columns]

    def view_performance(self, dataset_name=None, model_name=None):
        """view_performance

        Args:
        self : type
            Description
        dataset_name : type
            Description
        model_name : type
            Description

        Returns:
            type: Description
        """
        performance_cols = (
            self.get_dynamic_columns("train_")
            + self.get_dynamic_columns("validation_")
            + self.get_dynamic_columns("test_")
        )
        df_filtered = self.filter_by_params(
            dataset_name=dataset_name, model_name=model_name
        )
        return df_filtered[self.key_cols + performance_cols]

    def view_parameters(self, dataset_name=None, model_name=None):
        """view_parameters

        Args:
        self : type
            Description
        dataset_name : type
            Description
        model_name : type
            Description

        Returns:
            type: Description
        """
        param_cols = self.get_dynamic_columns("param_")
        df_filtered = self.filter_by_params(
            dataset_name=dataset_name, model_name=model_name
        )
        return df_filtered[self.key_cols + param_cols]
