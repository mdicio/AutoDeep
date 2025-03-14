import ast
import re

import pandas as pd


class ResultsAnalyzer:
    def __init__(self, results_file):
        """
        Initialize the ResultsAnalyzer class with a path to the results file (CSV).
        """
        self.results_df = pd.read_csv(results_file)
        self.key_cols = ["run_id", "dataset", "model", "best_score"]
        self.normalize_experiments()

    def clean_dict_string(self, dict_string):
        """
        Sanitize the dictionary string to ensure it can be processed by ast.literal_eval.
        Removes non-literal Python objects like <class 'torch.optim.sgd.SGD'>.
        """
        # Remove non-literal patterns such as <class 'torch.optim.sgd.SGD'>
        cleaned_string = re.sub(r"<class '.*'>", '""', dict_string)
        return cleaned_string

    def normalize_dict_column(self, df, col_name, prefix):
        """
        Expand a column containing dictionaries into separate columns with a specified prefix.
        """
        # Clean the dictionary string before applying literal_eval
        df[col_name] = df[col_name].apply(self.clean_dict_string)

        # Now, safely apply literal_eval on the cleaned string
        expanded = df[col_name].apply(lambda x: pd.Series(ast.literal_eval(x)))

        # Prefix the new columns and return the updated DataFrame
        expanded.columns = [f"{prefix}_{col}" for col in expanded.columns]
        return pd.concat([df, expanded], axis=1).drop(columns=[col_name])

    def normalize_experiments(self):
        """
        Normalize the entire experiments CSV, expanding dictionary columns and returning a fully expanded table.
        """
        # Expand dictionary columns with proper prefixes
        self.results_df = self.normalize_dict_column(self.results_df, "train_metrics", "train")
        self.results_df = self.normalize_dict_column(self.results_df, "validation_metrics", "validation")
        self.results_df = self.normalize_dict_column(self.results_df, "test_metrics", "test")
        self.results_df = self.normalize_dict_column(self.results_df, "best_params", "param")

    def get_dynamic_columns(self, column_prefix):
        """
        Get all columns that start with a specific prefix.
        """
        return [col for col in self.results_df.columns if col.startswith(column_prefix)]

    def filter_by_params(self, dataset_name=None, model_name=None):
        """
        Filter the dataframe by dataset_name and/or model_name.
        """
        df_filtered = self.results_df

        if dataset_name:
            df_filtered = df_filtered[df_filtered["dataset"] == dataset_name]

        if model_name:
            df_filtered = df_filtered[df_filtered["model"] == model_name]

        return df_filtered

    def view_results(self, dataset_name=None, model_name=None):
        """
        View the results including key columns and inferred columns based on the prefix.
        """
        # Get dynamic performance columns and parameter columns
        performance_cols = self.get_dynamic_columns("train_") + self.get_dynamic_columns("validation_") + self.get_dynamic_columns("test_")
        param_cols = self.get_dynamic_columns("param_")

        # Filter by dataset_name and model_name if needed
        df_filtered = self.filter_by_params(dataset_name=dataset_name, model_name=model_name)

        # Combine key columns, performance columns, and parameters columns
        all_columns = self.key_cols + performance_cols + param_cols

        return df_filtered[all_columns]

    def view_performance(self, dataset_name=None, model_name=None):
        """
        View performance columns (e.g., accuracy, best_score) with optional filtering by dataset.
        """
        performance_cols = self.get_dynamic_columns("train_") + self.get_dynamic_columns("validation_") + self.get_dynamic_columns("test_")

        # Filter by dataset_name and model_name if needed
        df_filtered = self.filter_by_params(dataset_name=dataset_name, model_name=model_name)

        # Combine key columns with performance columns
        return df_filtered[self.key_cols + performance_cols]

    def view_parameters(self, dataset_name=None, model_name=None):
        """
        View parameters columns (e.g., best_params) with optional filtering by dataset.
        """
        param_cols = self.get_dynamic_columns("param_")

        # Filter by dataset_name and model_name if needed
        df_filtered = self.filter_by_params(dataset_name=dataset_name, model_name=model_name)

        # Combine key columns with parameters columns
        return df_filtered[self.key_cols + param_cols]
