import logging
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
from Scripts.IGTD_Functions import table_to_image


class IGTDPreprocessor:
    """
    A class to run the IGTD ordering algorithm on tabular training data.

    The IGTD algorithm transforms a tabular dataset into an image representation using a given
    ordering of features. This class wraps the call to `table_to_image` from your IGTD library.

    Attributes:
        img_rows (int): Number of pixel rows in the image.
        img_columns (int): Number of pixel columns in the image.
        save_image_size (int or float): Size (in inches) of saved images.
        max_step (int): Maximum number of iterations for the IGTD algorithm.
        val_step (int): Iterations for convergence validation.
        min_gain (float): Minimum gain threshold for convergence.
        exclude_cols (list): List of column names to exclude from the transformation.
    """

    def __init__(
        self,
        img_rows: int,
        img_columns: int,
        save_image_size: int,
        max_step: int,
        val_step: int,
        min_gain: float = 0.01,
        exclude_cols: Optional[list] = None,
    ):
        self.img_rows = img_rows
        self.img_columns = img_columns
        self.save_image_size = save_image_size
        self.max_step = max_step
        self.val_step = val_step
        self.min_gain = min_gain
        self.exclude_cols = exclude_cols if exclude_cols is not None else []
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def run(
        self, X: pd.DataFrame, base_result_dir: str, ordering_configs: Dict[str, Dict]
    ) -> Dict[str, str]:
        """
        Run the IGTD algorithm on the provided DataFrame for each ordering configuration.

        Args:
            X (pd.DataFrame): The training data (features only) on which to run IGTD.
            base_result_dir (str): The base directory where IGTD results will be saved.
            ordering_configs (Dict[str, Dict]): A dictionary mapping configuration names to
                their parameters. Each configuration dict should contain:
                  - "fea_dist_method": (str) method for feature distance calculation.
                  - "image_dist_method": (str) method for pixel distance calculation.
                  - "error": (str) error function (e.g., "abs", "squared").

        Returns:
            Dict[str, str]: A mapping from configuration name to the folder where the IGTD
            result was saved.
        """
        result_dirs = {}
        for config_name, config in ordering_configs.items():
            # Construct a result directory for this configuration
            result_dir = os.path.join(base_result_dir, f"{config_name}")
            os.makedirs(result_dir, exist_ok=True)
            self.logger.info(
                f"Running IGTD ordering for config '{config_name}' in: {result_dir}"
            )

            # Call the table_to_image function with the given parameters.
            # Note: The IGTD function is assumed to save output to result_dir.
            table_to_image(
                X,
                [self.img_rows, self.img_columns],
                config["fea_dist_method"],
                config["image_dist_method"],
                self.save_image_size,
                self.max_step,
                self.val_step,
                result_dir,
                config["error"],
                min_gain=self.min_gain,
                save_mode="bulk",
                exclude_cols=self.exclude_cols,
            )
            result_dirs[config_name] = result_dir

        return result_dirs


# =============================================================================
# Example integration within a custom DataLoader
# =============================================================================

from dataloaders.dataloader import (
    DataLoader,
)  # Assume this is your base DataLoader class
from sklearn.model_selection import train_test_split


class DynamicDataLoader(DataLoader):
    def __init__(
        self,
        dataset_path: str,
        target_column: str = "target",
        test_size: float = 0.2,
        split_col: Optional[str] = None,
        train_value: Optional[str] = None,
        test_value: Optional[str] = None,
        random_state: int = 42,
        normalize_features: Optional[str] = "mean_std",
        return_extra_info: bool = False,
        encode_categorical: bool = False,
        num_targets: int = 1,
        igtd_preprocessor: Optional[IGTDPreprocessor] = None,
        igtd_configs: Optional[Dict[str, Dict]] = None,
        igtd_result_base_dir: Optional[str] = None,
    ):
        """
        Parameters for data loading remain the same. Additionally, if an IGTD preprocessor is provided,
        the IGTD transformation will be applied on the training features.
        """
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.split_col = split_col
        self.train_value = train_value
        self.test_value = test_value
        self.test_size = test_size
        self.random_state = random_state
        self.normalize_features = normalize_features
        self.return_extra_info = return_extra_info
        self.encode_categorical = encode_categorical
        self.num_targets = num_targets

        # IGTD related parameters
        self.igtd_preprocessor = igtd_preprocessor
        self.igtd_configs = igtd_configs  # Dictionary of ordering configurations
        self.igtd_result_base_dir = igtd_result_base_dir

    def load_data(self):
        # Load dataset from CSV
        df = pd.read_csv(self.dataset_path)

        if self.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found in dataset"
            )

        # Split into features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        # Optionally encode categorical features
        if self.encode_categorical:
            X = self.force_encode_categorical(X)

        # Splitting strategy
        if self.split_col and self.split_col in df.columns:
            if self.train_value is None or self.test_value is None:
                raise ValueError(
                    "When using split_col, you must specify train_value and test_value."
                )
            train_mask = df[self.split_col] == self.train_value
            test_mask = df[self.split_col] == self.test_value
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y,
            )

        # Normalize features if requested
        if self.normalize_features:
            X_train, X_test = self.scale_features(
                X_train, X_test, mode=self.normalize_features
            )

        # If an IGTD preprocessor is provided, run the transformation on X_train
        igtd_results = None
        if self.igtd_preprocessor is not None and self.igtd_configs is not None:
            if not self.igtd_result_base_dir:
                raise ValueError(
                    "igtd_result_base_dir must be provided if using the IGTD preprocessor."
                )
            igtd_results = self.igtd_preprocessor.run(
                X_train, self.igtd_result_base_dir, self.igtd_configs
            )

        # Optionally return extra info (could include IGTD result locations)
        extra_info = None
        if self.return_extra_info:
            extra_info = self.create_extra_info(df)
            # You might add the IGTD ordering info to extra_info for later use:
            if igtd_results is not None:
                extra_info["igtd_results"] = igtd_results

        return X_train, X_test, y_train, y_test, extra_info

    # (Implement or inherit force_encode_categorical, scale_features, create_extra_info, etc.)


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    # Define IGTD ordering configurations. For example, two different ordering strategies:

    # Create an instance of the IGTDPreprocessor with desired parameters.
    igtd_preprocessor = IGTDPreprocessor(
        img_rows=8,
        img_columns=7,
        save_image_size=3,
        max_step=1000000,
        val_step=1000,
        min_gain=0.01,
        exclude_cols=[],
    )

    # Set a base directory where IGTD results will be stored.
    igtd_result_base_dir = "./modelsdefinition/IGTD/results/ageconditions_igtd"

    # Create the data loader, passing the IGTD preprocessor and configuration.
    dataset_path = "./data/ageconditions.csv"  # Adjust this path as needed.
    data_loader = DynamicDataLoader(
        dataset_path=dataset_path,
        target_column="target",
        test_size=0.2,
        random_state=4200,
        normalize_features="mean_std",
        encode_categorical=True,
        return_extra_info=True,
        igtd_preprocessor=igtd_preprocessor,
        igtd_configs=igtd_configs,
        igtd_result_base_dir=igtd_result_base_dir,
    )

    # Load data. The IGTD transformation will be run on the training data.
    X_train, X_test, y_train, y_test, extra_info = data_loader.load_data()

    # For demonstration, print out any extra info including IGTD result folders.
    print("Extra Info:", extra_info)


import logging
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
from dataloaders.dataloader import (
    DataLoader,
)  # Assume this is your base DataLoader class
from Scripts.IGTD_Functions import table_to_image
from sklearn.model_selection import train_test_split


class IGTDPreprocessor:
    """
    A class to run the IGTD ordering algorithm on tabular training data.

    The IGTD algorithm transforms a tabular dataset into an image representation using a given
    ordering of features. This class wraps the call to `table_to_image` from your IGTD library.

    Attributes:
        img_rows (int): Number of pixel rows in the image.
        img_columns (int): Number of pixel columns in the image.
        save_image_size (int or float): Size (in inches) of saved images.
        max_step (int): Maximum number of iterations for the IGTD algorithm.
        val_step (int): Iterations for convergence validation.
        min_gain (float): Minimum gain threshold for convergence.
        exclude_cols (list): List of column names to exclude from the transformation.
    """

    def __init__(
        self,
        dataset_name: str,
        img_rows: int,
        img_columns: int,
        save_image_size: int,
        max_step: int,
        val_step: int,
        min_gain: float = 0.01,
        exclude_cols: Optional[list] = None,
    ):
        self.dataset_name = dataset_name
        self.img_rows = img_rows
        self.img_columns = img_columns
        self.save_image_size = save_image_size
        self.max_step = max_step
        self.val_step = val_step
        self.min_gain = min_gain
        self.exclude_cols = exclude_cols if exclude_cols is not None else []
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def run(
        self, X: pd.DataFrame, base_result_dir: str, ordering_configs: Dict[str, Dict]
    ) -> Dict[str, str]:
        """
        Run the IGTD algorithm on the provided DataFrame for each ordering configuration.

        Args:
            X (pd.DataFrame): The training data (features only) on which to run IGTD.
            base_result_dir (str): The base directory where IGTD results will be saved.
            ordering_configs (Dict[str, Dict]): A dictionary mapping configuration names to
                their parameters. Each configuration dict should contain:
                  - "fea_dist_method": (str) method for feature distance calculation.
                  - "image_dist_method": (str) method for pixel distance calculation.
                  - "error": (str) error function (e.g., "abs", "squared").

        Returns:
            Dict[str, str]: A mapping from configuration name to the folder where the IGTD
            result was saved.
        """
        result_dirs = {}
        for config_name, config in ordering_configs.items():
            # Construct a result directory for this configuration
            result_dir = os.path.join(
                base_result_dir, self.dataset_name, f"{config_name}"
            )
            os.makedirs(result_dir, exist_ok=True)
            self.logger.info(
                f"Running IGTD ordering for config '{config_name}' in: {result_dir}"
            )

            # Call the table_to_image function with the given parameters.
            # Note: The IGTD function is assumed to save its output to result_dir.
            table_to_image(
                X,
                [self.img_rows, self.img_columns],
                config["fea_dist_method"],
                config["image_dist_method"],
                self.save_image_size,
                self.max_step,
                self.val_step,
                result_dir,
                config["error"],
                min_gain=self.min_gain,
                save_mode="bulk",
                exclude_cols=self.exclude_cols,
            )
            result_dirs[config_name] = result_dir

        return result_dirs


class DynamicDataLoader(DataLoader):
    def __init__(
        self,
        dataset_path: str,
        target_column: str = "target",
        test_size: float = 0.2,
        split_col: Optional[str] = None,
        train_value: Optional[str] = None,
        test_value: Optional[str] = None,
        random_state: int = 42,
        normalize_features: Optional[str] = "mean_std",
        return_extra_info: bool = False,
        encode_categorical: bool = False,
        num_targets: int = 1,
        igtd_preprocessor: Optional[IGTDPreprocessor] = None,
        igtd_configs: Optional[Dict[str, Dict]] = None,
        igtd_result_base_dir: Optional[str] = None,
    ):
        """
        Parameters for data loading remain the same. Additionally, if an IGTD preprocessor is provided,
        the IGTD transformation will be applied on the training features.
        """
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.split_col = split_col
        self.train_value = train_value
        self.test_value = test_value
        self.test_size = test_size
        self.random_state = random_state
        self.normalize_features = normalize_features
        self.return_extra_info = return_extra_info
        self.encode_categorical = encode_categorical
        self.num_targets = num_targets

        # IGTD related parameters
        self.igtd_preprocessor = igtd_preprocessor
        self.igtd_configs = igtd_configs  # Dictionary of ordering configurations
        self.igtd_result_base_dir = igtd_result_base_dir

        # Set up logger
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def load_data(self):
        # Load dataset from CSV
        df = pd.read_csv(self.dataset_path)

        if self.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found in dataset"
            )

        # Split into features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        # Optionally encode categorical features
        if self.encode_categorical:
            X = self.force_encode_categorical(X)

        # Splitting strategy
        if self.split_col and self.split_col in df.columns:
            if self.train_value is None or self.test_value is None:
                raise ValueError(
                    "When using split_col, you must specify train_value and test_value."
                )
            train_mask = df[self.split_col] == self.train_value
            test_mask = df[self.split_col] == self.test_value
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y,
            )

        # Normalize features if requested
        if self.normalize_features:
            X_train, X_test = self.scale_features(
                X_train, X_test, mode=self.normalize_features
            )

        # Run the IGTD transformation on the training data if a preprocessor is provided.
        igtd_results = None
        if self.igtd_preprocessor is not None and self.igtd_configs is not None:
            if not self.igtd_result_base_dir:
                raise ValueError(
                    "igtd_result_base_dir must be provided if using the IGTD preprocessor."
                )
            igtd_results = self.igtd_preprocessor.run(
                X_train, self.igtd_result_base_dir, self.igtd_configs
            )
            self.logger.info(f"IGTD results: {igtd_results}")

        # Optionally return extra info
        extra_info = None
        if self.return_extra_info:
            # If IGTD was run, pick one ordering configuration (e.g., the first) and assume it saved an ordering file.
            igtd_path = None
            img_rows = None
            img_columns = None
            if igtd_results is not None:
                first_config = next(iter(igtd_results))
                # Assume the IGTD algorithm writes an ordering file called "ordering.txt" in the result directory.
                igtd_path = os.path.join(igtd_results[first_config], "ordering.txt")
                img_rows = self.igtd_preprocessor.img_rows
                img_columns = self.igtd_preprocessor.img_columns

            extra_info = self.create_extra_info(
                df, igtd_path=igtd_path, img_rows=img_rows, img_columns=img_columns
            )

        return X_train, X_test, y_train, y_test, extra_info

    def create_extra_info(self, df, igtd_path=None, img_rows=None, img_columns=None):
        """
        Create extra information about the dataset including:
          - Categorical column names, indices and unique counts.
          - Numerical column names and indices.
          - Total number of features.
          - (Optionally) IGTD ordering information.

        If igtd_path is provided, the method will read the ordering from that file.
        """
        # Get the categorical and numerical columns
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        img_columns = df.select_dtypes(exclude=["object", "category"]).columns

        cat_unique_vals = [len(df[col].unique()) for col in cat_cols]

        # Create a dictionary to store the column information
        column_info = {
            "cat_col_names": list(cat_cols),
            "cat_col_idx": list(df.columns.get_indexer(cat_cols)),
            "cat_col_unique_vals": cat_unique_vals,
            "num_col_names": list(img_columns),
            "num_col_idx": list(df.columns.get_indexer(img_columns)),
        }
        extra_info = column_info
        extra_info["num_features"] = len(df.columns)

        if igtd_path and os.path.exists(igtd_path):
            with open(igtd_path) as f:
                # Read the last line from the file and convert it into a list of integers
                lines = f.readlines()
                if lines:
                    extra_info["column_ordering"] = list(
                        map(int, lines[-1].strip().split())
                    )
            extra_info["img_rows"] = img_rows
            extra_info["img_columns"] = img_columns

        return extra_info

    # Placeholder methods that you might implement/inherit
    def force_encode_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        # Dummy implementation – replace with your actual encoding logic.
        return pd.get_dummies(X)

    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, mode: str):
        # Dummy implementation – replace with your actual normalization/scaling logic.
        if mode == "mean_std":
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns
            mean = X_train[numeric_cols].mean()
            std = X_train[numeric_cols].std().replace(0, 1)
            X_train[numeric_cols] = (X_train[numeric_cols] - mean) / std
            X_test[numeric_cols] = (X_test[numeric_cols] - mean) / std
        return X_train, X_test


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    # Define IGTD ordering configurations. For example, two different ordering strategies:
    igtd_configs = {
        "Euclidean_Euclidean": {
            "fea_dist_method": "Euclidean",
            "image_dist_method": "Euclidean",
            "error": "abs",
        },
        "Pearson_Manhattan": {
            "fea_dist_method": "Pearson",
            "image_dist_method": "Manhattan",
            "error": "squared",
        },
    }

    # Create an instance of the IGTDPreprocessor with desired parameters.
    igtd_preprocessor = IGTDPreprocessor(
        img_rows=8,
        img_columns=7,
        save_image_size=3,
        max_step=1000000,
        val_step=1000,
        min_gain=0.01,
        exclude_cols=[],
    )

    # Set a base directory where IGTD results will be stored.
    igtd_result_base_dir = "./modelsdefinition/IGTD/results/ageconditions_igtd"

    # Create the data loader, passing the IGTD preprocessor and configuration.
    dataset_path = "./data/ageconditions.csv"  # Adjust this path as needed.
    data_loader = DynamicDataLoader(
        dataset_path=dataset_path,
        target_column="target",
        test_size=0.2,
        random_state=4200,
        normalize_features="mean_std",
        encode_categorical=True,
        return_extra_info=True,
        igtd_preprocessor=igtd_preprocessor,
        igtd_configs=igtd_configs,
        igtd_result_base_dir=igtd_result_base_dir,
    )

    # Load data. The IGTD transformation will be run on the training data.
    X_train, X_test, y_train, y_test, extra_info = data_loader.load_data()

    # For demonstration, print out any extra info including IGTD result folders and ordering.
    print("Extra Info:", extra_info)
