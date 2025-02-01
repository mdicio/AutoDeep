import logging
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import shuffle

from autodeep.modelutils.igtdutilities import table_to_image


class IGTDPreprocessor:
    """
    A class to run the IGTD ordering algorithm on tabular training data.

    The IGTD algorithm transforms a tabular dataset into an image representation using a given
    ordering of features. This class wraps the call to `table_to_image` from your IGTD library.

    Attributes:
        num_row (int): Number of pixel rows in the image.
        num_col (int): Number of pixel columns in the image.
        save_image_size (int or float): Size (in inches) of saved images.
        max_step (int): Maximum number of iterations for the IGTD algorithm.
        val_step (int): Iterations for convergence validation.
        min_gain (float): Minimum gain threshold for convergence.
        igtd_configs (dict): Default dictionary of IGTD ordering configurations.
        exclude_cols (list): List of column names to exclude from the transformation.
    """

    def __init__(
        self,
        save_image_size: int = 3,
        max_step: int = 1000000,
        val_step: int = 1000,
        min_gain: float = 0.01,
        exclude_cols: Optional[list] = None,
        igtd_configs: Dict[str, Dict] = {
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
        },
        base_result_dir: str = "igtd",
    ):
        self.save_image_size = save_image_size
        self.max_step = max_step
        self.val_step = val_step
        self.min_gain = min_gain
        self.igtd_configs = igtd_configs
        self.base_result_dir = base_result_dir

        self.exclude_cols = exclude_cols if exclude_cols is not None else []
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def _auto_determine_img_size(X):
        num_features = len(X.columns)

        # Find valid (row, col) pairs where row * col = num_features
        factors = [
            (r, num_features // r)
            for r in range(1, int(np.sqrt(num_features)) + 1)
            if num_features % r == 0
        ]
        print(factors)
        # Pick the most square-like option (largest row)
        num_row, num_col = factors[-1]

        # Check if num_features is prime (only divisible by 1 and itself)
        if len(factors) == 1:  # Only (1, num_features) exists
            print(
                f"Warning: The number of features ({num_features}) is prime. "
                "For better IGTD performance, consider adding or removing columns to allow a more standard grid size."
            )

        return num_row, num_col

    def run(
        self,
        X: pd.DataFrame,
        num_col: Optional[int] = None,
        num_row: Optional[int] = None,
        img_size="custom",  # auto
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
        for config_name, config in self.igtd_configs.items():
            # Construct a result directory for this configuration
            result_dir = os.path.join(self.base_result_dir, f"{config_name}")
            os.makedirs(result_dir, exist_ok=True)
            self.logger.info(
                f"Running IGTD ordering for config '{config_name}' in: {result_dir}"
            )

            if img_size == "auto":
                self.num_row, self.num_col = self._auto_determine_img_size(X)

            table_to_image(
                X,
                [num_row, num_col],
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


###############################################################################
# Base DataLoader class with additional utility functions
###############################################################################


class DataLoader:
    def __init__(
        self,
        random_state=4200,
        target_column="target",
        test_size=0.2,
        normalize_features="mean_std",
        return_extra_info=False,
        encode_categorical=False,
        num_targets=None,
        data_path="data",
        igtd_path="igtd",
    ):
        self.target_column = target_column
        self.test_size = test_size
        self.normalize_features = normalize_features
        self.return_extra_info = return_extra_info
        self.encode_categorical = encode_categorical
        self.random_state = random_state
        self.num_targets = num_targets
        self.data_path = data_path
        self.igtd_path = igtd_path

    def load_data(self):
        raise NotImplementedError

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
        num_cols = df.select_dtypes(exclude=["object", "category"]).columns

        cat_unique_vals = [len(df[col].unique()) for col in cat_cols]

        # Create a dictionary to store the column information
        column_info = {
            "cat_col_names": list(cat_cols),
            "cat_col_idx": list(df.columns.get_indexer(cat_cols)),
            "cat_col_unique_vals": cat_unique_vals,
            "num_col_names": list(num_cols),
            "num_col_idx": list(df.columns.get_indexer(num_cols)),
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

    def bin_random_numerical_column(self, train_df, test_df, exclude_cols, bins=10):
        # Select a random numerical column to bin
        num_cols = train_df.select_dtypes(include=[np.number]).columns
        col_to_bin = np.random.choice(
            [col for col in num_cols if col not in exclude_cols]
        )
        print(f"Binning randomly chosen column {col_to_bin}")
        train_df[col_to_bin] = train_df[col_to_bin].fillna(0)
        test_df[col_to_bin] = train_df[col_to_bin].fillna(0)
        # Bin the column on both the training and test sets
        train_df[col_to_bin], bin_values = pd.cut(
            train_df[col_to_bin],
            bins=bins,
            labels=[i for i in range(bins)],
            include_lowest=True,
            right=True,
            retbins=True,
        )
        test_df[col_to_bin] = pd.cut(
            test_df[col_to_bin],
            bins=bin_values,
            labels=[i for i in range(bins)],
            include_lowest=True,
            right=True,
        )
        train_df[col_to_bin] = train_df[col_to_bin].astype(str)
        test_df[col_to_bin] = test_df[col_to_bin].astype(str)

        return train_df, test_df

    def _undersample(self, df, target_col, ratio):
        """
        Perform undersampling on a pandas dataframe with a target column
        with values 0 and 1, based on a user-defined ratio parameter.
        """
        counts = df[target_col].value_counts()
        num_0s = counts[0]
        num_1s = counts[1]
        num_0s_sampled = min(num_0s, num_1s * ratio) if ratio > 0 else num_0s
        df_0s = df[df[target_col] == 0].sample(num_0s_sampled, replace=True)
        df_1s = df[df[target_col] == 1]
        return pd.concat([df_0s, df_1s]).sample(frac=1)

    def balance_multiclass_dataset(self, X, y):
        """
        Undersample the majority class to balance the dataset.
        """
        # Count the occurrences of each class
        from collections import Counter

        class_counts = Counter(y)
        min_class_count = min(class_counts.values())
        balanced_X = pd.DataFrame(columns=X.columns)
        for class_label in class_counts.keys():
            class_indices = y[y == class_label].index
            class_indices = shuffle(class_indices, random_state=self.random_state)
            class_indices = class_indices[:min_class_count]
            balanced_X = pd.concat([balanced_X, X.loc[class_indices]])
        balanced_X, balanced_y = shuffle(
            balanced_X, y.loc[balanced_X.index], random_state=self.random_state
        )
        return balanced_X, balanced_y

    def scale_features(self, X_train, X_test, mode="mean_std"):
        # Get the numerical columns
        num_cols = X_train.select_dtypes(exclude=["object", "category"]).columns
        if mode == "mean_std":
            scaler = StandardScaler()
            X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
            X_test[num_cols] = scaler.transform(X_test[num_cols])
        elif mode == "min_max":
            scaler = MinMaxScaler(feature_range=(0, 1))
            X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
            X_test[num_cols] = scaler.transform(X_test[num_cols])
        elif mode == "false":
            pass
        else:
            raise ValueError("Invalid normalization method specified")
        return X_train, X_test

    def force_encode_categorical(self, df, exclude_cols=["target"]):
        input_cat_cols = df.select_dtypes(include=["object", "category"]).columns
        if len(input_cat_cols) > 0:
            to_encode = [i for i in input_cat_cols if i not in exclude_cols]
            encoded_cols = pd.get_dummies(df[to_encode], prefix=to_encode)
            df = pd.concat([df.drop(to_encode, axis=1), encoded_cols], axis=1)
        return df


###############################################################################
# DynamicDataLoader with IGTD integration
###############################################################################


class DynamicDataLoader(DataLoader):
    def __init__(
        self,
        dataset_path,
        target_column="target",
        test_size=0.2,
        split_col=None,
        train_value=None,
        test_value=None,
        random_state=42,
        normalize_features="mean_std",
        return_extra_info=False,
        encode_categorical=False,
        num_targets=1,
        run_igtd=False,
        igtd_configs: Optional[Dict[str, Dict]] = None,
        igtd_result_base_dir: Optional[str] = "igtd",
    ):
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
        self.run_igtd = False
        self.igtd_configs = igtd_configs  # A dict of ordering configurations
        self.igtd_result_base_dir = igtd_result_base_dir

    def load_data(self):
        df = pd.read_csv(self.dataset_path)

        if self.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found in dataset"
            )

        # Split into features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        # Handle categorical encoding if needed
        if self.encode_categorical:
            X = self.force_encode_categorical(X)

        # Split using split_col if provided; otherwise, use train_test_split
        if self.split_col and self.split_col in df.columns:
            if self.train_value is None or self.test_value is None:
                raise ValueError(
                    "When using split_col, you must specify train_value and test_value."
                )
            unique_values = df[self.split_col].unique()
            if set(unique_values) != {self.train_value, self.test_value}:
                raise ValueError(
                    f"split_col must contain exactly two values: {unique_values}, but expected {self.train_value} and {self.test_value}."
                )
            train_mask = df[self.split_col] == self.train_value
            test_mask = df[self.split_col] == self.test_value
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
        else:
            from sklearn.model_selection import train_test_split

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

        # If IGTD is configured, run the IGTD preprocessor on X_train.
        if self.run_igtd:
            igtd_preprocessor = IGTDPreprocessor(
                igtd_configs=self.igtd_configs
                or None,  # Use provided configs or default
                base_result_dir=self.igtd_result_base_dir
                or "igtd",  # Use provided dir or default
            )

            igtd_results = igtd_preprocessor.run(X_train)
            # Choose the first configuration and construct the full path to the _index.txt file.
            first_config = next(iter(igtd_results))
            igtd_path = os.path.join(igtd_results[first_config], "_index.txt")

        # Optionally return extra info (which includes IGTD ordering info if available)
        extra_info = None
        if self.return_extra_info:
            extra_info = self.create_extra_info(
                df,
                igtd_path=igtd_path,
                img_rows=(
                    self.igtd_preprocessor.num_row if self.igtd_preprocessor else None
                ),
                img_columns=(
                    self.igtd_preprocessor.num_col if self.igtd_preprocessor else None
                ),
            )

        return X_train, X_test, y_train, y_test, extra_info


class KaggleAgeConditionsLoader(DataLoader):
    def __init__(
        self,
        target_column="target",
        test_size=0.2,
        random_state=42,
        normalize_features="mean_std",
        return_extra_info=False,
        encode_categorical=False,
        num_targets=1,
    ):

        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.normalize_features = normalize_features
        self.return_extra_info = return_extra_info
        self.encode_categorical = encode_categorical
        self.num_targets = num_targets
        self.filename = (
            f"{self.data_path}kaggle/icr-identify-age-related-conditions/train.csv"
        )

    def load_data(self):
        # Load the Iris dataset from scikit-learn

        df = pd.read_csv(self.filename).drop(columns=["Id"])
        df = df.rename(columns={"Class": "target"})
        # map the values to 0 and 1
        df["EJ"] = df["EJ"].map({"A": 1, "B": 0})

        # find columns with NaN values
        cols_with_nans = df.columns[df.isna().any()].tolist()

        # fill NaN values in those columns with median
        for col in cols_with_nans:
            df[col] = df[col].fillna(df[col].median())

        if self.encode_categorical:
            df = self.force_encode_categorical(df, exclude_cols=[self.target_column])

        # Split the data into training and test sets
        df_train, df_test = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df[self.target_column],
        )
        # Get the categorical and numerical columns
        input_cat_cols = df_train.select_dtypes(include=["object", "category"]).columns
        print(len(input_cat_cols), input_cat_cols)
        # Check if encoding is disabled and there are no categorical columns
        ###

        # Extract the features and target variables from the dataset
        X_train = df_train.drop(columns=[self.target_column])
        X_test = df_test.drop(columns=[self.target_column])
        y_train = df_train[self.target_column]
        y_test = df_test[self.target_column]

        X_train, X_test = self.scale_features(X_train, X_test, self.normalize_features)

        extra_info = None
        if self.return_extra_info:
            extra_info = self.create_extra_info(
                X_train,
                img_rows=8,
                img_columns=7,
                igtd_path=f"{self.igtd_path}results/ageconditions_igtd_Euclidean_Euclidean/abs/_index.txt",
            )
            extra_info["num_targets"] = self.num_targets

        return X_train, X_test, y_train, y_test, extra_info


class BufixDataLoader(DataLoader):
    def __init__(
        self,
        target_column="target",
        test_size=0.2,
        random_state=42,
        normalize_features="mean_std",
        return_extra_info=False,
        encode_categorical=False,
        num_targets=1,
    ):

        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.normalize_features = normalize_features
        self.return_extra_info = return_extra_info
        self.encode_categorical = encode_categorical
        self.num_targets = num_targets
        self.filename = f"{self.data_path}/buf/sortedbulk_data-1.csv"

    def load_data(self):
        # Load the Iris dataset from scikit-learn

        df = pd.read_csv(self.filename)
        df = df.drop(["num_telefono", "target_event_date", "target_date"], axis=1)
        df_train = df.loc[df["partition_date"] < "2022-04-30"].drop(
            "partition_date", axis=1
        )
        df_train = self._undersample(df_train, "target", 6).reset_index(
            drop=True
        )  # Keep 6 times as many 0s as 1s

        df_test = (
            df.loc[df["partition_date"] >= "2022-04-30"]
            .drop("partition_date", axis=1)
            .reset_index(drop=True)
        )

        # Extract the features and target variables from the dataset
        X_train = df_train.drop(columns=[self.target_column])
        X_test = df_test.drop(columns=[self.target_column])
        y_train = df_train[self.target_column]
        y_test = df_test[self.target_column]

        # Normalize the features if requested
        X_train, X_test = self.scale_features(
            X_train, X_test, mode=self.normalize_features
        )

        extra_info = None
        if self.return_extra_info:
            extra_info = self.create_extra_info(
                X_train,
                img_rows=10,
                img_columns=11,
                igtd_path=f"{self.igtd_path}results/bufix_igtd_Euclidean_Euclidean/abs/_index.txt",
            )

        return X_train, X_test, y_train, y_test, extra_info


class TitanicDataLoader(DataLoader):
    def __init__(
        self,
        target_column="target",
        test_size=0.2,
        random_state=42,
        normalize_features="mean_std",
        return_extra_info=False,
        encode_categorical=False,
        num_targets=1,
    ):

        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.normalize_features = normalize_features
        self.return_extra_info = return_extra_info
        self.encode_categorical = encode_categorical
        self.num_targets = num_targets

    def load_data(self):
        # Load the Titanic dataset from seaborn
        df = sns.load_dataset("titanic")
        # Create a new column called 'family_size' that sums up 'sibsp' and 'parch' and adds 1
        df["family_size"] = df["sibsp"] + df["parch"] + 1
        print(f"titanic df len: {len(df)}")
        print(df["survived"].value_counts())
        # Drop irrelevant columns
        df = df.drop(columns=["who", "adult_male", "alive"])
        # Get all category columns
        category_cols = df.select_dtypes(include=["category"]).columns
        # Convert categories to object dtype
        df[category_cols] = df[category_cols].astype(object)
        # Fill all object columns with "NA"
        object_cols = df.select_dtypes(include=["object"]).columns
        df[object_cols] = df[object_cols].fillna("NA")

        df = df.rename(columns={"survived": "target"})
        bool_cols = df.select_dtypes(include=["bool"]).columns
        # Convert boolean columns to 0/1
        for col in bool_cols:
            df[col] = df[col].astype(int)
        df = df.fillna(0)

        if self.encode_categorical:
            df = self.force_encode_categorical(df, exclude_cols=[self.target_column])

        # Split the data into training and test sets
        df_train, df_test = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df[self.target_column],
        )

        # Extract the features and target variables from the dataset
        X_train = df_train.drop(columns=[self.target_column])
        y_train = df_train[self.target_column]
        X_test = df_test.drop(columns=[self.target_column])
        y_test = df_test[self.target_column]

        # Normalize the features if requested
        X_train, X_test = self.scale_features(
            X_train, X_test, mode=self.normalize_features
        )

        extra_info = None
        if self.return_extra_info:
            extra_info = self.create_extra_info(
                X_train,
                img_rows=4,
                img_columns=7,
                igtd_path=f"{self.igtd_path}results/titanic_igtd_Euclidean_Euclidean/abs/_index.txt",
            )
        return X_train, X_test, y_train, y_test, extra_info


class BreastCancerDataLoader(DataLoader):
    def __init__(
        self,
        target_column="target",
        test_size=0.2,
        random_state=42,
        normalize_features="mean_std",
        return_extra_info=False,
        encode_categorical=False,
        num_targets=1,
    ):

        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.normalize_features = normalize_features
        self.return_extra_info = return_extra_info
        self.encode_categorical = encode_categorical
        self.num_targets = num_targets

    def load_data(self):
        # Load the Breast Cancer Wisconsin (Diagnostic) dataset from sklearn
        data = load_breast_cancer()

        # Convert the dataset to a DataFrame
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df[self.target_column] = data.target

        # Split the data into training and test sets
        df_train, df_test = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df[self.target_column],
        )

        # Extract the features and target variables from the dataset
        X_train = df_train.drop(columns=[self.target_column])
        y_train = df_train[self.target_column]
        X_test = df_test.drop(columns=[self.target_column])
        y_test = df_test[self.target_column]

        # Normalize the features if requested
        X_train, X_test = self.scale_features(
            X_train, X_test, mode=self.normalize_features
        )

        extra_info = None
        if self.return_extra_info:
            extra_info = self.create_extra_info(
                X_train,
                img_rows=6,
                img_columns=5,
                igtd_path=f"{self.igtd_path}results/breastcancer_igtd_Euclidean_Euclidean/abs/_index.txt",
            )
        return X_train, X_test, y_train, y_test, extra_info


class CreditDataLoader(DataLoader):
    def __init__(
        self,
        target_column="target",
        test_size=0.2,
        random_state=42,
        normalize_features="mean_std",
        return_extra_info=False,
        encode_categorical=False,
        num_targets=1,
    ):

        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.normalize_features = normalize_features
        self.return_extra_info = return_extra_info
        self.encode_categorical = encode_categorical
        self.num_targets = num_targets
        self.filename = f"{self.data_path}creditcard/creditcard.csv"

    def load_data(self):
        # Load the Credit Card Fraud Detection dataset from Kaggle
        df = pd.read_csv(self.filename)
        df = df.rename(columns={"Class": self.target_column})

        # Split the data into training and test sets
        df_train, df_test = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df[self.target_column],
        )
        df_train = self._undersample(
            df_train, "target", 6
        )  # Keep 6 times as many 0s as 1s

        df_train.reset_index(drop=True, inplace=True)
        df_test.reset_index(drop=True, inplace=True)

        # Extract the features and target variables from the dataset
        X_train = df_train.drop(columns=[self.target_column])
        y_train = df_train[self.target_column]
        X_test = df_test.drop(columns=[self.target_column])
        y_test = df_test[self.target_column]

        # Normalize the features if requested
        X_train, X_test = self.scale_features(
            X_train, X_test, mode=self.normalize_features
        )

        extra_info = None
        if self.return_extra_info:
            extra_info = self.create_extra_info(
                X_train,
                img_rows=5,
                img_columns=6,
                igtd_path=f"{self.igtd_path}results/creditcard_igtd_Euclidean_Euclidean/abs/_index.txt",
            )

        return X_train, X_test, y_train, y_test, extra_info


class IrisDataLoader(DataLoader):
    def __init__(
        self,
        target_column="target",
        test_size=0.2,
        random_state=42,
        normalize_features="mean_std",
        return_extra_info=False,
        encode_categorical=False,
        num_targets=3,
    ):

        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.normalize_features = normalize_features
        self.return_extra_info = return_extra_info
        self.encode_categorical = encode_categorical
        self.num_targets = num_targets

    def load_data(self):
        # Load the Iris dataset from scikit-learn

        data = load_iris()
        # Convert the dataset to a DataFrame
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df[self.target_column] = data.target

        # Split the data into training and test sets
        df_train, df_test = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df[self.target_column],
        )

        # Extract the features and target variables from the dataset
        X_train = df_train.drop(columns=[self.target_column])
        y_train = df_train[self.target_column]
        X_test = df_test.drop(columns=[self.target_column])
        y_test = df_test[self.target_column]

        # Normalize the features if requested
        X_train, X_test = self.scale_features(
            X_train, X_test, mode=self.normalize_features
        )

        extra_info = None
        if self.return_extra_info:
            extra_info = self.create_extra_info(
                X_train,
                img_rows=2,
                img_columns=2,
                igtd_path=f"{self.igtd_path}results/iris_igtd_Euclidean_Euclidean/abs/_index.txt",
            )

        return X_train, X_test, y_train, y_test, extra_info


class CaliforniaHousingDataLoader(DataLoader):
    def __init__(
        self,
        target_column="target",
        test_size=0.2,
        random_state=42,
        normalize_features="mean_std",
        return_extra_info=False,
        encode_categorical=False,
        num_targets=1,
    ):

        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.normalize_features = normalize_features
        self.return_extra_info = return_extra_info
        self.encode_categorical = encode_categorical
        self.num_targets = num_targets

    def load_data(self):
        # Load the California Housing Prices dataset from scikit-learn
        # data = fetch_california_housing()
        # Convert the dataset to a DataFrame
        # df = pd.DataFrame(data.data, columns=data.feature_names)
        # df.to_csv(r"/home/boom/sdev/WTabRun/data/housing/cal_housing.csv")

        df = pd.read_csv(r"/home/boom/sdev/WTabRun/data/housing/cal_housing.csv")
        df[self.target_column] = data.target

        df["pop_density"] = df["Population"] / df["AveRooms"]

        # Split the data into training and test sets
        df_train, df_test = train_test_split(
            df, test_size=self.test_size, random_state=self.random_state
        )

        # Extract the features and target variables from the dataset
        X_train = df_train.drop(columns=[self.target_column])
        y_train = df_train[self.target_column]
        X_test = df_test.drop(columns=[self.target_column])
        y_test = df_test[self.target_column]

        # Normalize the features if requested
        X_train, X_test = self.scale_features(
            X_train, X_test, mode=self.normalize_features
        )

        extra_info = None
        if self.return_extra_info:
            extra_info = self.create_extra_info(
                X_train,
                img_rows=3,
                img_columns=3,
                igtd_path=f"{self.igtd_path}results/housing_igtd_Euclidean_Euclidean/abs/_index.txt",
            )

        return X_train, X_test, y_train, y_test, extra_info


class AdultDataLoader(DataLoader):
    def __init__(
        self,
        target_column="target",
        test_size=0.2,
        random_state=42,
        normalize_features="mean_std",
        return_extra_info=False,
        encode_categorical=False,
        num_targets=1,
    ):

        self.target_column = "target"
        self.test_size = test_size
        self.random_state = random_state
        self.normalize_features = normalize_features
        self.return_extra_info = return_extra_info
        self.encode_categorical = encode_categorical
        self.num_targets = num_targets

    def load_data(self):
        # Load the Adult dataset from UCI Machine Learning Repository

        df = pd.read_csv(r"/home/boom/sdev/WTabRun/data/adult/adult.csv")
        # Create the "target_income" column by replacing values in the "income" column
        df["target"] = df["income"].str.strip().replace({">50K": 1, "<=50K": 0})
        df.drop(columns=["income"], inplace=True)

        # Fill missing categorical values with the mode
        categorical_cols = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]
        df[categorical_cols] = df[categorical_cols].fillna(
            df[categorical_cols].mode().iloc[0]
        )

        # Fill missing numerical values with the mean
        numerical_cols = [
            "age",
            "fnlwgt",
            "education-num",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
        ]
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

        if self.encode_categorical:
            df = self.force_encode_categorical(df, exclude_cols=[self.target_column])

        # Split the data into training and test sets
        df_train, df_test = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df[self.target_column],
        )

        # Extract the features and target variables from the dataset
        X_train = df_train.drop(columns=[self.target_column])
        y_train = df_train[self.target_column]
        X_test = df_test.drop(columns=[self.target_column])
        y_test = df_test[self.target_column]

        # Normalize the features if requested
        X_train, X_test = self.scale_features(
            X_train, X_test, mode=self.normalize_features
        )

        extra_info = None
        if self.return_extra_info:
            extra_info = self.create_extra_info(
                X_train,
                img_rows=9,
                img_columns=12,
                igtd_path=f"{self.igtd_path}results/adult_igtd_Euclidean_Euclidean/abs/_index.txt",
            )

        return X_train, X_test, y_train, y_test, extra_info


class CoverTypeDataLoader(DataLoader):
    def __init__(
        self,
        target_column="target",
        test_size=0.2,
        random_state=42,
        normalize_features="mean_std",
        return_extra_info=False,
        encode_categorical=False,
        num_targets=1,
    ):

        self.target_column = "target"
        self.test_size = test_size
        self.random_state = random_state
        self.normalize_features = normalize_features
        self.return_extra_info = return_extra_info
        self.encode_categorical = encode_categorical
        self.num_targets = num_targets
        self.filename = f"{self.data_path}/covertype/covtype.data.gz"

    def load_data(self):
        # Load the Adult dataset from UCI Machine Learning Repository
        df = pd.read_csv(self.filename)

        df = df.rename(columns={"5": self.target_column})
        df[self.target_column] = df[self.target_column] - 1
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        if self.encode_categorical and len(cat_cols) > 0:
            df = self.force_encode_categorical(df, exclude_cols=[self.target_column])

        # Split the data into training and test sets
        df_train, df_test = train_test_split(
            df, test_size=self.test_size, random_state=self.random_state
        )

        # Extract the features and target variables from the dataset
        X_train = df_train.drop(columns=[self.target_column])
        y_train = df_train[self.target_column]
        X_test = df_test.drop(columns=[self.target_column])
        y_test = df_test[self.target_column]

        # Normalize the features if requested
        X_train, X_test = self.scale_features(
            X_train, X_test, mode=self.normalize_features
        )

        # Balance the training dataset
        # X_train, y_train = self.balance_multiclass_dataset(X_train, y_train)

        extra_info = None
        if self.return_extra_info:
            extra_info = self.create_extra_info(
                X_train,
                img_rows=6,
                img_columns=9,
                igtd_path=f"{self.igtd_path}results/covertype_igtd_Euclidean_Euclidean/abs/_index.txt",
            )

        return X_train, X_test, y_train, y_test, extra_info


class HelocDataLoader(DataLoader):
    def __init__(
        self,
        target_column="target",
        test_size=0.2,
        random_state=42,
        normalize_features="mean_std",
        return_extra_info=False,
        encode_categorical=False,
        num_targets=1,
    ):

        self.target_column = "target"
        self.test_size = test_size
        self.random_state = random_state
        self.normalize_features = normalize_features
        self.return_extra_info = return_extra_info
        self.encode_categorical = encode_categorical
        self.num_targets = num_targets
        self.filename = f"{self.data_path}heloc/heloc_dataset_v1.csv"

    def load_data(self):
        # Load the Adult dataset from UCI Machine Learning Repository
        df = pd.read_csv(self.filename)

        df = df.rename(columns={"RiskPerformance": self.target_column})
        df["target"] = (df["target"] == "Bad").astype(int)
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        if self.encode_categorical and len(cat_cols) > 0:
            df = self.force_encode_categorical(df, exclude_cols=[self.target_column])

        df["CreditUtilizationRatio"] = (
            df["NetFractionRevolvingBurden"] + df["NetFractionInstallBurden"]
        )

        # Split the data into training and test sets
        df_train, df_test = train_test_split(
            df, test_size=self.test_size, random_state=self.random_state
        )

        # Get the categorical and numerical columns
        input_cat_cols = df_train.select_dtypes(include=["object", "category"]).columns
        print(len(input_cat_cols), input_cat_cols)
        ###

        # Extract the features and target variables from the dataset
        X_train = df_train.drop(columns=[self.target_column])
        y_train = df_train[self.target_column]
        X_test = df_test.drop(columns=[self.target_column])
        y_test = df_test[self.target_column]

        # Normalize the features if requested
        X_train, X_test = self.scale_features(
            X_train, X_test, mode=self.normalize_features
        )

        extra_info = None
        if self.return_extra_info:
            extra_info = self.create_extra_info(
                X_train,
                img_rows=6,
                img_columns=4,
                igtd_path=f"{self.igtd_path}results/heloc_igtd_Euclidean_Euclidean/abs/_index.txt",
            )

        return X_train, X_test, y_train, y_test, extra_info
