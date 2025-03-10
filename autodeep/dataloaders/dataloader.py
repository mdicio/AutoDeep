import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import shuffle

from typing import Optional
from autodeep.modelutils.igtdutilities import table_to_image


class IGTDPreprocessor:
    def __init__(
        self,
        dataset_name: str = None,
        igtd_configs: Dict[str, Dict] = None,
        base_result_dir: str = "igtd",
        exclude_cols: Optional[List[str]] = None,
    ):
        self.dataset_name = dataset_name
        self.igtd_configs = igtd_configs or {
            "img_size": "auto",
            "save_image_size": 3,
            "max_step": 1_000_000,
            "val_step": 1_000,
            "min_gain": 0.01,
            "ordering_methods": {
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
        }
        self.base_result_dir = base_result_dir
        self.exclude_cols = exclude_cols if exclude_cols is not None else []
        self.img_size = self.igtd_configs.get("img_size")
        # Initialize img_rows and img_columns to None
        self.img_rows = None
        self.img_columns = None

        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        self._determine_previous_existence()

    def _determine_previous_existence(self):
        existing_filenames = []
        for config_name, config in self.igtd_configs["ordering_methods"].items():
            result_dir = os.path.join(
                self.base_result_dir, self.dataset_name, config_name
            )
            result_file_name = os.path.join(result_dir, config["error"], "_index.txt")
            if os.path.exists(result_file_name):
                existing_filenames.append(result_file_name)

        if len(existing_filenames) > 1:
            self.logger.info(
                f"Found an already executed IGTD for dataset {self.dataset_name}"
            )
            self.already_run = True
            self.result_file_name = existing_filenames[0]
        else:
            self.already_run = False

    def _auto_determine_img_size(self, X: pd.DataFrame):
        num_features = len(X.columns)
        factors = [
            (r, num_features // r)
            for r in range(1, int(np.sqrt(num_features)) + 1)
            if num_features % r == 0
        ]
        img_rows, img_columns = factors[-1]
        if len(factors) == 1:
            self.logger.warning(
                f"The number of features ({num_features}) is prime. "
                "For better IGTD performance, consider adding or removing columns to allow a more standard grid size."
            )
        return img_rows, img_columns

    def run(self, X: pd.DataFrame) -> None:
        # Always compute and set the image dimensions:
        if self.img_size == "auto":
            self.img_rows, self.img_columns = self._auto_determine_img_size(X)
        elif isinstance(self.img_size, list):
            self.img_rows, self.img_columns = self.img_size
        else:
            raise ValueError(
                "img_size must be either 'auto' or a list [img_rows, img_columns]"
            )

        # Even if IGTD was run before, we now have the dimensions computed.
        if self.already_run:
            return

        for config_name, config in self.igtd_configs["ordering_methods"].items():
            result_dir = os.path.join(
                self.base_result_dir, self.dataset_name, config_name
            )
            os.makedirs(result_dir, exist_ok=True)
            self.logger.info(
                f"Running IGTD ordering for config '{config_name}' in: {result_dir}"
            )
            result_file_name = os.path.join(result_dir, config["error"], "_index.txt")
            self.result_file_name = result_file_name
            table_to_image(
                X,
                [self.img_rows, self.img_columns],
                config["fea_dist_method"],
                config["image_dist_method"],
                self.igtd_configs["save_image_size"],
                self.igtd_configs["max_step"],
                self.igtd_configs["val_step"],
                result_dir,
                config["error"],
                min_gain=self.igtd_configs["min_gain"],
                save_mode="bulk",
                exclude_cols=self.exclude_cols,
            )


class ExtraInfoCreator:
    def __init__(
        self,
        dataset_name,
        run_igtd=False,
        igtd_preprocessor: Optional[IGTDPreprocessor] = None,
    ):
        self.dataset_name = dataset_name
        self.run_igtd = run_igtd
        self.igtd_preprocessor = igtd_preprocessor


class ExtraInfoCreator:
    def __init__(
        self,
        dataset_name,
        run_igtd=False,
        igtd_preprocessor: Optional[IGTDPreprocessor] = None,
    ):
        self.dataset_name = dataset_name
        self.run_igtd = run_igtd
        self.igtd_preprocessor = igtd_preprocessor

    def create_extra_info(self, df: pd.DataFrame, dataset_name: str):
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        num_cols = df.select_dtypes(exclude=["object", "category"]).columns
        
        # Base Dataset Characteristics
        extra_info = {
            "num_samples": len(df),
            "cat_col_names": list(cat_cols),
            "cat_col_idx": list(df.columns.get_indexer(cat_cols)),
            "cat_col_unique_vals": [len(df[col].unique()) for col in cat_cols],
            "num_col_names": list(num_cols),
            "num_col_idx": list(df.columns.get_indexer(num_cols)),
            "num_features": len(df.columns),
        }
        
        # Additional Dataset-Level Statistics (Aggregated)
        num_features = len(num_cols)
        cat_features = len(cat_cols)

        extra_info["cat_num_ratio"] = cat_features / (num_features + 1e-6)  # Avoid division by zero
        extra_info["feature_sparsity"] = df.isnull().mean().mean()  # Percentage of missing values

        # Class Imbalance Calculation (if target column exists)
        if "target" in df.columns and df["target"].nunique() > 1:
            class_counts = df["target"].value_counts().values
            extra_info["class_imbalance_ratio"] = max(class_counts) / sum(class_counts)

        # Aggregate Statistics for Numerical Features (Ensuring Fixed Shape)
        if num_features > 0:
            extra_info["num_col_mean"] = df[num_cols].mean().mean()  # Mean of means
            extra_info["num_col_std"] = df[num_cols].std().mean()  # Mean of stds
            # extra_info["num_col_skew"] = df[num_cols].apply(skew, nan_policy='omit').mean()  # Mean skewness
            # extra_info["num_col_kurtosis"] = df[num_cols].apply(kurtosis, nan_policy='omit').mean()  # Mean kurtosis

        # IGTD Processing (Unchanged from your original code)
        igtd_candidate = None
        if self.run_igtd and self.igtd_preprocessor:
            if not hasattr(self.igtd_preprocessor, "result_file_name"):
                self.igtd_preprocessor.run(df)
            else:
                if (
                    self.igtd_preprocessor.img_rows is None
                    or self.igtd_preprocessor.img_columns is None
                ):
                    (
                        self.igtd_preprocessor.img_rows,
                        self.igtd_preprocessor.img_columns,
                    ) = self.igtd_preprocessor._auto_determine_img_size(df)

            igtd_candidate = self.igtd_preprocessor.result_file_name
            print("IGTD result file path:", igtd_candidate)

            with open(igtd_candidate) as f:
                lines = f.readlines()
                if lines:
                    extra_info["column_ordering"] = list(
                        map(int, lines[-1].strip().split())
                    )

            extra_info["img_rows"] = self.igtd_preprocessor.img_rows
            extra_info["img_columns"] = self.igtd_preprocessor.img_columns
            print("img_rows:", extra_info["img_rows"])
            print("img_columns:", extra_info["img_columns"])

        return extra_info


class DataLoader:
    def __init__(
        self,
        random_state=4200,
        target_column="target",
        test_size=0.2,
        normalize_features="mean_std",
        return_extra_info=False,
        encode_categorical=False,
        
        data_path="data",
        igtd_path="igtd",
    ):
        self.target_column = target_column
        self.test_size = test_size
        self.normalize_features = normalize_features
        self.return_extra_info = return_extra_info
        self.encode_categorical = encode_categorical
        self.random_state = random_state
        
        self.data_path = data_path
        self.igtd_path = igtd_path

    def load_data(self):
        raise NotImplementedError

    def bin_random_numerical_column(self, train_df, test_df, exclude_cols, bins=10):
        # Select a random numerical column to bin
        img_columns = train_df.select_dtypes(include=[np.number]).columns
        col_to_bin = np.random.choice(
            [col for col in img_columns if col not in exclude_cols]
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
        img_columns = X_train.select_dtypes(exclude=["object", "category"]).columns
        if mode == "mean_std":
            scaler = StandardScaler()
            X_train[img_columns] = scaler.fit_transform(X_train[img_columns])
            X_test[img_columns] = scaler.transform(X_test[img_columns])
        elif mode == "min_max":
            scaler = MinMaxScaler(feature_range=(0, 1))
            X_train[img_columns] = scaler.fit_transform(X_train[img_columns])
            X_test[img_columns] = scaler.transform(X_test[img_columns])
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
        dataset_name,
        dataset_path,
        problem_type,
        target_column="target",
        test_size=0.2,
        split_col = None,
        train_value=None,
        test_value=None,
        random_state=42,
        normalize_features="mean_std",
        return_extra_info=False,
        encode_categorical=False,
        
        run_igtd=False,
        igtd_configs: Optional[Dict[str, Dict]] = None,
        igtd_result_base_dir: Optional[str] = "IGTD",
    ):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.problem_type = problem_type
        self.target_column = target_column
        self.split_col = split_col
        self.test_value = test_value
        self.train_value = train_value
        self.test_size = test_size
        self.random_state = random_state
        self.normalize_features = normalize_features
        self.return_extra_info = return_extra_info
        self.encode_categorical = encode_categorical
        
        self.run_igtd = run_igtd
        self.igtd_configs = igtd_configs
        self.igtd_result_base_dir = igtd_result_base_dir

        # Instead of creating an IGTDPreprocessor inline,
        # create an instance of ExtraInfoCreator that will handle IGTD if needed.
        self.extra_info_creator = ExtraInfoCreator(
            dataset_name=self.dataset_name,
            run_igtd=self.run_igtd,
            igtd_preprocessor=(
                IGTDPreprocessor(
                    dataset_name=self.dataset_name,
                    igtd_configs=self.igtd_configs,
                    base_result_dir=self.igtd_result_base_dir,
                )
                if self.run_igtd
                else None
            ),
        )

    def load_data(self):
        df = pd.read_csv(self.dataset_path)

        if self.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found in dataset"
            )

        df = df.astype({col: 'str' for col in df.select_dtypes('bool').columns})

        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(exclude=['number']).columns

        # Fill missing values: 
        # - Numeric columns: fill with median
        # - Categorical columns: fill with mode (most frequent value)
        df[numeric_cols] = df[numeric_cols].apply(lambda col: col.fillna(col.median()))
        df[categorical_cols] = df[categorical_cols].apply(lambda col: col.fillna(col.mode()[0] if not col.mode().empty else 'Unknown'))

        # Split into features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        if self.encode_categorical:
            X = self.force_encode_categorical(X)

        if self.split_col and self.split_col in df.columns:
            if self.train_value is None or self.test_value is None:
                raise ValueError(
                    "When using split_col, you must specify the train_value and test_value."
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

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y if self.problem_type != "regression" else None,
            )

        if self.normalize_features:
            X_train, X_test = self.scale_features(
                X_train, X_test, mode=self.normalize_features
            )

        # Use ExtraInfoCreator to obtain extra info (including IGTD ordering if run)
        extra_info = None
        if self.return_extra_info:
            extra_info = self.extra_info_creator.create_extra_info(
                X_train, self.dataset_name
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
        
    ):

        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.normalize_features = normalize_features
        self.return_extra_info = return_extra_info
        self.encode_categorical = encode_categorical
        
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
        
    ):

        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.normalize_features = normalize_features
        self.return_extra_info = return_extra_info
        self.encode_categorical = encode_categorical
        
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
        
    ):

        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.normalize_features = normalize_features
        self.return_extra_info = return_extra_info
        self.encode_categorical = encode_categorical
        

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
        
    ):

        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.normalize_features = normalize_features
        self.return_extra_info = return_extra_info
        self.encode_categorical = encode_categorical
        

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
        
    ):

        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.normalize_features = normalize_features
        self.return_extra_info = return_extra_info
        self.encode_categorical = encode_categorical
        
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
    ):

        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.normalize_features = normalize_features
        self.return_extra_info = return_extra_info
        self.encode_categorical = encode_categorical
        

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
        
    ):

        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.normalize_features = normalize_features
        self.return_extra_info = return_extra_info
        self.encode_categorical = encode_categorical
        

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
        
    ):

        self.target_column = "target"
        self.test_size = test_size
        self.random_state = random_state
        self.normalize_features = normalize_features
        self.return_extra_info = return_extra_info
        self.encode_categorical = encode_categorical
        

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
        
    ):

        self.target_column = "target"
        self.test_size = test_size
        self.random_state = random_state
        self.normalize_features = normalize_features
        self.return_extra_info = return_extra_info
        self.encode_categorical = encode_categorical
        
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
        
    ):

        self.target_column = "target"
        self.test_size = test_size
        self.random_state = random_state
        self.normalize_features = normalize_features
        self.return_extra_info = return_extra_info
        self.encode_categorical = encode_categorical
        
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
