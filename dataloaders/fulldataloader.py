import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, fetch_california_housing, load_breast_cancer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
from pathlib import Path
from collections import Counter
import pandas as pd
from sklearn.utils import shuffle
import math
from sklearn.model_selection import StratifiedShuffleSplit


class FullDataLoader:
    def __init__(
        self,
        random_state=4200,
        target_column="target",
        test_size=0.2,
        normalize_features="mean_std",
        return_extra_info=False,
        encode_categorical=False,
        num_targets=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_column = target_column
        self.test_size = test_size
        self.normalize_features = normalize_features
        self.return_extra_info = return_extra_info
        self.encode_categorical = encode_categorical
        self.random_state = random_state
        self.num_targets = num_targets
        # Construct the path to the CSV data file using pathlib]
        self.script_path = Path(__file__).resolve()
        self.data_path = f"{self.script_path.parents[1]}/data/"
        self.igtd_path = f"{self.script_path.parents[1]}/modelsdefinition/IGTD/"

    def load_data(self):
        raise NotImplementedError

    def create_extra_info(self, df, igtd_path, img_rows, img_columns):
        # Get the unique values for each categorical column
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

        with open(igtd_path) as f:
            extra_info["column_ordering"] = list(
                map(int, f.readlines()[-1].strip().split())
            )
        extra_info["img_rows"] = img_rows
        extra_info["img_columns"] = img_columns
        extra_info["num_features"] = len(df.columns)

        return extra_info

    def bin_random_numerical_column(self, train_df, test_df, exclude_cols, bins=10):
        # Select a random numerical column to bin
        num_cols = train_df.select_dtypes(include=[np.number]).columns
        col_to_bin = np.random.choice(
            [col for col in num_cols if col not in exclude_cols]
        )
        print(f"Binning randomly chosen columns {col_to_bin}")
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

    def bin_numerical_columns(self, df, cols_to_bin, num_bins=10):
        # Select a random numerical column to bin
        min_value = 0

        for col_to_bin in cols_to_bin:
            # Calculate the maximum value in the column
            max_value = df[col_to_bin].max()
            # Calculate the width of each bin
            bin_width = math.ceil((max_value - min_value) / num_bins)

            bin_edges = [min_value + i * bin_width for i in range(num_bins + 1)]
            print(bin_edges)

            # Define labels for each category
            labels = [f"category_{i + 1}" for i in range(num_bins)]

            # Use pd.cut to create the categorical column
            df[col_to_bin] = pd.cut(
                df[col_to_bin],
                bins=bin_edges,
                labels=labels,
                include_lowest=True,
                right=True,
            ).astype(str)

        return df

    def _undersample(self, df, target_col, ratio):
        """
        Perform undersampling on a pandas dataframe with a target column
        with values 0 and 1, based on a user-defined ratio parameter.

        Parameters:
            df (pandas.DataFrame): The dataframe to perform undersampling on.
            target_col (str): The name of the target column with values 0 and 1.
            ratio (int): The degree of undersampling, where if the parameter is 1
                then there will be as many 0s as 1s. If it is 6, there will be
                6 times as many 0s as 1s.

        Returns:
            A pandas dataframe with the same column names as the input dataframe,
            but undersampled based on the specified ratio.
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

        Args:
            X (DataFrame): Feature matrix.
            y (Series): Target variable.

        Returns:
            Balanced feature matrix (DataFrame) and target variable (Series).
        """
        # Count the occurrences of each class
        class_counts = Counter(y)

        # Determine the minimum class count
        min_class_count = min(class_counts.values())

        # Create a DataFrame with the same columns as X
        balanced_X = pd.DataFrame(columns=X.columns)

        # Undersample each class to have the same count as the minimum class count
        for class_label in class_counts.keys():
            class_indices = y[y == class_label].index
            class_indices = shuffle(class_indices, random_state=self.random_state)
            class_indices = class_indices[:min_class_count]
            balanced_X = pd.concat([balanced_X, X.loc[class_indices]])

        # Shuffle the balanced dataset
        balanced_X, balanced_y = shuffle(
            balanced_X, y.loc[balanced_X.index], random_state=self.random_state
        )

        return balanced_X, balanced_y

    def scale_features(self, X_train, mode="mean_std"):
        # Get the categorical and numerical columns
        num_cols = X_train.select_dtypes(exclude=["object", "category"]).columns
        # Normalize the features if requested
        if mode == "mean_std":
            # Normalize the features using mean and standard deviation
            scaler = StandardScaler()
            X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        elif mode == "min_max":
            # Normalize the features using min-max scaling
            scaler = MinMaxScaler(feature_range=(0, 1))
            X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        elif mode in ["false", False]:
            pass
        else:
            raise ValueError("Invalid normalization method specified")
        return X_train

    def force_encode_categorical(self, df, exclude_cols=["target"]):
        # Get the categorical and numerical columns
        input_cat_cols = df.select_dtypes(include=["object", "category"]).columns
        if len(input_cat_cols) > 0:
            to_encode = [i for i in input_cat_cols if i not in exclude_cols]
            # Select a random column and convert it to a categorical column with 10 buckets
            # Perform one-hot encoding on the object columns
            encoded_cols = pd.get_dummies(df[to_encode], prefix=to_encode)
            # Concatenate the encoded columns with the non-object columns
            df = pd.concat([df.drop(to_encode, axis=1), encoded_cols], axis=1)
        return df


class FullKaggleAgeConditionsLoader(FullDataLoader):
    def __init__(
        self,
        target_column="target",
        test_size=0.2,
        random_state=42,
        normalize_features="mean_std",
        return_extra_info=False,
        encode_categorical=False,
        num_targets=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.normalize_features = normalize_features
        self.return_extra_info = return_extra_info
        self.encode_categorical = encode_categorical
        self.num_targets = num_targets
        self.filename = (
            f"{self.data_path}kaggle/icr-identify-age-related-conditions/trainX.csv"
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

        df["target"] = df["target"].astype(int)
        ###dtt
        # Extract the features and target variables from the dataset
        X_train = df.drop(columns=[self.target_column])
        y_train = df[self.target_column]

        X_train = self.scale_features(X_train, self.normalize_features)

        extra_info = None
        if self.return_extra_info:
            extra_info = self.create_extra_info(
                X_train,
                img_rows=8,
                img_columns=7,
                igtd_path=f"{self.igtd_path}results/ageconditions_igtd_Euclidean_Euclidean/abs/_index.txt",
            )
            extra_info["num_targets"] = self.num_targets

        return X_train, y_train, extra_info


class FullTitanicDataLoader(FullDataLoader):
    def __init__(
        self,
        target_column="target",
        test_size=0.2,
        random_state=42,
        normalize_features="mean_std",
        return_extra_info=False,
        encode_categorical=False,
        num_targets=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
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
        df["target"] = df["target"].astype(int)
        bool_cols = df.select_dtypes(include=["bool"]).columns
        # Convert boolean columns to 0/1
        for col in bool_cols:
            df[col] = df[col].astype(int)
        df = df.fillna(0)

        if self.encode_categorical:
            df = self.force_encode_categorical(df, exclude_cols=[self.target_column])

        ###dtt
        X_train = df.drop(columns=[self.target_column])
        y_train = df[self.target_column]

        # Normalize the features if requested
        X_train = self.scale_features(X_train, mode=self.normalize_features)

        extra_info = None
        if self.return_extra_info:
            extra_info = self.create_extra_info(
                X_train,
                img_rows=4,
                img_columns=7,
                igtd_path=f"{self.igtd_path}results/titanic_igtd_Euclidean_Euclidean/abs/_index.txt",
            )
        return X_train, y_train, extra_info


class FullBreastCancerDataLoader(FullDataLoader):
    def __init__(
        self,
        target_column="target",
        test_size=0.2,
        random_state=42,
        normalize_features="mean_std",
        return_extra_info=False,
        encode_categorical=False,
        num_targets=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
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
        df["target"] = df["target"].astype(int)

        ###dtt

        # Extract the features and target variables from the dataset
        X_train = df.drop(columns=[self.target_column])
        y_train = df[self.target_column]

        # Normalize the features if requested
        X_train = self.scale_features(X_train, mode=self.normalize_features)

        extra_info = None
        if self.return_extra_info:
            extra_info = self.create_extra_info(
                X_train,
                img_rows=6,
                img_columns=5,
                igtd_path=f"{self.igtd_path}results/breastcancer_igtd_Euclidean_Euclidean/abs/_index.txt",
            )
        return X_train, y_train, extra_info


class FullCreditDataLoader(FullDataLoader):
    def __init__(
        self,
        target_column="target",
        test_size=0.2,
        random_state=42,
        normalize_features="mean_std",
        return_extra_info=False,
        encode_categorical=False,
        num_targets=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
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
        df.reset_index(drop=True, inplace=True)
        df["target"] = df["target"].astype(int)

        # Extract the features and target variables from the dataset
        X_train = df.drop(columns=[self.target_column])
        y_train = df[self.target_column]

        # Normalize the features if requested
        X_train = self.scale_features(X_train, mode=self.normalize_features)

        extra_info = None
        if self.return_extra_info:
            extra_info = self.create_extra_info(
                X_train,
                img_rows=5,
                img_columns=6,
                igtd_path=f"{self.igtd_path}results/creditcard_igtd_Euclidean_Euclidean/abs/_index.txt",
            )

        return X_train, y_train, extra_info


class FullIrisDataLoader(FullDataLoader):
    def __init__(
        self,
        target_column="target",
        test_size=0.2,
        random_state=42,
        normalize_features="mean_std",
        return_extra_info=False,
        encode_categorical=False,
        num_targets=3,
        **kwargs,
    ):
        super().__init__(**kwargs)
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
        df["target"] = df["target"].astype(int)
        ###dtt

        # Extract the features and target variables from the dataset
        X_train = df.drop(columns=[self.target_column])
        y_train = df[self.target_column]

        # Normalize the features if requested
        X_train = self.scale_features(X_train, mode=self.normalize_features)

        extra_info = None
        if self.return_extra_info:
            extra_info = self.create_extra_info(
                X_train,
                img_rows=2,
                img_columns=2,
                igtd_path=f"{self.igtd_path}results/iris_igtd_Euclidean_Euclidean/abs/_index.txt",
            )

        return X_train, y_train, extra_info


class FullCaliforniaHousingDataLoader(FullDataLoader):
    def __init__(
        self,
        target_column="target",
        test_size=0.2,
        random_state=42,
        normalize_features="mean_std",
        return_extra_info=False,
        encode_categorical=False,
        num_targets=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.normalize_features = normalize_features
        self.return_extra_info = return_extra_info
        self.encode_categorical = encode_categorical
        self.num_targets = num_targets
        self.foldername = "housing"
        self.filename = f"{self.data_path}{self.foldername}/cal_housing.csv"

    def load_data(self):
        # Load the California Housing Prices dataset from scikit-learn
        # data = fetch_california_housing()
        # Convert the dataset to a DataFrame
        # df = pd.DataFrame(data.data, columns=data.feature_names)
        # df.to_csv(r"/home/boom/sdev/WTabRun/data/housing/cal_housing.csv")
        #        df[self.target_column] = data.target

        df = pd.read_csv(self.filename)
        df["pop_density"] = df["Population"] / df["AveRooms"]
        df["target"] = df["target"].astype(float)
        # Extract the features and target variables from the dataset
        X_train = df.drop(columns=[self.target_column])
        y_train = df[self.target_column]

        # Normalize the features if requested
        X_train = self.scale_features(X_train, mode=self.normalize_features)

        extra_info = None
        if self.return_extra_info:
            extra_info = self.create_extra_info(
                X_train,
                img_rows=3,
                img_columns=3,
                igtd_path=f"{self.igtd_path}results/housing_igtd_Euclidean_Euclidean/abs/_index.txt",
            )

        return X_train, y_train, extra_info


class FullAdultDataLoader(FullDataLoader):
    def __init__(
        self,
        target_column="target",
        test_size=0.2,
        random_state=42,
        normalize_features="mean_std",
        return_extra_info=False,
        encode_categorical=False,
        num_targets=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_column = "target"
        self.test_size = test_size
        self.random_state = random_state
        self.normalize_features = normalize_features
        self.return_extra_info = return_extra_info
        self.encode_categorical = encode_categorical
        self.num_targets = num_targets
        self.filename = f"{self.data_path}/adult/adult.csv"

    def load_data(self):
        # Load the Adult dataset from UCI Machine Learning Repository

        # data_url = (
        #    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        # )
        # data_columns = [
        #    "age",
        #    "workclass",
        #    "fnlwgt",
        #    "education",
        #    "education-num",
        #    "marital-status",
        #    "occupation",
        #    "relationship",
        #    "race",
        #    "sex",
        #    "capital-gain",
        #    "capital-loss",
        #    "hours-per-week",
        #    "native-country",
        #    "income",
        # ]
        # df = pd.read_csv(data_url, names=data_columns)
        # Create the "target_income" column by replacing values in the "income" column
        # df["target"] = df["income"].str.strip().replace({">50K": 1, "<=50K": 0})
        # df.drop(columns=["income"], inplace=True)

        df = pd.read_csv(self.filename)

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
        df["target"] = df["target"].astype(int)
        if self.encode_categorical:
            df = self.force_encode_categorical(df, exclude_cols=[self.target_column])

        ###dtt

        # Extract the features and target variables from the dataset
        X_train = df.drop(columns=[self.target_column])
        y_train = df[self.target_column]

        # Normalize the features if requested
        X_train = self.scale_features(X_train, mode=self.normalize_features)

        extra_info = None
        if self.return_extra_info:
            extra_info = self.create_extra_info(
                X_train,
                img_rows=9,
                img_columns=12,
                igtd_path=f"{self.igtd_path}results/adult_igtd_Euclidean_Euclidean/abs/_index.txt",
            )

        return X_train, y_train, extra_info


class FullCoverTypeDataLoader(FullDataLoader):
    def __init__(
        self,
        target_column="target",
        test_size=0.2,
        random_state=42,
        normalize_features="mean_std",
        return_extra_info=False,
        encode_categorical=False,
        num_targets=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
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

        sample_size = 300000

        # Initialize StratifiedShuffleSplit
        stratified_split = StratifiedShuffleSplit(
            n_splits=1, test_size=sample_size, random_state=42
        )

        # Generate indices for the stratified sample
        for sample_index, _ in stratified_split.split(df, df["target"]):
            df = df.iloc[sample_index]
        print(f"covertype df {len(df)}")

        # Extract the features and target variables from the dataset
        X_train = df.drop(columns=[self.target_column])
        y_train = df[self.target_column]

        # Normalize the features if requested
        X_train = self.scale_features(X_train, mode=self.normalize_features)

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

        return X_train, y_train, extra_info


class FullHelocDataLoader(FullDataLoader):
    def __init__(
        self,
        target_column="target",
        test_size=0.2,
        random_state=42,
        normalize_features="mean_std",
        return_extra_info=False,
        encode_categorical=False,
        num_targets=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
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
        # Extract the features and target variables from the dataset
        X_train = df.drop(columns=[self.target_column])
        y_train = df[self.target_column]

        # Normalize the features if requested
        X_train = self.scale_features(X_train, mode=self.normalize_features)

        extra_info = None
        if self.return_extra_info:
            extra_info = self.create_extra_info(
                X_train,
                img_rows=6,
                img_columns=4,
                igtd_path=f"{self.igtd_path}results/heloc_igtd_Euclidean_Euclidean/abs/_index.txt",
            )

        return X_train, y_train, extra_info


class FullDiabetesDataLoader(FullDataLoader):
    def __init__(
        self,
        target_column="target",
        test_size=0.2,
        random_state=42,
        normalize_features="mean_std",
        return_extra_info=False,
        encode_categorical=False,
        num_targets=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_column = "target"
        self.test_size = test_size
        self.random_state = random_state
        self.normalize_features = normalize_features
        self.return_extra_info = return_extra_info
        self.encode_categorical = encode_categorical
        self.num_targets = num_targets
        self.filename = f"{self.data_path}/diabetes/diabetic_data.csv"

    def load_data(self):
        # Load the Adult dataset from UCI Machine Learning Repository
        df = pd.read_csv(self.filename)

        target_map = {"NO": 0, "<30": 1, ">30": 0}
        df["target"] = df["readmitted"].map(target_map)
        df["target"] = df["target"].astype(int)
        df.drop(columns=["encounter_id", "patient_nbr", "readmitted"], inplace=True)
        fake_num_cols = [
            "admission_type_id",
            "discharge_disposition_id",
            "admission_source_id",
        ]
        df[fake_num_cols] = df[fake_num_cols].astype(str)

        df["weight"] = df["weight"].apply(lambda x: x.replace("[", "").replace(")", ""))
        df["age"] = df["age"].apply(lambda x: x.replace("[", "").replace(")", ""))
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        numerical_cols = df.select_dtypes(exclude=["object", "category"]).columns
        df[categorical_cols] = df[categorical_cols].fillna(
            df[categorical_cols].mode().iloc[0]
        )
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

        # Set a threshold for value counts
        threshold = 100  # Adjust this threshold as needed

        # Calculate the value counts of the categorical column
        value_counts = df["medical_specialty"].value_counts()
        # Identify categories with counts below the threshold
        categories_to_replace = value_counts[value_counts < threshold].index.tolist()

        # Replace the identified categories with 'Other'
        df["medical_specialty"] = df["medical_specialty"].apply(
            lambda x: "Other" if x in categories_to_replace else x
        )

        cols_to_bin = ["diag_1", "diag_2", "diag_3"]
        for diag_col in cols_to_bin:
            frequency_map = df[diag_col].value_counts().to_dict()
            df[diag_col] = df[diag_col].map(frequency_map)

        df = self.bin_numerical_columns(df, cols_to_bin, num_bins=10)
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        if self.encode_categorical and len(cat_cols) > 0:
            df = self.force_encode_categorical(df, exclude_cols=[self.target_column])

        # Extract the features and target variables from the dataset
        X_train = df.drop(columns=[self.target_column])
        y_train = df[self.target_column]

        # Normalize the features if requested
        X_train = self.scale_features(X_train, mode=self.normalize_features)

        # Balance the training dataset
        # X_train, y_train = self.balance_multiclass_dataset(X_train, y_train)

        extra_info = None
        if self.return_extra_info:
            extra_info = self.create_extra_info(
                X_train,
                img_rows=16,
                img_columns=15,
                igtd_path=f"{self.igtd_path}results/diabetes_igtd_Euclidean_Euclidean/abs/_index.txt",
            )

        return X_train, y_train, extra_info
