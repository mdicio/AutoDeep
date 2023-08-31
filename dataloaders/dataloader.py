import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, fetch_california_housing, load_breast_cancer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
from pathlib import Path


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

    def scale_features(self, X_train, X_test, mode="mean_std"):
        # Get the categorical and numerical columns
        num_cols = X_train.select_dtypes(exclude=["object", "category"]).columns
        # Normalize the features if requested
        if mode == "mean_std":
            # Normalize the features using mean and standard deviation
            scaler = StandardScaler()
            X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
            X_test[num_cols] = scaler.transform(X_test[num_cols])
        elif mode == "min_max":
            # Normalize the features using min-max scaling
            scaler = MinMaxScaler(feature_range=(0, 1))
            X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
            X_test[num_cols] = scaler.transform(X_test[num_cols])
        elif mode == "false":
            pass
        else:
            raise ValueError("Invalid normalization method specified")
        return X_train, X_test

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
        # Load the California Housing Prices dataset from scikit-learn
        data = fetch_california_housing()

        # Convert the dataset to a DataFrame
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df.to_csv(r"/home/boom/sdev/WTabRun/data/housing/cal_housing.csv")
        # df = pd.read_csv(r"/home/boom/sdev/WTabRun/data/housing/cal_housing.csv")
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
