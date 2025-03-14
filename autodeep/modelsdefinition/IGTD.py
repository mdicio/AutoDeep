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
        """__init__

        Args:
        self : type
            Description
        img_rows : type
            Description
        img_columns : type
            Description
        save_image_size : type
            Description
        max_step : type
            Description
        val_step : type
            Description
        min_gain : type
            Description
        exclude_cols : type
            Description

        Returns:
            type: Description
        """
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
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def run(self, X: pd.DataFrame, base_result_dir: str, ordering_configs: Dict[str, Dict]) -> Dict[str, str]:
        """run

        Args:
        self : type
            Description
        X : type
            Description
        base_result_dir : type
            Description
        ordering_configs : type
            Description

        Returns:
            type: Description
        """
        result_dirs = {}
        for config_name, config in ordering_configs.items():
            result_dir = os.path.join(base_result_dir, f'{config_name}')
            os.makedirs(result_dir, exist_ok=True)
            self.logger.info(f"Running IGTD ordering for config '{config_name}' in: {result_dir}")
            table_to_image(
                X,
                [self.img_rows, self.img_columns],
                config['fea_dist_method'],
                config['image_dist_method'],
                self.save_image_size,
                self.max_step,
                self.val_step,
                result_dir,
                config['error'],
                min_gain=self.min_gain,
                save_mode='bulk',
                exclude_cols=self.exclude_cols,
            )
            result_dirs[config_name] = result_dir
        return result_dirs


from dataloaders.dataloader import DataLoader
from sklearn.model_selection import train_test_split


class DynamicDataLoader(DataLoader):

    def __init__(
        self,
        dataset_path: str,
        target_column: str = 'target',
        test_size: float = 0.2,
        split_col: Optional[str] = None,
        train_value: Optional[str] = None,
        test_value: Optional[str] = None,
        random_state: int = 42,
        normalize_features: Optional[str] = 'mean_std',
        return_extra_info: bool = False,
        encode_categorical: bool = False,
        igtd_preprocessor: Optional[IGTDPreprocessor] = None,
        igtd_configs: Optional[Dict[str, Dict]] = None,
        igtd_result_base_dir: Optional[str] = None,
    ):
        """__init__

        Args:
        self : type
            Description
        dataset_path : type
            Description
        target_column : type
            Description
        test_size : type
            Description
        split_col : type
            Description
        train_value : type
            Description
        test_value : type
            Description
        random_state : type
            Description
        normalize_features : type
            Description
        return_extra_info : type
            Description
        encode_categorical : type
            Description
        igtd_preprocessor : type
            Description
        igtd_configs : type
            Description
        igtd_result_base_dir : type
            Description

        Returns:
            type: Description
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
        self.igtd_preprocessor = igtd_preprocessor
        self.igtd_configs = igtd_configs
        self.igtd_result_base_dir = igtd_result_base_dir

    def load_data(self):
        """load_data

        Args:
        self : type
            Description

        Returns:
            type: Description
        """
        df = pd.read_csv(self.dataset_path)
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset")
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        if self.encode_categorical:
            X = self.force_encode_categorical(X)
        if self.split_col and self.split_col in df.columns:
            if self.train_value is None or self.test_value is None:
                raise ValueError('When using split_col, you must specify train_value and test_value.')
            train_mask = df[self.split_col] == self.train_value
            test_mask = df[self.split_col] == self.test_value
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state, stratify=y)
        if self.normalize_features:
            X_train, X_test = self.scale_features(X_train, X_test, mode=self.normalize_features)
        igtd_results = None
        if self.igtd_preprocessor is not None and self.igtd_configs is not None:
            if not self.igtd_result_base_dir:
                raise ValueError('igtd_result_base_dir must be provided if using the IGTD preprocessor.')
            igtd_results = self.igtd_preprocessor.run(X_train, self.igtd_result_base_dir, self.igtd_configs)
        extra_info = None
        if self.return_extra_info:
            extra_info = self.create_extra_info(df)
            if igtd_results is not None:
                extra_info['igtd_results'] = igtd_results
        return X_train, X_test, y_train, y_test, extra_info


if __name__ == '__main__':
    igtd_preprocessor = IGTDPreprocessor(
        img_rows=8, img_columns=7, save_image_size=3, max_step=1000000, val_step=1000, min_gain=0.01, exclude_cols=[]
    )
    igtd_result_base_dir = './modelsdefinition/IGTD/results/ageconditions_igtd'
    dataset_path = './data/ageconditions.csv'
    data_loader = DynamicDataLoader(
        dataset_path=dataset_path,
        target_column='target',
        test_size=0.2,
        random_state=4200,
        normalize_features='mean_std',
        encode_categorical=True,
        return_extra_info=True,
        igtd_preprocessor=igtd_preprocessor,
        igtd_configs=igtd_configs,
        igtd_result_base_dir=igtd_result_base_dir,
    )
    X_train, X_test, y_train, y_test, extra_info = data_loader.load_data()
    print('Extra Info:', extra_info)
import logging
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
from dataloaders.dataloader import DataLoader
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
        """__init__

        Args:
        self : type
            Description
        dataset_name : type
            Description
        img_rows : type
            Description
        img_columns : type
            Description
        save_image_size : type
            Description
        max_step : type
            Description
        val_step : type
            Description
        min_gain : type
            Description
        exclude_cols : type
            Description

        Returns:
            type: Description
        """
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
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def run(self, X: pd.DataFrame, base_result_dir: str, ordering_configs: Dict[str, Dict]) -> Dict[str, str]:
        """run

        Args:
        self : type
            Description
        X : type
            Description
        base_result_dir : type
            Description
        ordering_configs : type
            Description

        Returns:
            type: Description
        """
        result_dirs = {}
        for config_name, config in ordering_configs.items():
            result_dir = os.path.join(base_result_dir, self.dataset_name, f'{config_name}')
            os.makedirs(result_dir, exist_ok=True)
            self.logger.info(f"Running IGTD ordering for config '{config_name}' in: {result_dir}")
            table_to_image(
                X,
                [self.img_rows, self.img_columns],
                config['fea_dist_method'],
                config['image_dist_method'],
                self.save_image_size,
                self.max_step,
                self.val_step,
                result_dir,
                config['error'],
                min_gain=self.min_gain,
                save_mode='bulk',
                exclude_cols=self.exclude_cols,
            )
            result_dirs[config_name] = result_dir
        return result_dirs


class DynamicDataLoader(DataLoader):

    def __init__(
        self,
        dataset_path: str,
        target_column: str = 'target',
        test_size: float = 0.2,
        split_col: Optional[str] = None,
        train_value: Optional[str] = None,
        test_value: Optional[str] = None,
        random_state: int = 42,
        normalize_features: Optional[str] = 'mean_std',
        return_extra_info: bool = False,
        encode_categorical: bool = False,
        igtd_preprocessor: Optional[IGTDPreprocessor] = None,
        igtd_configs: Optional[Dict[str, Dict]] = None,
        igtd_result_base_dir: Optional[str] = None,
    ):
        """__init__

        Args:
        self : type
            Description
        dataset_path : type
            Description
        target_column : type
            Description
        test_size : type
            Description
        split_col : type
            Description
        train_value : type
            Description
        test_value : type
            Description
        random_state : type
            Description
        normalize_features : type
            Description
        return_extra_info : type
            Description
        encode_categorical : type
            Description
        igtd_preprocessor : type
            Description
        igtd_configs : type
            Description
        igtd_result_base_dir : type
            Description

        Returns:
            type: Description
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
        self.igtd_preprocessor = igtd_preprocessor
        self.igtd_configs = igtd_configs
        self.igtd_result_base_dir = igtd_result_base_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def load_data(self):
        """load_data

        Args:
        self : type
            Description

        Returns:
            type: Description
        """
        df = pd.read_csv(self.dataset_path)
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset")
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        if self.encode_categorical:
            X = self.force_encode_categorical(X)
        if self.split_col and self.split_col in df.columns:
            if self.train_value is None or self.test_value is None:
                raise ValueError('When using split_col, you must specify train_value and test_value.')
            train_mask = df[self.split_col] == self.train_value
            test_mask = df[self.split_col] == self.test_value
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state, stratify=y)
        if self.normalize_features:
            X_train, X_test = self.scale_features(X_train, X_test, mode=self.normalize_features)
        igtd_results = None
        if self.igtd_preprocessor is not None and self.igtd_configs is not None:
            if not self.igtd_result_base_dir:
                raise ValueError('igtd_result_base_dir must be provided if using the IGTD preprocessor.')
            igtd_results = self.igtd_preprocessor.run(X_train, self.igtd_result_base_dir, self.igtd_configs)
            self.logger.info(f'IGTD results: {igtd_results}')
        extra_info = None
        if self.return_extra_info:
            igtd_path = None
            img_rows = None
            img_columns = None
            if igtd_results is not None:
                first_config = next(iter(igtd_results))
                igtd_path = os.path.join(igtd_results[first_config], 'ordering.txt')
                img_rows = self.igtd_preprocessor.img_rows
                img_columns = self.igtd_preprocessor.img_columns
            extra_info = self.create_extra_info(df, igtd_path=igtd_path, img_rows=img_rows, img_columns=img_columns)
        return X_train, X_test, y_train, y_test, extra_info

    def create_extra_info(self, df, igtd_path=None, img_rows=None, img_columns=None):
        """create_extra_info

        Args:
        self : type
            Description
        df : type
            Description
        igtd_path : type
            Description
        img_rows : type
            Description
        img_columns : type
            Description

        Returns:
            type: Description
        """
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        img_columns = df.select_dtypes(exclude=['object', 'category']).columns
        cat_unique_vals = [len(df[col].unique()) for col in cat_cols]
        column_info = {
            'cat_col_names': list(cat_cols),
            'cat_col_idx': list(df.columns.get_indexer(cat_cols)),
            'cat_col_unique_vals': cat_unique_vals,
            'num_col_names': list(img_columns),
            'num_col_idx': list(df.columns.get_indexer(img_columns)),
        }
        extra_info = column_info
        extra_info['num_features'] = len(df.columns)
        if igtd_path and os.path.exists(igtd_path):
            with open(igtd_path) as f:
                lines = f.readlines()
                if lines:
                    extra_info['column_ordering'] = list(map(int, lines[-1].strip().split()))
            extra_info['img_rows'] = img_rows
            extra_info['img_columns'] = img_columns
        return extra_info

    def force_encode_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        """force_encode_categorical

        Args:
        self : type
            Description
        X : type
            Description

        Returns:
            type: Description
        """
        return pd.get_dummies(X)

    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, mode: str):
        """scale_features

        Args:
        self : type
            Description
        X_train : type
            Description
        X_test : type
            Description
        mode : type
            Description

        Returns:
            type: Description
        """
        if mode == 'mean_std':
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns
            mean = X_train[numeric_cols].mean()
            std = X_train[numeric_cols].std().replace(0, 1)
            X_train[numeric_cols] = (X_train[numeric_cols] - mean) / std
            X_test[numeric_cols] = (X_test[numeric_cols] - mean) / std
        return X_train, X_test


if __name__ == '__main__':
    igtd_configs = {
        'Euclidean_Euclidean': {'fea_dist_method': 'Euclidean', 'image_dist_method': 'Euclidean', 'error': 'abs'},
        'Pearson_Manhattan': {'fea_dist_method': 'Pearson', 'image_dist_method': 'Manhattan', 'error': 'squared'},
    }
    igtd_preprocessor = IGTDPreprocessor(
        img_rows=8, img_columns=7, save_image_size=3, max_step=1000000, val_step=1000, min_gain=0.01, exclude_cols=[]
    )
    igtd_result_base_dir = './modelsdefinition/IGTD/results/ageconditions_igtd'
    dataset_path = './data/ageconditions.csv'
    data_loader = DynamicDataLoader(
        dataset_path=dataset_path,
        target_column='target',
        test_size=0.2,
        random_state=4200,
        normalize_features='mean_std',
        encode_categorical=True,
        return_extra_info=True,
        igtd_preprocessor=igtd_preprocessor,
        igtd_configs=igtd_configs,
        igtd_result_base_dir=igtd_result_base_dir,
    )
    X_train, X_test, y_train, y_test, extra_info = data_loader.load_data()
    print('Extra Info:', extra_info)
