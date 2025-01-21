import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import yaml

class DynamicDataLoader:
    def __init__(self, dataset_path, target_column=None, problem_type=None, test_size=0.2, config=None, **kwargs):
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.problem_type = problem_type
        self.test_size = test_size
        self.config = config or {}
        self.data = None

    def load_and_preprocess_data(self):
        self.data = pd.read_csv(self.dataset_path)
        if self.target_column is None:
            raise ValueError("Target column is required")
        # Infer problem type if not provided
        if self.problem_type is None:
            self.problem_type = self.infer_problem_type()

        # Split the dataset into training and testing
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, stratify=y)

        # Normalize features if required
        normalize_method = self.config.get('normalize_features', 'mean_std')
        if normalize_method == 'mean_std':
            X_train, X_test = self.scale_features(X_train, X_test)
        
        # Return the processed data
        return X_train, X_test, y_train, y_test

    def infer_problem_type(self):
        target_unique_vals = self.data[self.target_column].nunique()
        if target_unique_vals == 2:
            return 'binary_classification'
        elif target_unique_vals > 2:
            return 'multiclass_classification'
        else:
            return 'regression'

    def scale_features(self, X_train, X_test, mode="mean_std"):
        num_cols = X_train.select_dtypes(exclude=["object", "category"]).columns
        if mode == "mean_std":
            scaler = StandardScaler()
            X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
            X_test[num_cols] = scaler.transform(X_test[num_cols])
        elif mode == "min_max":
            scaler = MinMaxScaler(feature_range=(0, 1))
            X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
            X_test[num_cols] = scaler.transform(X_test[num_cols])
        return X_train, X_test

    def generate_config(self):
        config = {
            'problem_type': self.problem_type,
            'target_column': self.target_column,
            'test_size': self.test_size,
            'normalize_features': self.config.get('normalize_features', 'mean_std'),
            'eval_metrics': self.config.get('eval_metrics', ['accuracy']),
            'model_params': self.config.get('model_params', {})
        }
        # Generate a YAML configuration file based on this
        config_file_path = os.path.join(self.dataset_path, 'config.yml')
        with open(config_file_path, 'w') as file:
            yaml.dump(config, file)

        return config

# Example usage:
dataset_path = "data/titanic.csv"
target_column = "Survived"
config = {
    'normalize_features': 'mean_std',
    'eval_metrics': ['accuracy', 'f1', 'roc_auc']
}

data_loader = DynamicDataLoader(dataset_path=dataset_path, target_column=target_column, config=config)
X_train, X_test, y_train, y_test = data_loader.load_and_preprocess_data()
generated_config = data_loader.generate_config()

print(f"Generated config: {generated_config}")
