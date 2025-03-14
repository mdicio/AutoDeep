import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import yaml

from autodeep.evaluation.generalevaluator import Evaluator
from autodeep.factory import create_dynamic_data_loader, create_model, seed_everything
from autodeep.outputhandler.outputwriter import OutputWriter

DEFAULT_MODELS = ['xgb', 'catboost', 'mlp', 'tabnet', 'resnet', 's1dcnn', 'categoryembedding', 'fttransformer', 'tabtransformer', 'gandalf', 'node']
DEFAULT_OUTPUT_FOLDER = Path('./output')
DEFAULT_MODEL_CONFIG_FILE = Path(__file__).parent / 'configuration' / 'model_config.yml'
DEFAULT_IGTD_CONFIG = {
    'img_size': 'auto',
    'save_image_size': 3,
    'max_step': 1000000,
    'val_step': 1000,
    'min_gain': 0.01,
    'ordering_methods': {
        'Euclidean_Euclidean': {'fea_dist_method': 'Euclidean', 'image_dist_method': 'Euclidean', 'error': 'abs'},
        'Pearson_Manhattan': {'fea_dist_method': 'Pearson', 'image_dist_method': 'Manhattan', 'error': 'squared'},
    },
}
DEFAULT_IGTD_DIR = Path('./IGTD')


class AutoRunner:

    def __init__(
        self,
        execution_mode='hyperopt',
        max_evals=50,
        random_state=42,
        default_models=DEFAULT_MODELS,
        model_config=DEFAULT_MODEL_CONFIG_FILE,
        data_config=None,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        output_filename='experiments',
        igtd_path=DEFAULT_IGTD_DIR,
        igtd_config=DEFAULT_IGTD_CONFIG,
    ):
        """__init__

        Args:
        self : type
            Description
        execution_mode : type
            Description
        max_evals : type
            Description
        random_state : type
            Description
        default_models : type
            Description
        model_config : type
            Description
        data_config : type
            Description
        output_folder : type
            Description
        output_filename : type
            Description
        igtd_path : type
            Description
        igtd_config : type
            Description

        Returns:
            type: Description
        """
        self.model_config = self._load_config(model_config)
        self.data_config = self._validate_data_config(data_config)
        self.output_folder = output_folder
        self.output_filename = output_filename + '.csv'
        self.default_models = default_models
        self.execution_mode = execution_mode
        self.max_evals = max_evals
        self.random_state = random_state
        self.igtd_path = igtd_path
        self.igtd_config = igtd_config
        self.results = []
        self._initialize()

    def _initialize(self):
        """_initialize

        Args:
        self : type
            Description

        Returns:
            type: Description
        """
        seed_everything(self.random_state)
        os.makedirs(self.output_folder, exist_ok=True)

    def _load_config(self, path):
        """_load_config

        Args:
        self : type
            Description
        path : type
            Description

        Returns:
            type: Description
        """
        with open(path, 'r') as file:
            return yaml.safe_load(file)

    def _validate_data_config(self, data_config):
        """_validate_data_config

        Args:
        self : type
            Description
        data_config : type
            Description

        Returns:
            type: Description
        """
        if not isinstance(data_config, dict):
            raise ValueError('data_config must be a dictionary.')
        return data_config

    def run(self):
        """run

        Args:
        self : type
            Description

        Returns:
            type: Description
        """
        for dataset_name, data_config in self.data_config.items():
            dataset_path = data_config.get('dataset_path')
            if not dataset_path or not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset path '{dataset_path}' not found.")
            for model_name in map(str.lower, self.default_models):
                print(f'Running {model_name} on {dataset_name}...')
                run_id = datetime.now().strftime('%Y%m%d-%H%M%S-') + str(uuid4())
                igtd_configs = data_config.get('igtd_configs', self.igtd_config)
                igtd_configs['img_size'] = data_config.get('igtd_configs', {}).get('img_size', 'auto')
                run_igtd = model_name == 'resnet'
                data_loader = create_dynamic_data_loader(
                    dataset_name=dataset_name,
                    dataset_path=dataset_path,
                    problem_type=data_config.get('problem_type'),
                    target_column=data_config.get('target_col'),
                    split_col=data_config.get('split_col'),
                    test_size=data_config.get('test_size'),
                    train_value=data_config.get('train_value'),
                    test_value=data_config.get('test_value'),
                    random_state=self.random_state,
                    normalize_features=self.model_config['model_configs'].get(model_name, {}).get('data_params', {}).get('normalize_features'),
                    return_extra_info=True,
                    encode_categorical=self.model_config['model_configs'].get(model_name, {}).get('data_params', {}).get('encode_categorical'),
                    run_igtd=run_igtd,
                    igtd_configs=igtd_configs,
                    igtd_result_base_dir=self.igtd_path,
                )
                X_train, X_test, y_train, y_test, extra_info = data_loader.load_data()
                model = create_model(model_name=model_name, random_state=self.random_state, problem_type=data_config['problem_type'])
                model.num_workers = 12
                (best_params, best_score, train_metrics, validation_metrics) = self._train_model(
                    model, X_train, y_train, model_name, data_config, extra_info
                )
                y_pred, y_prob = self._get_predictions(model, X_test, data_config)
                test_metrics = self._evaluate(y_test, y_pred, y_prob, data_config)
                self._save_results(run_id, dataset_name, model_name, train_metrics, validation_metrics, test_metrics, best_params, best_score)

    def _train_model(self, model, X_train, y_train, model_name, data_config, extra_info):
        """_train_model

        Args:
        self : type
            Description
        model : type
            Description
        X_train : type
            Description
        y_train : type
            Description
        model_name : type
            Description
        data_config : type
            Description
        extra_info : type
            Description

        Returns:
            type: Description
        """
        model_config = self.model_config['model_configs'].get(model_name, {})
        if self.execution_mode == 'hyperopt_kfold':
            raise ValueError('Not implemented yet, coming soon!')
        elif self.execution_mode == 'hyperopt':
            return model.hyperopt_search(
                X_train,
                y_train,
                model_config=model_config,
                metric=data_config['metric'],
                eval_metrics=data_config['eval_metrics'],
                max_evals=self.max_evals,
                extra_info=extra_info,
            )
        else:
            raise ValueError('Not implemented yet, coming soon!')

    def _get_predictions(self, model, X_test, data_config):
        """_get_predictions

        Args:
        self : type
            Description
        model : type
            Description
        X_test : type
            Description
        data_config : type
            Description

        Returns:
            type: Description
        """
        if data_config['problem_type'] == 'binary_classification':
            return model.predict(X_test, predict_proba=True)
        return model.predict(X_test), None

    def _evaluate(self, y_test, y_pred, y_prob, data_config):
        """_evaluate

        Args:
        self : type
            Description
        y_test : type
            Description
        y_pred : type
            Description
        y_prob : type
            Description
        data_config : type
            Description

        Returns:
            type: Description
        """
        evaluator = Evaluator(
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            run_metrics=data_config['eval_metrics'],
            metric=data_config['metric'],
            problem_type=data_config['problem_type'],
        )
        return evaluator.evaluate_model()

    def _save_results(self, run_id, dataset_name, model_name, train_metrics, validation_metrics, test_metrics, best_params, best_score):
        """_save_results

        Args:
        self : type
            Description
        run_id : type
            Description
        dataset_name : type
            Description
        model_name : type
            Description
        train_metrics : type
            Description
        validation_metrics : type
            Description
        test_metrics : type
            Description
        best_params : type
            Description
        best_score : type
            Description

        Returns:
            type: Description
        """
        output_path = os.path.join(self.output_folder, self.output_filename)
        fields = ['run_id', 'dataset', 'model', 'train_metrics', 'validation_metrics', 'test_metrics', 'best_params', 'best_score']
        output_writer = OutputWriter(output_path, fields)
        result_data = {
            'run_id': run_id,
            'dataset': dataset_name,
            'model': model_name,
            'train_metrics': train_metrics,
            'validation_metrics': validation_metrics,
            'test_metrics': test_metrics,
            'best_params': best_params,
            'best_score': best_score,
        }
        output_writer.write_row(**result_data)
