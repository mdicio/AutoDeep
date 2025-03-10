import inspect

import pandas as pd
from pytorch_tabular import TabularModel
from pytorch_tabular.config import OptimizerConfig
from pytorch_tabular.models import GANDALFConfig
from sklearn.preprocessing import StandardScaler

from autodeep.modelsdefinition.CommonStructure import PytorchTabularTrainer


class GandalfTrainer(PytorchTabularTrainer):

    def __init__(
        self,
        problem_type,
    ):
        super().__init__(problem_type)
        self.logger.info("Trainer initialized")
        self.model_name = "gandalf"

    def prepare_tabular_model(self, params, default_params, default=False):
        print("tabular model params")
        print(params)
        print("tabular model outer params")
        print(default_params)

        data_config, trainer_config, optimizer_config, learning_rate = self.prepare_shared_tabular_configs(
            params=params,
            default_params=default_params,
            extra_info=self.extra_info,
        )

        valid_params = inspect.signature(GANDALFConfig).parameters
        compatible_params = {param: value for param, value in params.items() if param in valid_params}
        invalid_params = {param: value for param, value in params.items() if param not in valid_params}
        self.logger.warning(f"You are passing some invalid parameters to the model {invalid_params}")

        if self.task == "regression":
            compatible_params["target_range"] = self.target_range

        self.logger.debug(f"valid parameters: {compatible_params}")
        model_config = GANDALFConfig(task=self.task, learning_rate=learning_rate, **compatible_params)

        if default:
            model_config = GANDALFConfig(task=self.task)
            optimizer_config = OptimizerConfig()

        tabular_model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
        )
        return tabular_model

    def scale_regression_target(self, y):
        y_array = y.to_numpy().reshape(-1, 1)
        self.target_scaler = StandardScaler()
        y_scaled = self.target_scaler.fit_transform(y_array)
        y_scaled_series = pd.Series(y_scaled.flatten(), name="target")
        return y_scaled_series
