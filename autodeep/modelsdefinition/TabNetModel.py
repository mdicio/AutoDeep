import inspect

from pytorch_tabular import TabularModel
from pytorch_tabular.config import OptimizerConfig
from pytorch_tabular.models import TabNetModelConfig

from autodeep.modelsdefinition.CommonStructure import PytorchTabularTrainer


class TabNetTrainer(PytorchTabularTrainer):

    def __init__(self, problem_type, num_targets=None):
        super().__init__(problem_type, num_targets)
        self.logger.info("Trainer initialized")
        self.model_name = "tabnet"
        

    def prepare_tabular_model(self, params, outer_params, default=False):
        print("tabular model params")
        print(params)
        print("tabular model outer params")
        print(outer_params)

        data_config, trainer_config, optimizer_config, learning_rate = (
            self.prepare_shared_tabular_configs(
                params=params,
                outer_params=outer_params,
                extra_info=self.extra_info,
            )
        )

        valid_params = inspect.signature(TabNetModelConfig).parameters
        compatible_params = {
            param: value for param, value in params.items() if param in valid_params
        }
        invalid_params = {
            param: value for param, value in params.items() if param not in valid_params
        }
        self.logger.warning(
            f"You are passing some invalid parameters to the model {invalid_params}"
        )

        if self.task == "regression":
            compatible_params["target_range"] = self.target_range

        self.logger.debug(f"compatible parameters: {compatible_params}")

        model_config = TabNetModelConfig(
            task=self.task,
            learning_rate=learning_rate,
            **compatible_params,
        )

        if default:
            model_config = TabNetModelConfig(task=self.task)
            optimizer_config = OptimizerConfig()

        print(data_config)
        print(model_config)
        print(optimizer_config)
        print(trainer_config)

        tabular_model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
        )
        return tabular_model
