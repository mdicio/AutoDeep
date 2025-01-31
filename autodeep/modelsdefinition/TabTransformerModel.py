import inspect
import logging

from pytorch_tabular import TabularModel
from pytorch_tabular.config import OptimizerConfig
from pytorch_tabular.models import TabTransformerConfig

from autodeep.modelsdefinition.CommonStructure import PytorchTabularTrainer
from autodeep.modelutils.trainingutilities import prepare_shared_tabular_configs


class TabTransformerTrainer(PytorchTabularTrainer):

    def __init__(self, problem_type, num_classes=None):
        super().__init__(problem_type, num_classes)
        self.logger.info("Trainer initialized")

    def prepare_tabular_model(self, params, outer_params, default=False):
        print("tabular model params")
        print(params)
        print("tabular model outer params")
        print(outer_params)

        data_config, trainer_config, optimizer_config, learning_rate = (
            prepare_shared_tabular_configs(
                params=params,
                outer_params=outer_params,
                extra_info=self.extra_info,
                save_path=self.save_path,
                task=self.task,
            )
        )

        # input_dim' (input_embed_dim) must be multiples of 'num_heads'
        input_embed_dim_multiplier = params.get("input_embed_dim_multiplier", None)
        num_heads = params.get("num_heads", None)

        if num_heads is not None and input_embed_dim_multiplier is not None:
            params["input_embed_dim"] = input_embed_dim_multiplier * num_heads

        valid_params = inspect.signature(TabTransformerConfig).parameters
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

        model_config = TabTransformerConfig(
            task=self.task,
            learning_rate=learning_rate,
            **compatible_params,
        )

        if default:
            model_config = TabTransformerConfig(task=self.task)
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
