import inspect

from pytorch_tabular import TabularModel
from pytorch_tabular.config import OptimizerConfig
from pytorch_tabular.models import FTTransformerConfig

from autodeep.modelsdefinition.CommonStructure import PytorchTabularTrainer


class FTTransformerTrainer(PytorchTabularTrainer):

    def __init__(self, problem_type):
        """__init__

        Args:
        self : type
            Description
        problem_type : type
            Description

        Returns:
            type: Description
        """
        super().__init__(problem_type)
        self.logger.info("Trainer initialized")
        self.model_name = "fttransformer"

    def prepare_tabular_model(self, params, default_params, default=False):
        """prepare_tabular_model

        Args:
        self : type
            Description
        params : type
            Description
        default_params : type
            Description
        default : type
            Description

        Returns:
            type: Description
        """
        print("tabular model params")
        print(params)
        print("tabular model outer params")
        print(default_params)
        data_config, trainer_config, optimizer_config, learning_rate = (
            self.prepare_shared_tabular_configs(
                params=params, default_params=default_params, extra_info=self.extra_info
            )
        )
        input_embed_dim_multiplier = params.get("input_embed_dim_multiplier", None)
        num_heads = params.get("num_heads", None)
        if num_heads is not None and input_embed_dim_multiplier is not None:
            params["input_embed_dim"] = input_embed_dim_multiplier * num_heads
        valid_params = inspect.signature(FTTransformerConfig).parameters
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
        model_config = FTTransformerConfig(
            task=self.task, learning_rate=learning_rate, **compatible_params
        )
        if default:
            model_config = FTTransformerConfig(task=self.task)
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
