from __future__ import annotations

from autodeep.modelsdefinition.CommonStructure import SklearnLikeTrainer


class MLP(SklearnLikeTrainer):
    """scikit-learn multi-layer perceptron (CPU)."""

    model_name = "mlp"

    def fit_trial(self, params, X_train, y_train, X_val, y_val):
        from sklearn.neural_network import MLPClassifier, MLPRegressor

        params = dict(params)
        if "hidden_layer_sizes" in params:
            params["hidden_layer_sizes"] = tuple(int(v) for v in params["hidden_layer_sizes"])
        if "batch_size" in params:
            params["batch_size"] = int(min(params["batch_size"], len(X_train)))
        n_iter_no_change = int(params.pop("n_iter_no_change", self.default_params.get("n_iter_no_change", 10)))
        max_iter = int(params.pop("max_iter", self.default_params.get("max_iter", 500)))
        if self.max_epochs:
            max_iter = min(max_iter, int(self.max_epochs))
        cls = MLPRegressor if self.problem_type == "regression" else MLPClassifier
        # sklearn's MLP does its own internal early-stopping split of the training data
        model = cls(
            early_stopping=True,
            n_iter_no_change=n_iter_no_change,
            max_iter=max_iter,
            random_state=self.random_state,
            **params,
        )
        model.fit(X_train.to_numpy(dtype=float), y_train.to_numpy())
        return model

    def predict_with(self, model, X):
        return super().predict_with(model, X.to_numpy(dtype=float))


MLPTrainer = MLP
