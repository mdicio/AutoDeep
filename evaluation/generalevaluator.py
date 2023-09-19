import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    multilabel_confusion_matrix,
)


class Evaluator:
    def __init__(
        self,
        y_true=None,
        y_pred=None,
        y_prob=None,
        problem_type=None,
        run_metrics=None,
        metric=None,
    ):
        self.y_true = y_true
        self.y_pred = y_pred
        self.run_metrics = run_metrics
        self.metric = metric
        self.problem_type = problem_type
        self.y_prob = y_prob

        self.maximize = {
            "f1": (True, 1.0),
            "area_under_pr": (True, 1.0),
            "accuracy": (True, 1.0),
            "roc_auc": (True, 1.0),
            "precision": (True, 1.0),
            "recall": (True, 1.0),
            "mse": (False, 0.0),
            "rmse": (False, 0.0),
            "r2_score": (True, 1.0),
            "lift": (True, 100000),
        }

    def recall(self):
        if self.problem_type == "binary_classification":
            average = "binary"
        elif self.problem_type == "multiclass_classification":
            average = "weighted"
        else:
            raise ValueError(f"Invalid problem type: {self.problem_type}")
        return recall_score(self.y_true, self.y_pred, average=average, zero_division=0)

    def precision(self):
        if self.problem_type == "binary_classification":
            average = "binary"
        elif self.problem_type == "multiclass_classification":
            average = "weighted"
        else:
            raise ValueError(f"Invalid problem type: {self.problem_type}")
        return precision_score(
            self.y_true, self.y_pred, average=average, zero_division=0
        )

    def mse(self):
        return mean_squared_error(self.y_true, self.y_pred)

    def mae(self):
        return mean_absolute_error(self.y_true, self.y_pred)

    def rmse(self):
        return mean_squared_error(self.y_true, self.y_pred, squared=False)

    def r2_score(self):
        return r2_score(self.y_true, self.y_pred)

    def accuracy(self):
        if self.problem_type == "binary_classification":
            return accuracy_score(self.y_true, self.y_pred)
        elif self.problem_type == "multiclass_classification":
            if self.y_pred.ndim == 1:
                n_classes = len(np.unique(self.y_true))
                y_pred_one_hot = np.eye(n_classes)[self.y_pred.astype(int)]
            else:
                y_pred_one_hot = self.y_pred.astype(int)
            return accuracy_score(self.y_true, y_pred_one_hot.argmax(axis=1))
        else:
            raise ValueError(f"Invalid problem type: {self.problem_type}")

    def f1(self):
        if self.problem_type == "binary_classification":
            average = "binary"
        elif self.problem_type == "multiclass_classification":
            average = "weighted"
        else:
            raise ValueError(f"Invalid problem type: {self.problem_type}")
        return f1_score(self.y_true, self.y_pred, average=average, zero_division=0)

    def auc(self):
        if self.problem_type == "binary_classification":
            return roc_auc_score(self.y_true, self.y_prob)
        else:
            raise ValueError(f"Invalid problem type: {self.problem_type}")

    def lift(self, percentile=0.1):
        n = len(self.y_true)
        p_true = np.sum(self.y_true) / n

        top_indices = np.argsort(self.y_prob)[::-1][: int(percentile * n)]
        p_top_true = np.sum(np.take(self.y_true, top_indices, axis=0)) / len(
            top_indices
        )

        return p_top_true / p_true

    def area_under_pr(self):
        # return average_precision_score(self.y_true, self.y_pred)
        return average_precision_score(self.y_true, self.y_prob)

    def confusion_matrix(self):
        if self.problem_type == "binary_classification":
            return multilabel_confusion_matrix(self.y_true, self.y_pred.round())
        elif self.problem_type == "multiclass_classification":
            return multilabel_confusion_matrix(self.y_true, self.y_pred.argmax(axis=1))
        else:
            raise ValueError(f"Invalid problem type: {self.problem_type}")

    def evaluate_model(self):
        if self.problem_type not in [
            "binary_classification",
            "multiclass_classification",
            "regression",
        ]:
            raise ValueError(f"Invalid problem type: {self.problem_type}")

        results = {}

        if "recall" in self.run_metrics:
            results["recall"] = self.recall()

        if "precision" in self.run_metrics:
            results["precision"] = self.precision()

        if "mse" in self.run_metrics:
            results["mse"] = self.mse()

        if "mae" in self.run_metrics:
            results["mae"] = self.mae()

        if "rmse" in self.run_metrics:
            results["rmse"] = self.rmse()

        if "r2_score" in self.run_metrics:
            results["r2_score"] = self.r2_score()

        if "accuracy" in self.run_metrics:
            results["accuracy"] = self.accuracy()

        if "f1" in self.run_metrics:
            results["f1"] = self.f1()

        if "roc_auc" in self.run_metrics:
            results["roc_auc"] = self.auc()

        if "area_under_pr" in self.run_metrics:
            results["area_under_pr"] = self.area_under_pr()

        if "lift" in self.run_metrics:
            results["lift"] = self.lift()
        return results

    def evaluate_metric(self, metric_name):
        if self.problem_type not in [
            "binary_classification",
            "multiclass_classification",
            "regression",
        ]:
            raise ValueError(f"Invalid problem type: {self.problem_type}")

        if metric_name == "recall":
            return self.recall()

        if metric_name == "precision":
            return self.precision()

        if metric_name == "mse":
            return self.mse()

        if metric_name == "r2_score":
            return self.r2_score()

        if metric_name == "accuracy":
            return self.accuracy()

        if metric_name == "f1":
            return self.f1()

        if metric_name == "roc_auc":
            return self.auc()

        if metric_name == "lift":
            return self.lift()

        if metric_name == "area_under_pr":
            return self.area_under_pr()

        if metric_name == "lift":
            return self.lift()
        else:
            raise ValueError(f"Invalid metric name: {metric_name}")
