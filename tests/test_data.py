import numpy as np
import pandas as pd
import pytest

from autodeep.dataloaders.dataloader import DynamicDataLoader
from autodeep.datasets import DatasetSpec, infer_problem_type, parse_datasets, profile_dataframe, read_table, sanitize_columns
from autodeep.modelutils.trainingutilities import infer_hyperopt_space, infer_hyperopt_space_pytorch_tabular


def test_infer_problem_type():
    assert infer_problem_type(pd.Series(["a", "b", "a"])) == "binary_classification"
    assert infer_problem_type(pd.Series([0, 1, 2, 1, 0] * 20)) == "multiclass_classification"
    assert infer_problem_type(pd.Series(np.linspace(0, 1, 100))) == "regression"
    with pytest.raises(ValueError):
        infer_problem_type(pd.Series([1, 1, 1]))


def test_sanitize_columns_unique_and_safe():
    m = sanitize_columns(["a b", "a_b", "x[<1]", "1st", ""])
    assert len(set(m.values())) == 5
    assert all("[" not in v and "<" not in v and v[0].isalpha() for v in m.values())


@pytest.mark.parametrize("name", ["iris", "wine", "breast_cancer", "diabetes", "digits", "synthetic_mixed"])
def test_builtin_datasets_resolve(name):
    spec = DatasetSpec.builtin(name).resolve()
    assert spec.target and spec.problem_type and spec.metric in spec.eval_metrics


def test_legacy_keys_and_unknown_option(mixed_csv):
    spec = parse_datasets({"m": {"dataset_path": str(mixed_csv), "target_col": "label", "eval_metrics": ["f1_score"]}})[0]
    spec.resolve()
    assert spec.target == "label" and spec.problem_type == "binary_classification"
    assert "f1" in spec.eval_metrics and spec.metric == "roc_auc"
    with pytest.raises(ValueError, match="unknown option"):
        DatasetSpec.from_dict("m", {"path": str(mixed_csv), "taget": "label"})


def test_missing_target_gives_helpful_error(mixed_csv):
    with pytest.raises(ValueError, match="target column 'nope' not found"):
        DatasetSpec.from_dict("m", {"path": str(mixed_csv), "target": "nope"}).resolve()


def test_loader_encodes_imputes_and_excludes_split_col(mixed_csv):
    spec = DatasetSpec.from_dict(
        "m", {"path": str(mixed_csv), "target": "label", "split_col": "split", "train_value": "train", "test_value": "test", "drop_columns": ["id"]}
    ).resolve()
    X_tr, X_te, y_tr, y_te, info = DynamicDataLoader.from_spec(spec, normalize_features="mean_std").load_data()
    assert len(X_tr) == 240 and len(X_te) == 60
    assert "split" not in X_tr.columns and "id" not in X_tr.columns
    assert "age_years" in X_tr.columns  # sanitized name
    assert not X_tr.isna().any().any() and not X_te.isna().any().any()
    assert set(np.unique(y_tr)) == {0, 1} and info["classes"] == ["no", "yes"]
    assert set(info["cat_col_names"]) == {"city", "member"}


def test_one_hot_is_aligned_and_capped():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"c": [f"v{i}" for i in rng.integers(0, 80, 400)], "x": rng.normal(size=400), "y": rng.integers(0, 2, 400)})
    spec = DatasetSpec.from_dict("d", {"path": "unused", "target": "y"}).resolve(df)
    X_tr, X_te, *_ = DynamicDataLoader.from_spec(spec, dataframe=df, encode_categorical=True).load_data()
    assert list(X_tr.columns) == list(X_te.columns)
    assert X_tr.shape[1] <= 1 + 31


def test_read_table_sniffs_separator(tmp_path):
    p = tmp_path / "semi.csv"
    p.write_text("a;b;y\n1;2;0\n3;4;1\n")
    assert list(read_table(p).columns) == ["a", "b", "y"]


def test_profile_flags_id_columns():
    df = read_table("builtin:iris")
    df.insert(0, "row_id", range(len(df)))
    prof = profile_dataframe(df, target="species")
    hints = {c["name"]: c["hints"] for c in prof["columns"]}
    assert hints["row_id"] and prof["inferred_problem_type"] == "multiclass_classification"


def test_search_spaces():
    from hyperopt.pyll.stochastic import sample

    space = infer_hyperopt_space_pytorch_tabular(
        {
            "batch_size": [16, 512],
            "use_batch_norm": [True, False],
            "layers": ["64-32", "128-64"],
            "dropout": [0.0, 0.3],
            "optimizer_fn": {"Adam": {"learning_rate": [0.01, 0.0001]}, "SGD": {"learning_rate": [0.01, 0.0001], "momentum": [0.9, 0.99]}},
        }
    )
    rng = np.random.default_rng(0)
    for _ in range(200):
        s = sample(space, rng=rng)
        assert s["batch_size"] >= 16 and s["batch_size"] % 16 == 0
        assert s["use_batch_norm"] in (True, False)
        assert 0.0 <= s["dropout"] <= 0.3
    plain = infer_hyperopt_space({"max_depth": [3, 8], "learning_rate": [0.3, 0.01], "hidden_layer_sizes": [[64], [64, 32]]})
    s = sample(plain, rng=rng)
    assert 3 <= s["max_depth"] <= 8 and isinstance(s["hidden_layer_sizes"], (list, tuple))
