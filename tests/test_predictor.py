"""Saved models: preprocessing replay, save/load round trip, predict CLI."""

import numpy as np
import pandas as pd
import pytest

from autodeep import AutoRunner, load_model
from autodeep.cli import main
from autodeep.dataloaders.dataloader import DynamicDataLoader
from autodeep.datasets import parse_datasets, read_table
from autodeep.preprocessing import TabularPreprocessor


@pytest.mark.parametrize("normalize,encode", [("mean_std", False), ("min_max", True), (False, True)])
def test_preprocessor_replays_training_transform(mixed_csv, normalize, encode):
    spec = parse_datasets({"m": {"path": str(mixed_csv), "target": "label", "drop_columns": ["id", "split"]}})[0]
    df = spec.read()
    spec.resolve(df)
    loader = DynamicDataLoader.from_spec(spec, dataframe=df, normalize_features=normalize, encode_categorical=encode)
    X_train, X_test, *_ = loader.load_data()
    raw = df.drop(columns=["label"]).loc[X_test.index]  # extra columns (id, split) are ignored
    pd.testing.assert_frame_equal(loader.preprocessor.transform_raw(raw), X_test, check_dtype=False)


def test_preprocessor_handles_unseen_values_and_types():
    train = pd.DataFrame({"a b": [1.0, 2.0, None, 4.0], "c": ["x", "y", "x", None], "flag": [True, False, True, True]})
    prep = TabularPreprocessor(normalize="mean_std", encode_categorical=True)
    prep.fit(prep.fit_schema(train))
    new = pd.DataFrame({"a b": ["3", "oops"], "c": ["z", None], "flag": [False, True], "extra": [0, 0]})  # numbers as text, unseen category
    out = prep.transform_raw(new)
    assert list(out.columns) == prep.output_columns and not out.isna().any().any()
    with pytest.raises(ValueError, match="Missing column"):
        prep.transform_raw(new.drop(columns=["c"]))
    assert [c["type"] for c in prep.schema()] == ["numeric", "categorical", "categorical"]


def test_best_model_save_load_predict(tmp_path, mixed_csv):
    runner = AutoRunner(
        datasets={
            "mixed": {"path": str(mixed_csv), "target": "label", "drop_columns": ["id", "split"]},
            "iris": {"path": "builtin:iris"},
            "diabetes": {"path": "builtin:diabetes"},
        },
        models=["xgb", "mlp"],
        max_evals=1,
        output_folder=tmp_path / "out",
        log_level="WARNING",
    )
    board = runner.run()
    assert board.groupby("dataset")["selected"].sum().eq(1).all()
    raw = pd.read_csv(mixed_csv).head(25)
    for name, data in [("mixed", raw), ("iris", read_table("builtin:iris").head(25)), ("diabetes", read_table("builtin:diabetes").head(25))]:
        mem = runner.best_model(name)
        path = tmp_path / "out" / "models" / name / "best"
        assert (path / "info.json").exists() and runner.best_model_paths[name] == path
        disk = load_model(path, device="cpu")
        assert disk.model_name == mem.model_name
        np.testing.assert_array_equal(np.asarray(disk.predict(data), dtype=str), np.asarray(mem.predict(data), dtype=str))
        if disk.problem_type != "regression":
            proba = disk.predict_proba(data)
            assert list(proba.columns) == disk.classes and np.allclose(proba.sum(axis=1), 1.0)
            assert set(disk.predict(data)) <= set(disk.classes)
        scores = disk.evaluate(data)
        assert disk.info["metric"] in scores
    assert set(runner.best_model("mixed").predict(raw)) <= {"yes", "no"}  # original labels, not 0/1


def test_save_models_and_predict_cli(tmp_path, mixed_csv, capsys):
    runner = AutoRunner(
        datasets={"m": {"path": str(mixed_csv), "target": "label", "drop_columns": ["id", "split"]}},
        models=["xgb", "catboost"],
        max_evals=1,
        output_folder=tmp_path / "out",
        save_models=True,
        log_level="WARNING",
    )
    runner.run()
    models = tmp_path / "out" / "models" / "m"
    assert {p.name for p in models.iterdir()} == {"xgb", "catboost", "best"}
    rows = pd.read_csv(mixed_csv).drop(columns=["label"]).head(10)
    rows.loc[0, "city"] = "paris"  # unseen category
    rows.to_csv(tmp_path / "new.csv", index=False)
    out = tmp_path / "pred.csv"
    assert main(["predict", str(models / "best"), str(tmp_path / "new.csv"), "-o", str(out)]) == 0
    pred = pd.read_csv(out)
    assert len(pred) == 10 and {"prediction", "proba_no", "proba_yes", "city"} <= set(pred.columns)
    assert main(["info", str(models / "catboost")]) == 0
    assert "Input columns" in capsys.readouterr().out
    assert main(["predict", str(models / "best"), "builtin:iris"]) == 2  # wrong columns -> clean error


def test_no_saving_when_disabled(tmp_path):
    runner = AutoRunner(datasets={"iris": {"path": "builtin:iris"}}, models=["xgb"], max_evals=1, output_folder=tmp_path / "o", save_best=False)
    runner.run()
    assert not (tmp_path / "o" / "models").exists()
