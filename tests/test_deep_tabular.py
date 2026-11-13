"""The nine embedding-based deep tabular models ported from pytorch-tabular (no pytorch-tabular dependency)."""

import pickle

import numpy as np
import pandas as pd
import pytest

from autodeep.modelsdefinition.tabular_nets.architectures import BUILDERS, build_network

ARCHS = sorted(BUILDERS)
# small settings so the forward pass is cheap; everything else is pytorch-tabular's default
SMALL = {
    "node": {"num_trees": 16, "depth": 3},
    "fttransformer": {"num_attn_blocks": 1, "input_embed_dim": 8, "num_heads": 2},
    "tabtransformer": {"num_attn_blocks": 1, "input_embed_dim": 8, "num_heads": 2},
    "gate": {"num_trees": 2, "tree_depth": 2, "gflu_stages": 1},
    "danet": {"n_layers": 2, "virtual_batch_size": 16},
    "tabnet": {"virtual_batch_size": 16},
}


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("task,output_dim", [("classification", 1), ("classification", 3), ("regression", 1)])
@pytest.mark.parametrize("n_cat", [0, 2])
def test_every_architecture_forward_and_backward(arch, task, output_dim, n_cat):
    import torch

    torch.manual_seed(0)
    cards = [4, 7][:n_cat]
    net = build_network(
        arch, {"batch_size": 32, **SMALL.get(arch, {})}, continuous_dim=5, categorical_cardinality=cards, output_dim=output_dim, task=task
    )
    x_cont = torch.randn(32, 5)
    x_cat = torch.stack([torch.randint(0, c, (32,)) for c in cards], dim=1) if cards else torch.empty(32, 0, dtype=torch.long)
    net.data_aware_init(x_cont, x_cat, torch.randn(32))
    out = net(x_cont, x_cat)
    assert out.shape == (32, output_dim)
    out.sum().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


def _frame(n, rng):
    return pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "city": rng.choice(["rome", "milan", "turin"], n),
        }
    )


@pytest.mark.parametrize("model", ["categoryembedding", "gandalf", "tabnet"])
def test_trainer_round_trip_and_unseen_categories(model):
    from autodeep.factory import create_model

    rng = np.random.default_rng(0)
    X = _frame(200, rng)
    y = pd.Series(((X["a"] + (X["city"] == "rome")) > 0.5).astype(int))
    trainer = create_model(model, "binary_classification", 0)
    cfg = {"default_params": {"max_epochs": 2, "batch_size": 64}, "param_grid": {"batch_size": [64]}}
    trainer.hyperopt_search(X, y, cfg, "roc_auc", ["roc_auc"], max_evals=1, extra_info={"num_classes": 2, "cat_col_names": ["city"]})
    assert trainer.categorical_cols == ["city"] and trainer.cat_cardinalities == [4]

    X_new = _frame(10, rng)
    X_new.loc[0, "city"], X_new.loc[1, "city"] = "naples", None  # unseen and missing map to code 0
    codes = trainer.to_tensor(X_new)[:, -1]
    assert codes[0] == 0 and codes[1] == 0 and (codes[2:] > 0).all()
    pred, prob = trainer.predict(X_new, predict_proba=True)
    assert pred.shape == (10,) and np.isfinite(prob).all()

    clone = pickle.loads(pickle.dumps(trainer))
    np.testing.assert_allclose(clone.predict(X_new, predict_proba=True)[1], prob, rtol=1e-5, atol=1e-6)


def test_tabtransformer_out_ff_options_configure_the_head():
    from autodeep.modelsdefinition.deep_tabular_models import TabTransformerTrainer

    params = TabTransformerTrainer().adjust_params({"out_ff_layers": "16-8", "out_ff_dropout": 0.1, "input_embed_dim_multiplier": 2, "num_heads": 4})
    assert params == {"head_config": {"layers": "16-8", "dropout": 0.1}, "input_embed_dim": 8, "num_heads": 4}
    net = build_network("tabtransformer", params, continuous_dim=3, categorical_cardinality=[3], output_dim=1, task="classification")
    linears = [m for m in net.head.modules() if m.__class__.__name__ == "Linear"]
    assert [m.out_features for m in linears] == [16, 8, 1]
