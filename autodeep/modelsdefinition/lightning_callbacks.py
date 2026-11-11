"""Lightning callbacks (module level so that saved PyTorch Tabular models can be pickled)."""

import pytorch_lightning as pl


class BestWeights(pl.Callback):
    """Keep an in-memory copy of the weights with the lowest validation loss.

    Replaces PyTorch Tabular's checkpoint reload, which fails with ``torch.load(weights_only=True)``
    (the default since PyTorch 2.6) because checkpoints contain omegaconf objects.
    """

    def __init__(self):
        self.best = float("inf")
        self.state = None

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        value = trainer.callback_metrics.get("valid_loss")
        if value is not None and float(value) < self.best:
            self.best = float(value)
            self.state = {k: t.detach().clone() for k, t in pl_module.state_dict().items()}

    def state_dict(self):  # nothing worth persisting with a saved model
        return {}
