"""Defines training-related classes."""
from pytorch_lightning.callbacks import EarlyStopping


class EarlyStoppingWithMinEpochs(EarlyStopping):
    """Defines model used for early stopping."""

    def __init__(self, min_epochs, *args, **kwargs):
        """Build EarlyStoppingWithMinEpochs."""
        super().__init__(*args, **kwargs)
        self.min_epochs = min_epochs

    def on_validation_end(self, trainer, pl_module):
        """Check if early stopping should be performed."""
        if trainer.current_epoch < self.min_epochs:
            return  # skip early stopping
        super().on_validation_end(trainer, pl_module)
