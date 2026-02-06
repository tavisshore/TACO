"""Callbacks for CVGL training."""

from typing import TYPE_CHECKING

import lightning as L

if TYPE_CHECKING:
    from taco.sensors.cvgl.cvusa import ProgressiveAugmentation


class ProgressiveAugmentationCallback(L.Callback):
    """
    Lightning callback to update progressive augmentation strength during training.

    This callback automatically increases augmentation strength based on the current epoch.
    It searches for ProgressiveAugmentation instances in the training dataset and updates
    their strength at the start of each epoch.

    Example:
        >>> from taco.sensors.cvgl.callbacks import ProgressiveAugmentationCallback
        >>> callback = ProgressiveAugmentationCallback()
        >>> trainer = L.Trainer(callbacks=[callback])
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize the callback.

        Args:
            verbose: If True, print augmentation strength updates
        """
        super().__init__()
        self.verbose = verbose
        self._progressive_augs = []

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Find and store progressive augmentation instances."""
        # Access the training dataset
        if hasattr(trainer, "datamodule") and hasattr(trainer.datamodule, "train_dataset"):
            dataset = trainer.datamodule.train_dataset
        elif hasattr(trainer, "train_dataloader"):
            dataloader = trainer.train_dataloader()
            dataset = dataloader.dataset if hasattr(dataloader, "dataset") else None
        else:
            dataset = None

        if dataset is None:
            return

        # Find ProgressiveAugmentation instances
        from taco.sensors.cvgl.cvusa import ProgressiveAugmentation

        self._progressive_augs = []

        if hasattr(dataset, "augmentations") and isinstance(
            dataset.augmentations, ProgressiveAugmentation
        ):
            self._progressive_augs.append(("augmentations", dataset.augmentations))

        if hasattr(dataset, "augmentations_sync") and isinstance(
            dataset.augmentations_sync, ProgressiveAugmentation
        ):
            self._progressive_augs.append(("augmentations_sync", dataset.augmentations_sync))

        if self.verbose and self._progressive_augs:
            print(f"\nFound {len(self._progressive_augs)} progressive augmentation(s)")
            for name, aug in self._progressive_augs:
                print(
                    f"  - {name}: warmup={aug.warmup_epochs} epochs, "
                    f"strength={aug.start_strength:.2f} -> {aug.end_strength:.2f}"
                )

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Update augmentation strength at the start of each epoch."""
        current_epoch = trainer.current_epoch

        for name, aug in self._progressive_augs:
            strength = aug.update_strength(current_epoch)

            if self.verbose:
                status = "warming up" if current_epoch < aug.warmup_epochs else "steady"
                print(f"Epoch {current_epoch}: {name} strength = {strength:.3f} ({status})")
