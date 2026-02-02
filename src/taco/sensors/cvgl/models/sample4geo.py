import lightning.pytorch as pl
import numpy as np
import timm
import torch
import torch.nn.functional as F


class TimmModel(pl.LightningModule):
    def __init__(self, model_name, pretrained=True, img_size=383, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.img_size = img_size
        self.lr = lr

        if "vit" in model_name:
            self.model = timm.create_model(
                model_name, pretrained=pretrained, num_classes=0, img_size=img_size
            )
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def get_config(self):
        return timm.data.resolve_model_data_config(self.model)

    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

    def forward(self, img1, img2=None):
        if img2 is not None:
            image_features1 = self.model(img1)
            image_features2 = self.model(img2)
            return image_features1, image_features2
        else:
            return self.model(img1)

    def training_step(self, batch, batch_idx):
        # Adjust based on your batch structure
        img1, img2, labels = batch  # or however your data is structured
        feat1, feat2 = self(img1, img2)

        # Add your loss computation here
        loss = self.compute_loss(feat1, feat2, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img1, img2, labels = batch
        feat1, feat2 = self(img1, img2)
        loss = self.compute_loss(feat1, feat2, labels)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def compute_loss(self, feat1, feat2, labels):
        # Placeholder - implement your actual loss
        raise NotImplementedError("Implement your loss function")


class Sample4GeoEncoder(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        img_size: int = 384,
        freeze: bool = False,
    ):
        super().__init__()

        # Create the underlying timm model
        if "vit" in model_name:
            self.model = timm.create_model(
                model_name, pretrained=pretrained, num_classes=0, img_size=img_size
            )
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        # Freeze if requested
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass for single batch of images.

        Args:
            images: Batch of images (B, C, H, W)

        Returns:
            Features (B, D) where D is the model's feature dimension
        """
        x = self.model(images)
        x = F.normalize(x, dim=-1)
        return x

    def get_config(self):
        """Get the data config for this model."""
        return timm.data.resolve_model_data_config(self.model)

    def set_grad_checkpointing(self, enable: bool = True):
        """Enable or disable gradient checkpointing."""
        self.model.set_grad_checkpointing(enable)
