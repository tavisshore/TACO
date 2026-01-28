import time

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm


class TimmModel(nn.Module):
    def __init__(self, model_name, pretrained=True, img_size=383):
        super(TimmModel, self).__init__()

        self.img_size = img_size

        if "vit" in model_name:
            # automatically change interpolate pos-encoding to img_size
            self.model = timm.create_model(
                model_name, pretrained=pretrained, num_classes=0, img_size=img_size
            )
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def get_config(
        self,
    ):
        data_config = timm.data.resolve_model_data_config(self.model)
        return data_config

    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

    def forward(self, img1, img2=None):
        if img2 is not None:
            image_features1 = self.model(img1)
            image_features2 = self.model(img2)

            return image_features1, image_features2

        else:
            image_features = self.model(img1)

            return image_features


class Sample4GeoEncoder(nn.Module):
    """Wrapper for TimmModel to use as encoder in ImageRetrievalModel.

    This adapter class makes TimmModel compatible with the encoder interface
    expected by ImageRetrievalModel, which requires a single-image forward pass.

    Args:
        model_name: Name of the timm model (e.g., 'resnet50', 'vit_base_patch16_224')
        pretrained: Whether to load pretrained weights
        img_size: Input image size (important for ViT models)
        freeze: If True, freeze all encoder parameters
    """

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
