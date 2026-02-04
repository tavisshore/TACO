import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
from timm.data import create_transform, resolve_data_config


class ConvNeXtEncoder(pl.LightningModule):
    def __init__(
        self,
        pretrained=True,
        model_name="timm/convnextv2_tiny.fcmae_ft_in22k_in1k_384",
        img_size=384,
        freeze=False,
        lr=1e-4,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.img_size = img_size

        # Two separate branches with no shared weights
        if "vit" in model_name:
            self.branch1 = timm.create_model(
                model_name, pretrained=pretrained, num_classes=0, img_size=img_size
            )
            self.branch2 = timm.create_model(
                model_name, pretrained=pretrained, num_classes=0, img_size=img_size
            )
        else:
            self.branch1 = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
            self.branch2 = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        config = resolve_data_config({}, model=self.branch1)
        self.train_transform = create_transform(**config, is_training=True)
        self.eval_transform = create_transform(**config, is_training=False)

        # PyTorch's autograd produces weight grads for depthwise convs with
        # stride (C, 1, kH, 1) — the size-1 dim gets stride 1 instead of
        # matching the param's (C, C*kH, kH, 1).  Both are "contiguous" per
        # PyTorch's rules, but DDP does a strict stride comparison against
        # its bucket view and warns.  Register a backward hook on every
        # depthwise conv weight to reshape the grad to match the param stride.
        for branch in (self.branch1, self.branch2):
            for module in branch.modules():
                if isinstance(module, nn.Conv2d) and module.groups == module.in_channels:
                    module.weight.register_hook(lambda g, p=module.weight: g.reshape(p.shape))
        ########## TO DORT OUT multi-GPU warning

        if freeze:
            for param in self.branch1.parameters():
                param.requires_grad = False
            for param in self.branch2.parameters():
                param.requires_grad = False

        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

        # Example input for model summary
        self.example_input_array = (
            torch.randn(1, 3, img_size, img_size),
            torch.randn(1, 3, img_size, img_size),
        )

    def get_config(self):
        return timm.data.resolve_model_data_config(self.branch1)

    def set_grad_checkpointing(self, enable=True):
        self.branch1.set_grad_checkpointing(enable)
        self.branch2.set_grad_checkpointing(enable)

    def forward(self, img1, img2):
        feat1 = self.branch1(img1)
        feat2 = self.branch2(img2)
        return feat1, feat2

    def training_step(self, batch, batch_idx):
        img1, img2, labels = batch
        feat1, feat2 = self(img1, img2)
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
        raise NotImplementedError("Implement your loss function")
