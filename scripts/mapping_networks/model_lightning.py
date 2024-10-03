from diffusers import PriorTransformer
from lightning import LightningModule
import torch 


class TextToDinoDiffusion(LightningModule):
    def __init__(self, **hparams):
        super().__init__()
        self.hparams = hparams
        self.diffuser = PriorTransformer(**hparams)

    def forward(self, x):
        return self.diffuser(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        b, seq_len, d_text = x.size() # text embedding size
        _, d_image = y.size() # image embedding size

        

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        loss = self.diffuser(x)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.diffuser.parameters(), lr=self.hparams.lr)