from utils.model_init import probe_model
from torch.nn import Module
import torch


from lightning import LightningModel

IGNORE_INDEX_IN_LOSS = -1


class ProbeModelWordLabel(Module):
    """
    Defines a probe class to deal with tasks that are of the form where each word gets a label
    Uses the probe definition models to get the actual layers
    """

    def __init__(
        self, input_dim: int, output_dim: int, hidden_dims: list, non_linearity: str
    ):
        super(ProbeModelWordLabel, self).__init__()
        self.model = probe_model(input_dim, output_dim, hidden_dims, non_linearity)

    def forward(self, x):
        """
        x (b, s, d) -> b: batch_size, s: squence_length, d: dimensions from the model
        d -> input_dim on the probe
        """
        return self.model(x)


class ProbeModelWordLabelLightning(LightningModel):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list,
        non_linearity: str,
        lr: float = 1e-3,
    ):
        super(ProbeModelWordLabelLightning, self).__init__()
        self.model = ProbeModelWordLabel(
            input_dim, output_dim, hidden_dims, non_linearity
        )
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX_IN_LOSS)
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        return {"val_loss": avg_loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("test_loss", loss)
        return {"test_loss": loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        return {"test_loss": avg_loss}
