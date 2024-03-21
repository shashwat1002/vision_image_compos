from .utils.model_init import probe_model
from torch.nn import Module
import torch
from torchmetrics.functional import precision, recall


from lightning import LightningModule

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
        x (b, d) -> b: batch_size, d: dimensions from the model
        d -> input_dim on the probe
        """
        return self.model(x)


class ProbeModelWordLabelLightning(LightningModule):

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
        self.output_dim = output_dim

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

        # get top pred
        loss = self.loss(y_hat, y)
        y_hat = torch.argmax(y_hat, dim=-1)
        self.log("val_loss", loss)

        # get metrics
        precision_score = precision(
            task="multiclass",
            preds=y_hat,
            target=y,
            num_classes=self.output_dim,
            ignore_index=IGNORE_INDEX_IN_LOSS,
        )
        recall_score = recall(
            task="multiclass",
            preds=y_hat,
            target=y,
            num_classes=self.output_dim,
            ignore_index=IGNORE_INDEX_IN_LOSS,
        )
        self.log("val_precision", precision_score)
        self.log("val_recall", recall_score)
        return {
            "val_loss": loss,
            "val_precision": precision_score,
            "val_recall": recall_score,
        }

    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    #     return {"val_loss": avg_loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        # get top pred
        y_hat = torch.argmax(y_hat, dim=-1)
        self.log("test_loss", loss)
        loss = self.loss(y_hat, y)

        # get metrics
        precision_score = precision(y_hat, y)
        recall_score = recall(y_hat, y)
        self.log("test_precision", precision_score)
        self.log("test_recall", recall_score)
        return {
            "test_loss": loss,
            "test_precision": precision_score,
            "test_recall": recall_score,
        }

    def on_test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        return {"test_loss": avg_loss}


class ProbeWordPairLabel(Module):
    """
    Defines a probe that is based on bilinear transformations
    """

    def __init__(
        self, input_dim: int, output_dim: int, hidden_dims: list, non_linearity: str
    ):
        super(ProbeWordPairLabel, self).__init__()
        self.model = probe_model(input_dim, output_dim, hidden_dims, non_linearity)

    def forward(self, x):
        """
        x (b, s, d) -> b: batch_size, s: sequence length, d: dimensions from the model
        d -> input_dim on the probe
        """
        transformed = self.model(x)

        # get pairwise dot product
        b, s, d = x.shape
        transformed = transformed.view(b, s, 1, d)
        x = x.view(b, s, d, 1)
        pairwise = torch.matmul(transformed, x)

        # softmax across the sequence
        pairwise = pairwise.view(b, s, s)
        pairwise = pairwise + 1e-6
        pairwise = torch.nn.functional.softmax(pairwise, dim=-1)

        return pairwise


