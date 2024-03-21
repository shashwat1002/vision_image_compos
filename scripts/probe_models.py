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
        self.model_q = probe_model(
            input_dim, output_dim, hidden_dims, non_linearity
        )  # d -> d'
        self.model_k = probe_model(
            input_dim, output_dim, hidden_dims, non_linearity
        )  # d -> d' d' < d

    def forward(self, x):
        """
        x (b, s, d) -> b: batch_size, s: sequence length, d: dimensions from the model
        d -> input_dim on the probe
        """
        transformed_q = self.model_q(x)
        transformed_k = self.model_k(x)

        # get pairwise dot product
        b, s, d = transformed_q.shape
        transformed_k_transpose = transformed_k.transpose(-1, -2)
        pairwise = torch.matmul(transformed_q, transformed_k_transpose)  # b, s, s
        # normalize
        pairwise = pairwise / torch.sqrt(torch.tensor(d))

        # softmax across the sequence
        pairwise = pairwise.view(b, s, s)
        # pairwise = torch.nn.functional.softmax(pairwise, dim=-1)
        print(pairwise.shape)
        return pairwise


class ProbeWordPairLabelModelLightning(LightningModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list,
        non_linearity: str,
        lr: float = 1e-3,
    ):
        super(ProbeWordPairLabelModelLightning, self).__init__()
        self.model = ProbeWordPairLabel(
            input_dim, output_dim, hidden_dims, non_linearity
        )
        self.output_dim = output_dim

        self.loss = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX_IN_LOSS)
        self.lr = lr
        print("hi")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        print("t")
        x, y = batch
        y_hat = self.model(x)
        # print(y_hat)
        print(y.shape)
        b, s1, s2 = y_hat.shape
        loss = self.loss(y_hat.view(-1, s2), y.view(-1))
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def uas(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        b, s1, s2 = y_hat.shape
        loss = self.loss(y_hat.view(-1, s2), y.view(-1))
        y_hat_pred = torch.argmax(y_hat, dim=-1)
        # print(y_hat_pred.shape)

        mask = y != IGNORE_INDEX_IN_LOSS

        total = mask.sum(dim=-1)

        correct = ((y_hat_pred == y) * mask).sum(dim=-1)
        proportion = correct / total
        proportion_avg = proportion.mean()


        return {"loss": loss, "uas": proportion_avg}

    def validation_step(self, batch, batch_idx):
        losses = self.uas(batch, batch_idx)
        val_loss = losses["loss"]
        val_uas = losses["uas"]
        self.log("val_loss", val_loss)
        self.log("val_uas", val_uas)
        return {"val_loss": val_loss, "val_uas": val_uas}

    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    #     return {"val_loss": avg_loss}

    def test_step(self, batch, batch_idx):
        losses = self.uas(batch, batch_idx)
        test_loss = losses["loss"]
        test_uas = losses["uas"]
        self.log("test_loss", test_loss)
        self.log("test_uas", test_uas)
        return {"test_loss": test_loss, "test_uas": test_uas}
