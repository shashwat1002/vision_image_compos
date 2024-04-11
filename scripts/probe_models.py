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
        loss = self.loss(y_hat, y)
        self.log("test_loss", loss)

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


class ProbeSimilarityOrdering(Module):
    """
    Dealing with probing tasks that involve comparing two pairs of distances.
    For instance, we have v_1, c_1, c_2 and we want to say d(v_1, c_1) > d(v_1, c_2).
    We phrase this as a classification problem.
    v_1 -> f_v(v_1)
    c_1 -> f_c(c_1)
    c_2 -> f_c(c_2)
    we check Softmax(f(v_1)^T f(c_1), f(v_1)^T f(c_2)) and loss against 1, 0
    """

    def __init__(
        self,
        input_dim_c,
        input_dim_v,
        output_dim,
        hidden_dims,
        non_linearity,
        symmetric: bool = False,
    ):
        super(ProbeSimilarityOrdering, self).__init__()
        print(f"input_dim_c: {input_dim_c}, input_dim_v: {input_dim_v}")
        print(f"output_dim: {output_dim}, hidden_dims: {hidden_dims}")
        print(f"non_linearity: {non_linearity}, symmetric: {symmetric}")

        if symmetric:
            self.model_v = probe_model(
                input_dim_v, output_dim, hidden_dims, non_linearity
            )
            self.model_c = self.model_v
        else:
            self.model_v = probe_model(
                input_dim_v, output_dim, hidden_dims, non_linearity
            )
            self.model_c = probe_model(
                input_dim_c, output_dim, hidden_dims, non_linearity
            )

    def forward(self, x_dict):
        """
        x_dict -> has "text" and "image". either (b,1,d) (b,2,d) or (b,2,d) (b,1,d)
        d -> input_dim on the probe
        """

        text_embeds = x_dict["text"]
        image_embeds = x_dict["image"]

        text_transformed = self.model_c(text_embeds)
        image_transformed = self.model_v(image_embeds)

        if text_transformed.shape[1] == 1:
            one = text_transformed
            two_1 = image_transformed[:, [0], :]
            two_2 = image_transformed[:, [1], :]
        else:
            one = image_transformed
            two_1 = text_transformed[:, [0], :]
            two_2 = text_transformed[:, [1], :]

        d = text_transformed.shape[-1]

        # get pairwise dot product
        pairwise1 = (
            torch.matmul(one, two_1.transpose(-1, -2)).squeeze(dim=-1) / d
        )  # b, 1
        pairwise2 = (
            torch.matmul(one, two_2.transpose(-1, -2)).squeeze(dim=-1) / d
        )  # b, 1
        print("pairwise batch", pairwise1.shape, pairwise2.shape)
        print(torch.cat([pairwise1, pairwise2], dim=-1).shape)

        return torch.cat([pairwise1, pairwise2], dim=-1)  # b, 2


class ProbeSimilarityOrderingWinnogroundStyle(LightningModule):
    def __init__(
        self,
        input_dim_c,
        input_dim_v,
        output_dim,
        hidden_dims,
        non_linearity,
        symmetric: bool = False,
        lr: float = 1e-5,
    ):
        super(ProbeSimilarityOrderingWinnogroundStyle, self).__init__()
        self.model = ProbeSimilarityOrdering(
            input_dim_c=input_dim_c,
            input_dim_v=input_dim_v,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            non_linearity=non_linearity,
            symmetric=symmetric,
        )
        self.output_dim = output_dim

        self.loss = torch.nn.CrossEntropyLoss()
        self.lr = lr

    def image_input_arrangement(self, batch, batch_idx):
        """
        Assuming batch: (b, 4, d)
        where the first 2 are captions and the last two images

        returns Tup[(b, 3, d), (b, 3, d)]

        The function will return batch equivalent of the following
        (c_0, i_0, i_1), (c_1, i_1, i_0)

        The gold truth will always be 1, 0
        """
        text_embeds, image_embeds = batch
        comb1 = {
            "text": text_embeds[:, [0], :],
            "image": image_embeds[:, [0, 1], :],
        }
        comb2 = {
            "text": text_embeds[:, [1], :],
            "image": image_embeds[:, [1, 0], :],
        }
        return comb1, comb2

    def text_input_arrangement(self, batch, batch_idx):
        """
        Assuming batch: (b, 4, d)
        where the first 2 are captions and the last two images

        returns Tup[(b, 3, d), (b, 3, d)]

        The function will return batch equivalent of the following
        (i_0, c_0, c_1), (i_1, c_0, c_1)

        The gold truth will always be 1, 0
        """

        text_embeds, image_embeds = batch
        comb1 = {
            "image": image_embeds[:, [0], :],
            "text": text_embeds[:, [0, 1], :],
        }
        comb2 = {
            "image": image_embeds[:, [1], :],
            "text": text_embeds[:, [1, 0], :],
        }
        return comb1, comb2

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch  # (b, 2, dc), (b, 2, di)
        b = batch[0].shape[0]
        device = batch[0].device
        print(b, device)
        # gold truth is always 0
        y = torch.zeros(b, device=device).long()  # (b,)

        # get the text arrangement of the task
        comb1, comb2 = self.text_input_arrangement(x, batch_idx)
        y_hat_text_1, y_hat_text_2 = self.model(comb1), self.model(
            comb2
        )  # (b, 2), (b, 2)
        text_loss_11, text_loss_12 = self.loss(
            y_hat_text_1.view(-1, 2), y.view(-1)
        ), self.loss(y_hat_text_2.view(-1, 2), y.view(-1))

        # get the image arrangement of the task
        comb1, comb2 = self.image_input_arrangement(x, batch_idx)
        y_hat_image_1, y_hat_image_2 = self.model(comb1), self.model(comb2)
        image_loss_11, image_loss_12 = self.loss(
            y_hat_image_1.view(-1, 2), y.view(-1)
        ), self.loss(y_hat_image_2.view(-1, 2), y.view(-1))

        self.log("train_text_loss", text_loss_11 + text_loss_12)
        self.log("train_image_loss", image_loss_11 + image_loss_12)
        self.log(
            "train_loss", text_loss_11 + text_loss_12 + image_loss_11 + image_loss_12
        )
        return text_loss_12 + text_loss_11 + image_loss_11 + image_loss_12

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def _evaluation_code(self, batch, batch_idx):
        x = batch
        b = batch[0].shape[0]
        device = batch[0].device
        y = torch.zeros(b, device=device).long()  # (b, 1)

        # get the text arrangement of the task
        comb1, comb2 = self.text_input_arrangement(x, batch_idx)
        y_hat_text_1, y_hat_text_2 = self.model(comb1), self.model(comb2)
        print("y_hat shape", y_hat_text_1.shape, y_hat_text_2.shape)
        text_accuracy_1 = torch.argmax(y_hat_text_1, dim=-1) == y
        text_accuracy_2 = torch.argmax(y_hat_text_2, dim=-1) == y
        print(text_accuracy_1.shape)
        text_accuracy_fr = (
            torch.logical_and(text_accuracy_1, text_accuracy_2).float().mean()
        )
        text_loss = self.loss(y_hat_text_1.view(-1, 2), y.view(-1)) + self.loss(
            y_hat_text_2.view(-1, 2), y.view(-1)
        )

        # get the image arrangement of the task
        comb1, comb2 = self.image_input_arrangement(x, batch_idx)
        y_hat_image_1, y_hat_image_2 = self.model(comb1), self.model(comb2)
        image_accuracy_1 = torch.argmax(y_hat_image_1, dim=-1) == y
        image_accuracy_2 = torch.argmax(y_hat_image_2, dim=-1) == y
        image_accuracy_fr = (
            torch.logical_and(image_accuracy_1, image_accuracy_2).float().mean()
        )
        image_loss = self.loss(y_hat_image_1.view(-1, 2), y.view(-1)) + self.loss(
            y_hat_image_2.view(-1, 2), y.view(-1)
        )

        group_pred_accuracy = (
            torch.logical_and(
                torch.logical_and(
                    text_accuracy_1,
                    text_accuracy_2,
                ),
                torch.logical_and(image_accuracy_1, image_accuracy_2),
            )
            .float()
            .mean()
        )

        return {
            "text_accuracy": text_accuracy_fr,
            "image_accuracy": image_accuracy_fr,
            "group_accuracy": group_pred_accuracy,
            "loss": image_loss + text_loss,
        }

    def validation_step(self, batch, batch_idx):
        print("validation ye")
        text_accuracy = self._evaluation_code(batch, batch_idx)["text_accuracy"]
        image_accuracy = self._evaluation_code(batch, batch_idx)["image_accuracy"]
        group_accuracy = self._evaluation_code(batch, batch_idx)["group_accuracy"]
        val_loss = self._evaluation_code(batch, batch_idx)["loss"]

        self.log("val_text_accuracy", text_accuracy)
        self.log("val_image_accuracy", image_accuracy)
        self.log("val_group_accuracy", group_accuracy)
        self.log("val_loss", val_loss)

        return {
            "val_text_accuracy": text_accuracy,
            "val_image_accuracy": image_accuracy,
            "val_group_accuracy": group_accuracy,
            "val_loss": val_loss,
        }

    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    #     return {"val_loss": avg_loss}

    def test_step(self, batch, batch_idx):
        text_accuracy = self._evaluation_code(batch, batch_idx)["text_accuracy"]
        image_accuracy = self._evaluation_code(batch, batch_idx)["image_accuracy"]
        group_accuracy = self._evaluation_code(batch, batch_idx)["group_accuracy"]
        test_loss = self._evaluation_code(batch, batch_idx)["loss"]

        self.log("test_text_accuracy", text_accuracy)
        self.log("test_image_accuracy", image_accuracy)
        self.log("test_group_accuracy", group_accuracy)
        self.log("test_loss", test_loss)

        return {
            "test_text_accuracy": text_accuracy,
            "test_image_accuracy": image_accuracy,
            "test_group_accuracy": group_accuracy,
            "test_loss": test_loss,
        }
