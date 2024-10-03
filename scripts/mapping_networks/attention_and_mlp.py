import torch
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from ..utils.model_init import probe_model
from lightning import LightningModule
from torchmetrics import Accuracy, Precision, Recall


class AttentionAndMLP(torch.nn.Module):
    def __init__(self, input_size, num_heads, num_layers, mlp_hiddens, final_size):
        super(AttentionAndMLP, self).__init__()
        self.d_model = input_size
        self.nhead = num_heads
        self.transformer_encoder_layer = TransformerEncoderLayer(
            d_model=input_size,
            nhead=num_heads,
            dim_feedforward=input_size,
            batch_first=True,
        )
        self.transformer_encoder = TransformerEncoder(
            self.transformer_encoder_layer, num_layers=num_layers
        )

        self.mlp = probe_model(
            input_dim=input_size,
            output_dim=final_size,
            hidden_dims=mlp_hiddens,
            non_linearity="relu",
        )

    def forward(self, input_embed, attention_mask=None):
        input_embed = input_embed / input_embed.shape[-1]
        x = self.transformer_encoder(
            src=input_embed, src_key_padding_mask=attention_mask
        )  # batch_size, seq_len, input_size
        # check if x has nan
        # if torch.isnan(x).any():
        #     print("nan in x")
        #     print(x)
        #     print(input_embed)
        #     print(attention_mask)
        #     exit(1)

        x = x / x.shape[-1]
        # x = x.masked_fill(torch.isnan(x), 0)
        x = self.mlp(x[:, 0, :])  # batch_size, final_size
        return x


class AttentionAndMLPLightning(LightningModule):
    def __init__(
        self,
        input_size,
        num_heads,
        num_layers,
        mlp_hiddens,
        final_size,
        model_normed: bool = False,
    ):
        super(AttentionAndMLPLightning, self).__init__()
        self.model = AttentionAndMLP(
            input_size, num_heads, num_layers, mlp_hiddens, final_size
        )

        self.model_normed = model_normed

    def forward(self, input_embed, attention_mask=None):
        return self.model(input_embed=input_embed, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
        x, y = batch["input_embed"], batch["image_embed"]

        if self.model_normed:
            # normalize y
            y = y / y.norm(dim=-1, keepdim=True)

        attention_mask = batch["attention_mask"]
        y_hat = self.model(input_embed=x, attention_mask=attention_mask)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def _eval_step(self, batch, batch_idx):
        x, y = batch["input_embed"], batch["image_embed"]
        if self.model_normed:
            # normalize y
            y = y / y.norm(dim=-1, keepdim=True)

        attention_mask = batch["attention_mask"]
        y_hat = self.model(input_embed=x, attention_mask=attention_mask)
        loss = torch.nn.functional.mse_loss(y_hat, y)

        # check contrastive loss

        # pair-wise dot product between y and y_hat
        dot_product = torch.matmul(y, y_hat.T)  # batch_size, batch_size

        # softmax on both dimensions
        dot_product_x = torch.nn.functional.softmax(
            dot_product, dim=-1
        )  # batch_size, batch_size
        dot_product_y = torch.nn.functional.softmax(
            dot_product, dim=-2
        )  # batch_size, batch_size

        # calculate kl divergence
        kl_div_x = torch.nn.functional.kl_div(
            dot_product_x.log(),
            torch.eye(dot_product_x.shape[0], device=dot_product_x.device),
            reduction="batchmean",
        )
        kl_div_y = torch.nn.functional.kl_div(
            dot_product_y.log(),
            torch.eye(dot_product_y.shape[0], device=dot_product_y.device),
            reduction="batchmean",
        )

        contrastive_x_preds = torch.argmax(dot_product_x, dim=-1)
        contrastive_y_preds = torch.argmax(dot_product_y, dim=-1)

        # contrastive accuracy
        contrastive_accuracy_x = torch.mean(
            torch.eq(
                contrastive_x_preds, torch.arange(y.shape[0], device=y.device)
            ).float()
        )
        contrastive_accuracy_y = torch.mean(
            torch.eq(
                contrastive_y_preds, torch.arange(y.shape[0], device=y.device)
            ).float()
        )

        # self.log("contrastive_accuracy_x", contrastive_accuracy_x)
        # self.log("contrastive_accuracy_y", contrastive_accuracy_y)

        return loss, kl_div_x, kl_div_y, contrastive_accuracy_x, contrastive_accuracy_y

    def validation_step(self, batch, batch_idx):
        loss, kl_div_x, kl_div_y, contrastive_accuracy_x, contrastive_accuracy_y = (
            self._eval_step(batch, batch_idx)
        )
        self.log("val_loss", loss)
        self.log("val_kl_div_x", kl_div_x)
        self.log("val_kl_div_y", kl_div_y)
        self.log("val_contrastive_accuracy_x", contrastive_accuracy_x)
        self.log("val_contrastive_accuracy_y", contrastive_accuracy_y)
        return loss

    def test_step(self, batch, batch_idx):
        loss, kl_div_x, kl_div_y, contrastive_accuracy_x, contrastive_accuracy_y = (
            self._eval_step(batch, batch_idx)
        )
        self.log("test_kl_div_x", kl_div_x)
        self.log("test_kl_div_y", kl_div_y)
        self.log("test_loss", loss)
        self.log("test_contrastive_accuracy_x", contrastive_accuracy_x)
        self.log("test_contrastive_accuracy_y", contrastive_accuracy_y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
