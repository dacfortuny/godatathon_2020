import pytorch_lightning as pl
import torch
from src.model.nets import Seq2Seq
from src.metrics.loss import custom_metric


class RNNModel(pl.LightningModule):

    def __init__(self, input_dim, hidden_dim, num_layers, lr=1e-2):
        super().__init__()
        self.lr = lr

        self.save_hyperparameters()

        self.model = Seq2Seq(input_dim, hidden_dim, num_layers)
        # self.loss_fc = torch.nn.MSELoss()
        self.loss_fc = custom_metric

    def forward(self, temp_features, num_features, cat_features, y):
        temp_features = temp_features.permute(1, 0, 2)
        y = y.permute(1, 0, 2)

        # num_features = num_features.permute(1, 0)
        cat_features = cat_features.permute(1, 0)

        return self.model(temp_features, num_features, cat_features, y)

    def training_step(self, batch, batch_idx):
        # Unpack batch
        encoder_temp_features = batch["encoder_temp_features"]
        encoder_num_features = batch["encoder_num_features"]
        encoder_cat_features = batch["encoder_cat_features"]

        decoder_temp_features = batch["decoder_temp_features"]

        y = batch["y_norm"]
        avg_12_volume = batch["avg_12_volume"]
        max_volume = batch["max_volume"]

        # Predict
        y_hat = self(encoder_temp_features,
                     encoder_num_features,
                     encoder_cat_features,
                     y)

        # Flatten
        y_hat = y_hat.flatten()
        y = y.flatten()

        loss = self.loss_fc(actuals=y,
                            forecast=y_hat,
                            max_volume=max_volume,
                            avg_volume=avg_12_volume)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        encoder_temp_features = batch["encoder_temp_features"]
        encoder_num_features = batch["encoder_num_features"]
        encoder_cat_features = batch["encoder_cat_features"]

        decoder_temp_features = batch["decoder_temp_features"]

        y = batch["y_norm"]
        avg_12_volume = batch["avg_12_volume"]
        max_volume = batch["max_volume"]

        # Predict
        y_hat = self(encoder_temp_features,
                     encoder_num_features,
                     encoder_cat_features,
                     y)

        # Flatten
        y_hat = y_hat.flatten()
        y = y.flatten()

        loss = self.loss_fc(actuals=y,
                            forecast=y_hat,
                            max_volume=max_volume,
                            avg_volume=avg_12_volume)

        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
