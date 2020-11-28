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

    def forward(self, x, y):
        x = x.permute(1, 0, 2)
        y = y.permute(1, 0, 2)

        return self.model(x, y)

    def training_step(self, batch, batch_idx):
        # Unpack batch
        x = batch["x"]
        y = batch["y_norm"]
        avg_12_volume = batch["avg_12_volume"]
        max_volume = batch["max_volume"]

        # Predict
        y_hat = self(x, y)

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
        x = batch["x"]
        y = batch["y_norm"]
        avg_12_volume = batch["avg_12_volume"]
        max_volume = batch["max_volume"]

        # Predict
        y_hat = self(x, y)

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
