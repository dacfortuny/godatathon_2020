import pytorch_lightning as pl
import torch
from src.model.nets import Seq2Seq


class RNNModel(pl.LightningModule):

    def __init__(self, input_dim, hidden_dim, num_layers, lr=1e-2):
        super().__init__()
        self.lr = lr

        self.save_hyperparameters()

        self.model = Seq2Seq(input_dim, hidden_dim, num_layers)
        self.loss_fc = torch.nn.MSELoss()

    def forward(self, x, y):
        x = x.permute(1, 0, 2)
        y = y.permute(1, 0, 2)

        return self.model(x, y)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x, y)

        loss = self.loss_fc(y_hat.flatten(), y.flatten())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x, y)

        loss = self.loss_fc(y_hat.flatten(), y.flatten())
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
