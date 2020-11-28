import pytorch_lightning as pl
import torch
from src.model.nets import Seq2Seq
from src.metrics.loss import custom_metric, uncertainty_metric


class RNNModel(pl.LightningModule):

    def __init__(self, input_dim, hidden_dim, num_layers, lr=1e-2):
        super().__init__()
        self.lr = lr

        self.save_hyperparameters()

        self.model = Seq2Seq(input_dim, hidden_dim, num_layers)

        self.prediction_loss_fc = custom_metric
        self.confidence_loss_fc = uncertainty_metric

    def forward(self, temp_features, num_features, cat_features, y):
        return self.model(temp_features, num_features, cat_features, y)

    def _base_run(self, batch):
        # Unpack batch
        encoder_temp_features = batch["encoder_temp_features"]
        encoder_num_features = batch["encoder_num_features"]
        encoder_cat_features = batch["encoder_cat_features"]
        decoder_temp_features = batch["decoder_temp_features"]
        y = batch["y_norm"]
        avg_12_volume = batch["avg_12_volume"]
        max_volume = batch["max_volume"]

        # Permute arrays
        encoder_temp_features = encoder_temp_features.permute(1, 0, 2)
        y = y.permute(1, 0, 2)

        # encoder_num_features = encoder_num_features.permute(1, 0)
        encoder_cat_features = encoder_cat_features.permute(1, 0)

        # Predict
        y_hat = self(encoder_temp_features,
                     encoder_num_features,
                     encoder_cat_features,
                     y)

        # Loss
        pred_loss = self.prediction_loss_fc(actuals=y,
                                            forecast=y_hat["prediction"],
                                            max_volume=max_volume,
                                            avg_volume=avg_12_volume)
        confidence_loss = self.confidence_loss_fc(actuals=y,
                                                  upper_bound=y_hat["upper_bound"],
                                                  lower_bound=y_hat["lower_bound"],
                                                  max_volume=max_volume,
                                                  avg_volume=avg_12_volume)

        loss = (0.5 * pred_loss + (1 - 0.5) * confidence_loss)

        return loss, pred_loss, confidence_loss

    def training_step(self, batch, batch_idx):
        loss, pred_loss, confidence_loss = self._base_run(batch)
        self.log('train/loss', loss)
        self.log('train/prediction_loss', pred_loss)
        self.log('train/confidence_loss', confidence_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred_loss, confidence_loss = self._base_run(batch)
        self.log('val/loss', loss)
        self.log('val/prediction_loss', pred_loss)
        self.log('val/confidence_loss', confidence_loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        lambda1 = lambda epoch: 0.9 ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=lambda1)

        return [optimizer], [scheduler]
