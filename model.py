from torch import optim, nn, utils, Tensor
import torch.nn.functional as F
import pytorch_lightning as pl

EARLY_STOP_CONDITION = False

class Encoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim)
        )

    def forward(self, x):
        return self.l1(x)


class Decoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim)
        )

    def forward(self, x):
        return self.l1(x)


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, in_dim=28*28, hid1_dim=64, inner_dim=3, hid2_dim=64, out_dim=28*28, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.encoder = Encoder(in_dim=in_dim, hid_dim=hid1_dim, out_dim=inner_dim)
        self.decoder = Decoder(in_dim=inner_dim, hid_dim=hid2_dim, out_dim=out_dim)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        
        logger = self.logger.experiment # this can actually happen in any function or hoook
        # also, multpiple loggers can be used through list and indexing
        ...
        
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        x_hat = self.decoder(self.encoder(x))
        test_loss = F.mse_loss(x_hat, x)
        self.log('test_loss', test_loss)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        x_hat = self.decoder(self.encoder(x))
        val_loss = F.mse_loss(x_hat, x)
        self.log('val_loss', val_loss)

    def on_train_batch_start(self, batch, batch_idx):
        # runs before each batch, can be used for early stopping epoch
        if EARLY_STOP_CONDITION:
            return -1

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer 


class CIFAR10Classifier(pl.LightningModule):
    def __init__(self, fe_checkpoint_path):
        super().__init__()
        self.feature_extractor = LitAutoEncoder.load_from_checkpoint(fe_checkpoint_path) # basically transfer learning with pretrained model
        self.feature_extractor.freeze()

        self.classifier = nn.Linear(100, 10)

    def forward(self, x):
        repr = self.feature_extractor(x)
        x = self.classifier(repr)
        return x
