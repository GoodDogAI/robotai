import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import wandb
from pytorch_lightning.loggers import WandbLogger

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))


class Encoder(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.chs = chs
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs

    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)

        return x


class UNet(pl.LightningModule):
    def __init__(self, enc_chs=[2,64,128,256,512,1024], dec_chs=[1024, 512, 256, 128, 64], lr=1e-3):
        super().__init__()

        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head = nn.Conv2d(dec_chs[-1], enc_chs[0], 1) # Do the final upsampling to the original input number of channels
        self.crop = torchvision.transforms.CenterCrop([532,1092]) # TODO UNhardcode these
        self.lr = lr

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x = train_batch
        z = self.encoder(x)
        x_hat = self.decoder(z[::-1][0], z[::-1][1:])
        x_hat = self.head(x_hat)

        loss = F.mse_loss(x_hat, self.crop(x))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch
        z = self.encoder(x)
        x_hat = self.decoder(z[::-1][0], z[::-1][1:])
        x_hat = self.head(x_hat)
        loss = F.mse_loss(x_hat, self.crop(x))

        self.log('val_loss', loss)

        if batch_idx == 0 and isinstance(self.logger, WandbLogger):
            images = wandb.Image(x_hat[0, 0], caption="xhat from validation")
            self.logger.experiment.log({"examples": images, "global_step": self.trainer.global_step})