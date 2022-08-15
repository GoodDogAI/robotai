import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms

from multiprocessing import freeze_support
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
import wandb


class LitAutoEncoder(pl.LightningModule):
	def __init__(self, lr=1e-3):
		super().__init__()

		self.encoder = nn.Sequential(
			nn.Linear(28 * 28, 64),
			nn.ReLU(),
			nn.Linear(64, 3))

		self.decoder = nn.Sequential(
			nn.Linear(3, 64),
			nn.ReLU(),
			nn.Linear(64, 28 * 28))

		self.lr = lr

	def forward(self, x):
		embedding = self.encoder(x)
		return embedding

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
		return optimizer

	def training_step(self, train_batch, batch_idx):
		x, y = train_batch
		x = x.view(x.size(0), -1)
		z = self.encoder(x)    
		x_hat = self.decoder(z)
		loss = F.mse_loss(x_hat, x)
		self.log('train_loss', loss)
		return loss

	def validation_step(self, val_batch, batch_idx):
		x, y = val_batch
		x = x.view(x.size(0), -1)
		z = self.encoder(x)
		x_hat = self.decoder(z)
		loss = F.mse_loss(x_hat, x)

		self.log('val_loss', loss)

		if batch_idx == 0 and isinstance(self.logger, WandbLogger):
			images = wandb.Image(x_hat.reshape((-1, 28, 28))[0], caption="xhat from validation")
			self.logger.experiment.log({"examples": images, "global_step": self.trainer.global_step})


sweep_config = {
	"method": "bayes",
	"metric": {  # We want to minimize `val_loss`
		"name": "val_loss",
		"goal": "minimize"
	},
	"parameters": {
		"epoch": {
			"distribution": "constant",
			"value": 10,
		},
		"lr": {
			# log uniform distribution between exp(min) and exp(max)
			"distribution": "log_uniform",
			"min": -9.21,  # exp(-9.21) = 1e-4
			"max": -4.61  # exp(-4.61) = 1e-2
		},
		"batch_size": {
			"distribution": "q_log_uniform_values",
			"min": 16,
			"max": 2048,
		},
	}
}


def sweep_iteration():
	rootdir = os.path.join(os.path.dirname(__name__), "..", "..", "_train_logs")

	# set up W&B logger
	wandb.init(dir=rootdir)  # required to have access to `wandb.config`
	wandb_logger = WandbLogger(log_model='all')  # log final model

	# data
	dataset = MNIST(rootdir, train=True, download=True, transform=transforms.ToTensor())
	mnist_train, mnist_val = random_split(dataset, [55000, 5000])

	train_loader = DataLoader(mnist_train, batch_size=wandb.config.batch_size)
	val_loader = DataLoader(mnist_val, batch_size=64)

	trainer = pl.Trainer(gpus=1, max_epochs=wandb.config.epoch, logger=wandb_logger, default_root_dir=rootdir)

	model = LitAutoEncoder(wandb.config.lr)

	# train
	trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == '__main__':
	sweep_id = wandb.sweep(sweep_config, project="mnist-autoencoder-test1")
	wandb.agent(sweep_id, function=sweep_iteration)

# # data
	# dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
	# mnist_train, mnist_val = random_split(dataset, [55000, 5000])
	#
	# train_loader = DataLoader(mnist_train, batch_size=64)
	# val_loader = DataLoader(mnist_val, batch_size=64)
	#
	# # model
	#
	#
	# # training
	# wandb_logger = WandbLogger(project="mnist-autoencoder-test1", log_model="all")
	#
	# trainer = pl.Trainer(accelerator="gpu", devices=1, logger=wandb_logger, max_epochs=5)
	# trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)