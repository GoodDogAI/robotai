import os
import pytorch_lightning as pl
import torch
import wandb

from src.train.model_vae import VanillaVAE
from src.train.videoloader import build_datapipe
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

train_datapipe = build_datapipe(train_or_valid="train", split=0.9)
valid_datapipe = build_datapipe(train_or_valid="valid", split=0.9)

train_loader = DataLoader(dataset=train_datapipe, batch_size=4)
valid_loader = DataLoader(dataset=valid_datapipe, batch_size=4)

rootdir = os.path.join(os.path.dirname(__file__), "..", "..", "_pretrain_logs")

model = VanillaVAE(in_channels=2, latent_dim=64, hidden_dims = [32, 64, 64, 64, 128, 128, 128, 128, 128])

if __name__ == '__main__':
    wandb.init(dir=rootdir, project="vae_video1")
    wandb_logger = WandbLogger()
    trainer = pl.Trainer(gpus=1, max_epochs=1, default_root_dir=rootdir, logger=wandb_logger, val_check_interval=100, limit_val_batches=10)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
