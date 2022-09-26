import os
import torch
import wandb

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import torch.nn as nn
import torch.nn.functional as F
from src.config.config import MODEL_CONFIGS
from src.train.videoloader import build_datapipe
from src.train.modelloader import load_vision_model, model_fullname
from torch.utils.data import DataLoader

class SimpleNet(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
       
        self.net = nn.Sequential(
            nn.Linear(17003, 1000),
            nn.ReLU(),
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
        )
        self.lr = lr

    def forward(self, intermediate):
        return self.net(intermediate)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        intermediates, rewards = train_batch

        x_hat = self(intermediates)

        loss = F.mse_loss(x_hat, rewards)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        intermediates, rewards = val_batch

        x_hat = self(intermediates)

        loss = F.mse_loss(x_hat, rewards)
        self.log('val_loss', loss)
        return loss

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True


rootdir = os.path.join(os.path.dirname(__file__), "..", "..", "_pretrain_logs")
model = SimpleNet(lr=1e-3)

if __name__ == '__main__':
    wandb.init(dir=rootdir, project="vision_to_reward_1")
    wandb_logger = WandbLogger()
    trainer = pl.Trainer(gpus=1, amp_level="O2", amp_backend="apex", max_epochs=4, default_root_dir=rootdir, logger=wandb_logger, val_check_interval=100, limit_val_batches=10, accumulate_grad_batches=1)

    with load_vision_model(model_fullname(MODEL_CONFIGS["yolov7-tiny-s53"])) as intermediate_engine, \
        load_vision_model(model_fullname(MODEL_CONFIGS["yolov7-tiny-prioritize_centered_nms"])) as reward_engine:

        train_datapipe = build_datapipe(train_or_valid="train", split=0.9).calculate_intermediate_and_reward(intermediate_engine=intermediate_engine, reward_engine=reward_engine)
        valid_datapipe = build_datapipe(train_or_valid="valid", split=0.9).calculate_intermediate_and_reward(intermediate_engine=intermediate_engine, reward_engine=reward_engine)

        train_datapipe = train_datapipe.shuffle(buffer_size=1000)

        train_loader = DataLoader(dataset=train_datapipe, batch_size=8)
        valid_loader = DataLoader(dataset=valid_datapipe, batch_size=8)

        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
