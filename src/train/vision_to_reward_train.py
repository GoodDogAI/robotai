import os
import torch
import wandb
import numpy as np
import pyarrow as pa
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import torch.nn as nn
import torch.nn.functional as F
from src.config.config import MODEL_CONFIGS, HOST_CONFIG
from src.train.videodataset import VideoFrameDataset
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
        intermediate = train_batch["intermediate"][0]
        reward = train_batch["reward"]

        x_hat = self(intermediate)

        loss = F.mse_loss(x_hat, reward)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        intermediate = val_batch["intermediate"][0]
        reward = val_batch["reward"]

        x_hat = self(intermediate)

        loss = F.mse_loss(x_hat, reward)
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
    trainer = pl.Trainer(gpus=1, amp_level="O2", amp_backend="apex", default_root_dir=rootdir, logger=wandb_logger, val_check_interval=1.0, accumulate_grad_batches=1)

    with load_vision_model(model_fullname(MODEL_CONFIGS["yolov7-tiny-s53"])) as intermediate_engine, \
        load_vision_model(model_fullname(MODEL_CONFIGS["yolov7-tiny-prioritize_centered_nms"])) as reward_engine:

        ds = VideoFrameDataset(base_path=HOST_CONFIG.RECORD_DIR)
        ds.download_and_prepare()

        def mapfn(example):
            feed = {
                "y": np.expand_dims(example["y"], 0),
                "uv": np.expand_dims(example["uv"], 0),
            }

            intermediates = intermediate_engine.infer(feed, copy_outputs_to_host=True)
            rewards = reward_engine.infer(feed, copy_outputs_to_host=True)
            
            return {
                "intermediate": intermediates["intermediate"],
                "reward": rewards["reward"],
            }
    

        ds = ds.as_dataset()
        print(ds)

        ds = ds.with_format("numpy").map(mapfn, writer_batch_size=1, remove_columns=["y", "uv"], cache_file_names={"train": "train-mapped.arrow", "validation": "validation-mapped.arrow"})
        ds = ds.with_format("torch")    

        for x in ds["validation"]:
            print(x)

        train_loader = DataLoader(dataset=ds["train"], batch_size=64, shuffle=True)
        valid_loader = DataLoader(dataset=ds["validation"], batch_size=64, shuffle=True)


        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
