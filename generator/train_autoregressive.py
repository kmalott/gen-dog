import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb
import torch.nn.functional as F

from tqdm import tqdm
import math

from .autoregressive import AutoregressiveModel
from .data import TokenDataset

def train(exp_dir: str = "logs",
    model_name: str = "AutoRegressive",
    num_epoch: int = 5,
    lr: float = 1e-3,
    batch_size: int = 1024,
    seed: int = 2024,
    codebook: int = 14,
    latent: int = 1024,
    d_model: int = 512,
    nhead: int = 8,
    nlayer: int = 4,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    autoregressive = AutoregressiveModel(latent, d_model, codebook, nhead, nlayer)
    autoregressive.to(device)
    # load data loaders
    train_token = TokenDataset("train")
    val_token = TokenDataset("val")
    train_data = torch.utils.data.DataLoader(train_token, batch_size=batch_size, num_workers=4, shuffle=True)
    val_data = torch.utils.data.DataLoader(val_token, batch_size=batch_size, num_workers=4, shuffle=False)
    # create optimizer
    optimizer = torch.optim.AdamW(params=autoregressive.parameters(), lr=lr)

    global_step = 0
    # training loop
    for epoch in range(num_epoch):
        autoregressive.train()
        # reset losses
        train_loss = torch.tensor([0.0])
        val_loss = torch.tensor([0.0])
        for x in tqdm(train_data):
            x = x.squeeze(1).to(device)
            x_hat = autoregressive(x)
            loss = (
                    F.cross_entropy(x_hat.reshape(-1, x_hat.shape[-1]), x.reshape(-1), reduction="sum")
                    / math.log(2)
                    / x.shape[0]
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            global_step += 1
        train_loss /= len(train_token)

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            autoregressive.eval()
            for x in tqdm(val_data):
                x = x.squeeze(1).to(device)
                x_hat = autoregressive(x)
                loss = (
                    F.cross_entropy(x_hat.reshape(-1, x_hat.shape[-1]), x.reshape(-1), reduction="sum")
                    / math.log(2)
                    / x.shape[0]
                )
                val_loss += loss.item()
            val_loss /= len(val_token)

        # log average train and val accuracy to tensorboard
        logger.add_scalar('train_loss', train_loss, global_step)
        logger.add_scalar('val_loss', val_loss, global_step)

        print(
            f"Epoch {epoch + 1:2d} / {num_epoch:2d}: \n"
            f"train_loss={train_loss} \n"
            f"val_loss={val_loss} \n"
        )

    # save a copy of model weights in the log directory
    torch.save(autoregressive, log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--num_epoch", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--codebook", type=int, default=14)
    parser.add_argument("--latent", type=int, default=1024)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--nlayer", type=int, default=4)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
