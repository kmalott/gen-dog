import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb
from tqdm import tqdm
import torchvision
import lpips

from .tokenizer import BSQTokenizer
from .data import load_data_loader, load_data

def train(exp_dir: str = "logs",
    model_name: str = "BSQTokenizer",
    num_epoch: int = 5,
    lr: float = 1e-3,
    batch_size: int = 1024,
    seed: int = 2024,
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

    # model = load_model(model_name, **kwargs)
    model = BSQTokenizer(patch_size=2, latent_dim=128, codebook=10)
    model = model.to(device)
    model.train()

    # load data loaders
    # train_data, val_data = load_data_loader()
    # alternatively run: (if dataloaders haven't been made yet)
    train_data, val_data = load_data('./rawdata/Images/', batch_size=batch_size)

    # create loss functions and optimizer
    mse_loss = torch.nn.MSELoss()
    lpips_loss = lpips.LPIPS(net='vgg')
    # gan_loss = ...
    # entropy_loss = ...
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)

    lpips_loss.to(device)

    global_step = 0
    metrics = {"train_loss": [], "val_loss": [], "train_lpips": [], "val_lpips": []}

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        for key in metrics:
            metrics[key].clear()

        model.train()
        train_loss = torch.tensor([0.0])
        val_loss = torch.tensor([0.0])
        train_lpips = torch.tensor([0.0])
        val_lpips = torch.tensor([0.0])

        for img, label in tqdm(train_data):
            # img = img.float() / 255.0 - 0.5
            img, label = img.to(device), label.to(device)
            img_hat = model(img)
            loss_val = mse_loss(img_hat, img)
            lpips_val = lpips_loss(img_hat, img)
            train_loss += loss_val.item()
            train_lpips += lpips_val.sum().item()
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1
        metrics["train_loss"].append(train_loss)
        metrics["train_lpips"].append(train_lpips)

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            for img, label in tqdm(val_data):
                img, label = img.to(device), label.to(device)
                img_hat = model(img)
                loss_val = mse_loss(img_hat, img)
                lpips_val = lpips_loss(img_hat, img)
                val_loss += loss_val.item()
                val_lpips += lpips_val.sum().item()
            metrics["val_loss"].append(val_loss)
            metrics["val_lpips"].append(val_lpips)

        # log average train and val accuracy to tensorboard
        epoch_train_loss = torch.as_tensor(metrics["train_loss"])
        epoch_val_loss = torch.as_tensor(metrics["val_loss"])
        logger.add_scalar('train_loss', epoch_train_loss, global_step)
        logger.add_scalar('val_loss', epoch_val_loss, global_step)

        # add last of the reconstructed images to tensorboard
        grid = torchvision.utils.make_grid(img)
        logger.add_image('images', grid, global_step)
        grid = torchvision.utils.make_grid(img_hat)
        logger.add_image('images_reconstructed', grid, global_step)

        # print on first, last, every 10th epoch
        #if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
            f"train_loss={epoch_train_loss} "
            f"val_loss={epoch_val_loss} "
            f"train_lpips={train_lpips} "
            f"val_lpips={val_lpips} "
        )

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--num_epoch", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--batch_size", type=int, default=1024)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
