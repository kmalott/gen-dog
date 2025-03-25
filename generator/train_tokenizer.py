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
from .discriminator import Discriminator
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

    tokenizer = BSQTokenizer(patch_size=2, latent_dim=128, codebook=10)
    tokenizer = tokenizer.to(device)
    tokenizer.train()
    discriminator = Discriminator()
    discriminator = discriminator.to(device)
    discriminator.train()

    # load data loaders
    # train_data, val_data = load_data_loader()
    # alternatively run: (if dataloaders haven't been made yet)
    train_data, val_data = load_data('./rawdata/Images/', batch_size=batch_size)

    # create loss functions and optimizer
    mse_loss = torch.nn.MSELoss()
    lpips_loss = lpips.LPIPS(net='vgg')
    lpips_loss.to(device)
    d_loss = torch.nn.BCEWithLogitsLoss()
    # entropy_loss = ...
    optimizer_t = torch.optim.AdamW(params=tokenizer.parameters(), lr=lr)
    optimizer_d = torch.optim.AdamW(params=discriminator.parameters(), lr=lr)


    global_step = 0
    metrics = {"train_loss": [], "val_loss": [], 
               "train_bce": [], "val_bce": [], 
               "train_mse": [], "val_mse": [], 
               "train_lpips": [], "val_lpips": []
               }

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        for key in metrics:
            metrics[key].clear()

        tokenizer.train()
        discriminator.train()

        train_loss = torch.tensor([0.0])
        val_loss = torch.tensor([0.0])
        train_bce = torch.tensor([0.0])
        val_bce = torch.tensor([0.0])

        train_mse = torch.tensor([0.0])
        val_mse = torch.tensor([0.0])
        train_lpips = torch.tensor([0.0])
        val_lpips = torch.tensor([0.0])

        for img, label in tqdm(train_data):
            # img = img.float() / 255.0 - 0.5
            img, label = img.to(device), label.to(device)
            # train discriminator
            img_hat = tokenizer(img)
            bce_fake = d_loss(discriminator(img_hat), torch.zeros((batch_size, 1)))
            bce_real = d_loss(discriminator(img), torch.ones((batch_size, 1)))
            total_loss_d = bce_fake + bce_real
            optimizer_d.zero_grad()
            total_loss_d.backward()
            optimizer_d.step()

            # train tokenizer (generator)
            mse = mse_loss(img_hat, img)
            lpips = lpips_loss(img_hat, img)
            total_loss_t = mse + (0.001*lpips.sum())
            optimizer_t.zero_grad()
            total_loss_t.backward()
            optimizer_t.step()

            # store losses
            train_bce += total_loss_d.item()
            train_loss += total_loss_t.item()
            train_mse += mse.item()
            train_lpips += lpips.sum().item()
            global_step += 1
        metrics["train_loss"].append(train_loss)
        metrics["train_bce"].append(train_bce)
        metrics["train_lpips"].append(train_lpips)
        metrics["train_mse"].append(train_mse)

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            tokenizer.eval()
            discriminator.eval()

            for img, label in tqdm(val_data):
                # img = img.float() / 255.0 - 0.5
                img, label = img.to(device), label.to(device)
                # validate discriminator
                img_hat = tokenizer(img)
                bce_fake = d_loss(discriminator(img_hat), torch.zeros((batch_size, 1)))
                bce_real = d_loss(discriminator(img), torch.ones((batch_size, 1)))
                total_loss_d = bce_fake + bce_real

                # validate tokenizer (generator)
                mse = mse_loss(img_hat, img)
                lpips = lpips_loss(img_hat, img)
                total_loss_t = mse + (0.001*lpips.sum())
                
                # store losses
                val_bce += total_loss_d.item()
                val_loss += total_loss_t.item()
                val_mse += mse.item()
                val_lpips += lpips.sum().item()
            metrics["val_loss"].append(val_loss)
            metrics["val_bce"].append(val_bce)
            metrics["val_lpips"].append(val_lpips)
            metrics["val_mse"].append(val_mse)

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
            f"Epoch {epoch + 1:2d} / {num_epoch:2d}: \n"
            f"train_loss={epoch_train_loss} \n"
            f"val_loss={epoch_val_loss} \n"
            f"train_mse={train_mse} \n"
            f"val_mse={val_mse} \n"
            f"train_lpips={train_lpips} \n"
            f"val_lpips={val_lpips} \n"
            f"train_bce={train_bce} \n"
            f"val_bce={val_bce} \n"
            
        )

    # save a copy of model weights in the log directory
    torch.save(tokenizer.state_dict(), log_dir / f"{model_name}.th")
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
