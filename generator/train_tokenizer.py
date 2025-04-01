import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb
from tqdm import tqdm
import torchvision
from lpips import LPIPS

from .tokenizer import BSQTokenizer
from .discriminator import Discriminator
from .data import load_data_loader, load_data

def calc_gradient_penalty(netD, real_data, fake_data, device):
    alpha = torch.rand(real_data.shape[0], 1, 1, 1)
    alpha = alpha.to(device)
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates.to(device)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates)
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size(), device=device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

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

    tokenizer = BSQTokenizer(patch_size=2, latent_dim=128, codebook=14)
    tokenizer = tokenizer.to(device)
    discriminator = Discriminator()
    discriminator = discriminator.to(device)

    # load data loaders
    # train_data, val_data = load_data_loader()
    # alternatively run: (if dataloaders haven't been made yet)
    train_data, val_data = load_data('./rawdata/Images/', batch_size=batch_size)

    # create loss functions and optimizer
    mse_loss = torch.nn.MSELoss()
    lpips_loss = LPIPS(net='vgg')
    lpips_loss = lpips_loss.to(device)
    # bce_loss = torch.nn.BCEWithLogitsLoss()
    # entropy_loss = ...
    optimizer_t = torch.optim.AdamW(params=tokenizer.parameters(), lr=lr)
    optimizer_d = torch.optim.AdamW(params=discriminator.parameters(), lr=0.0001, betas=(0.0, 0.9))


    global_step = 0
    metrics = {"train_loss": [], "val_loss": [], 
               "train_bce": [], "val_bce": [], 
               "train_mse": [], "val_mse": [], 
               "train_lpips": [], "val_lpips": [],
               "train_disc": [], "val_disc": []
               }
    
    # warmup loop
    i = 0
    tokenizer.train()
    for img, label in tqdm(train_data):
        img, label = img.to(device), label.to(device)
        img_hat = tokenizer(img)
        mse = mse_loss(img_hat, img)
        lpips = lpips_loss(img_hat, img)
        total_loss_t = 5*mse + 0.1*lpips.sum()
        optimizer_t.zero_grad()
        total_loss_t.backward()
        optimizer_t.step()
        i += 1

    # log imgs after warmup
    grid = torchvision.utils.make_grid(img)
    logger.add_image('images', grid, global_step)
    grid = torchvision.utils.make_grid(img_hat)
    logger.add_image('images_reconstructed', grid, global_step)

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
        train_disc = torch.tensor([0.0])
        val_disc = torch.tensor([0.0])

        train_fake = torch.tensor([0.0])
        train_real = torch.tensor([0.0])
        train_gp = torch.tensor([0.0])

        i = 0
        for img, label in tqdm(train_data):
            img, label = img.to(device), label.to(device)
            # train discriminator
            img_hat = tokenizer(img).detach()
            gan_fake = discriminator(img_hat).mean()
            gan_real = discriminator(img).mean()
            gp = calc_gradient_penalty(discriminator, img, img_hat, device)
            total_loss_d = gan_fake - gan_real + (10*gp) 
            optimizer_d.zero_grad()
            total_loss_d.backward()
            optimizer_d.step()

            if i % 5 == 0:
                # train tokenizer (generator)
                img_hat = tokenizer(img)
                mse = mse_loss(img_hat, img)
                lpips = lpips_loss(img_hat, img)
                gan = discriminator(img_hat).mean()
                total_loss_t = (10*mse) - (0.1*gan) + (0.5*lpips.sum())
                optimizer_t.zero_grad()
                total_loss_t.backward()
                optimizer_t.step()
                train_loss += total_loss_t.item()
                train_mse += mse.item() * 10
                train_bce += gan.item() * 0.1
                train_lpips += lpips.sum().item() * 0.5
            train_disc += total_loss_d.item()
            train_fake += gan_fake.item()
            train_real += gan_real.item()
            train_gp += gp.item()
           
            global_step += 1
            i += 1
            # if i % 100 == 0 and epoch == num_epoch - 1:
            #     print((cnt == 0).float().mean().detach())
            #     print((cnt <= 2).float().mean().detach())
            #     print(cnt.min())
            #     print(cnt.max())
            #     print(cnt.sum())
            #     print(cnt.shape)
        metrics["train_loss"].append(train_loss)
        metrics["train_bce"].append(train_bce)
        metrics["train_lpips"].append(train_lpips)
        metrics["train_mse"].append(train_mse)
        metrics["train_disc"].append(train_disc)

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            tokenizer.eval()
            discriminator.eval()

            for img, label in tqdm(val_data):
                img, label = img.to(device), label.to(device)
                # validate discriminator
                # img_hat, cnt = tokenizer(img)
                img_hat = tokenizer(img)
                gan_fake = discriminator(img_hat).mean()
                gan_real = discriminator(img).mean()
                total_loss_d = gan_fake - gan_real

                # validate tokenizer (generator)
                mse = mse_loss(img_hat, img)
                lpips = lpips_loss(img_hat, img)
                total_loss_t = (10*mse) - (0.1*gan_fake) + (0.5*lpips.sum())
                
                # store losses
                val_loss += total_loss_t.item()
                val_bce += gan_fake.item() * 0.1
                val_mse += mse.item() * 10
                val_lpips += lpips.sum().item() * 0.5
                val_disc += total_loss_d.item()
            metrics["val_loss"].append(val_loss)
            metrics["val_bce"].append(val_bce)
            metrics["val_lpips"].append(val_lpips)
            metrics["val_mse"].append(val_mse)
            metrics["val_disc"].append(val_disc)

        # log average train and val accuracy to tensorboard
        epoch_train_loss = torch.as_tensor(metrics["train_loss"])
        epoch_val_loss = torch.as_tensor(metrics["val_loss"])
        logger.add_scalar('train_loss', epoch_train_loss, global_step)
        logger.add_scalar('val_loss', epoch_val_loss, global_step)

        # add last of the reconstructed images to tensorboard
        grid = torchvision.utils.make_grid()
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
            f"train_disc={train_disc} \n"
            f"val_disc={val_disc} \n"
        )
        print(
            f"Real Img Loss: {train_real} \n"
            f"Fake Img Loss: {train_fake} \n"
            f"Gradient Penalty: {train_gp}"
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
